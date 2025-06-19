import logging
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import interp1d
from Gait_Phase.gpu_utils import configure_gpu

def run_nmf_channel_reduction_pipeline():
    # Configure GPU
    configure_gpu()

    # For PyTorch operations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #  GPU details
    if device.type == 'cuda':
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")
        torch.cuda.empty_cache()

    # -------------------------------
    # Setup logging
    # -------------------------------
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    # -------------------------------
    # Step 1:Define STRIDE intervals (no transition intervals)
    # -------------------------------
    STRIDE_INTERVALS = [
        (89108, 92245), (92246, 95175), (95176, 97447),
        (97448, 100575), (100576, 103623), (103624, 106831),
        (106832, 109945), (109946, 112821), (112822, 115905),
        (115906, 119045), (119046, 122367), (122368, 126179),
        (126180, 128417), (128418, 131367), (131368, 135647),
        (135648, 137665), (137666, 141871), (141872, 145001),
        (145002, 147199), (147200, 150373), (150374, 153445),
        (153446, 156627), (156628, 159901)
    ]
    
    def label_frames(frame_array):
        """
        Label frames based on predefined stride intervals.
        Frames within any stride interval are labeled as 1,
        and frames outside these intervals (including beyond 159901) are labeled as 3.
        """
        labels = np.full(frame_array.shape, 3)
        for start, end in STRIDE_INTERVALS:
            condition = (frame_array >= start) & (frame_array <= end)
            labels[condition] = 1
        return labels
    
    # --------------------------------------------------
    # Step 2:Compute Explained Variance
    # --------------------------------------------------
    def compute_explained_variance(X_full, X_reconstructed):
        total_variance = np.sum((X_full - np.mean(X_full, axis=0)) ** 2)
        residual_variance = np.sum((X_full - X_reconstructed) ** 2)
        explained_variance_ratio = 1 - residual_variance / total_variance
        return explained_variance_ratio * 100
    
    # --------------------------------------------------
    # Step 3:Post-processing: Spline Interpolation
    # --------------------------------------------------
    def post_process_activations(activations):
        processed = activations.copy()
        n_frames = processed.shape[0]
        x_full = np.arange(n_frames)
        for j in range(processed.shape[1]):
            col = processed[:, j]
            mask = np.isnan(col) | (col == 0)
            if np.sum(~mask) < 2:
                logging.warning(f"Not enough valid data for spline interpolation in synergy {j}.")
                continue
            interp_func = interp1d(x_full[~mask], col[~mask], kind='cubic', fill_value="extrapolate")
            col[mask] = interp_func(x_full[mask])
            processed[:, j] = col
        return np.abs(processed)
    
    # --------------------------------------------------
    # Step 4:GPU-based NMF using Multiplicative Updates
    # --------------------------------------------------
    def torch_nmf(X, n_synergies=5, max_iter=1000, tol=1e-4, device='cuda'):
        eps = 1e-8
        n_frames, n_features = X.shape
        W = torch.rand((n_frames, n_synergies), device=device)
        H = torch.rand((n_synergies, n_features), device=device)
        for i in range(max_iter):
            # Update H
            numerator = torch.mm(W.t(), X)
            denominator = torch.mm(torch.mm(W.t(), W), H) + eps
            H = H * (numerator / denominator)
            # Update W
            numerator = torch.mm(X, H.t())
            denominator = torch.mm(W, torch.mm(H, H.t())) + eps
            W = W * (numerator / denominator)
            if i % 50 == 0:
                X_approx = torch.mm(W, H)
                error = torch.norm(X - X_approx, p='fro')
                logging.info(f"NMF iter {i}, error: {error.item():.6f}")
                if error < tol:
                    logging.info(f"NMF converged at iter {i} with error {error.item():.4f}")
                    break
        return W, H
    
    # --------------------------------------------------
    # Step 5:Neural Network Mapping from Reduced EMG to Synergy Activations
    # --------------------------------------------------
    class EnhancedMappingNet(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(EnhancedMappingNet, self).__init__()
            self.fc = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, output_dim),
                nn.ReLU()  # Ensured nonnegative outputs
            )
    
        def forward(self, x):
            return self.fc(x)
    
    def train_enhanced_mapping_net(X_reduced_np, S_full_np, max_epochs=20000, lr=1e-3, device='cuda'):
        # Convert numpy arrays to torch tensors
        X_reduced = torch.from_numpy(X_reduced_np).float().to(device)
        S_full = torch.from_numpy(S_full_np).float().to(device)
        input_dim = X_reduced.shape[1]
        output_dim = S_full.shape[1]
    
        # Initialize the enhanced mapping network
        model = EnhancedMappingNet(input_dim, output_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # Scheduler to reduce learning rate if loss plateaus
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1000)
    
        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            S_pred = model(X_reduced)
            loss = criterion(S_pred, S_full)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            if epoch % 100 == 0:
                logging.info(f"EnhancedMappingNet Epoch {epoch}, Loss: {loss.item():.6f}")
        return model
    
    # --------------------------------------------------
    # Step 6:Channel Selection Methods
    # --------------------------------------------------
    def anatomical_selection(block2_indices):
        """
        Method 1: Anatomical mapping.
        """
        # predetermined VL channels
        anatomical_indices = [170, 171, 172, 173, 174]
        return anatomical_indices
    
    def variance_based_selection(X_full, block2_indices, top_k=10):
        """
        Method 2: Preliminary analysis based on variance.
        Computes the variance of each channel (from block2_indices) and selects the top_k channels.
        """
        variances = np.var(X_full[:, block2_indices], axis=0)
        # Get indices of top_k channels (relative to block2)
        top_indices_local = np.argsort(variances)[-top_k:]
        # Map local indices back to absolute indices.
        selected_indices = [block2_indices[i] for i in top_indices_local]
        return selected_indices
    
    def l1_feature_selection(X_reduced, synergy_activations, top_k=10, device='cuda'):
        """
        Method 3: Feature selection using a simple linear model with L1 regularization.
        Trains a linear mapping from the full block2 data to synergy activations,
        then selects channels with the highest average absolute weight values.
        """
        n_samples, n_features = X_reduced.shape
        n_outputs = synergy_activations.shape[1]
        model = nn.Linear(n_features, n_outputs).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        X_tensor = torch.from_numpy(X_reduced).float().to(device)
        Y_tensor = torch.from_numpy(synergy_activations).float().to(device)
        epochs = 500
        l1_lambda = 1e-3
    
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, Y_tensor)
            # Add L1 regularization on weights
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l1_lambda * l1_norm
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            weights = model.weight.cpu().numpy()  # shape: (n_outputs, n_features)
        # Average absolute weight across outputs for each channel.
        avg_abs_weights = np.mean(np.abs(weights), axis=0)  # shape: (n_features,)
        # Select the top_k channels (indices relative to the block2 subset)
        top_indices_local = np.argsort(avg_abs_weights)[-top_k:]
        return top_indices_local
    # --------------------------------------------------
    # Step 7: Load EMG data
    # --------------------------------------------------
    emg_file = "C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\emg_all_samples_features.csv"
    if not os.path.exists(emg_file):
        logging.error(f"EMG file not found at {emg_file}")
        exit()
    
    try:
        emg_df = pd.read_csv(emg_file)
    except Exception as e:
        logging.error(f"Error loading file: {e}")
        exit()
    
    logging.info(f"EMG data loaded with shape: {emg_df.shape}")
    # --------------------------------------------------
    # Step 8: Ensure required columns exist
    # --------------------------------------------------
    if "Frame" not in emg_df.columns:
        emg_df.insert(0, "Frame", np.arange(len(emg_df)))
    if "Time (Seconds)" not in emg_df.columns:
        logging.error("Missing 'Time (Seconds)' column in EMG data.")
        exit()
    # --------------------------------------------------
    # Step 9: Label Frames based on stride intervals
    # --------------------------------------------------
    frame_indices = emg_df["Frame"].values
    emg_df["Label"] = label_frames(frame_indices)
    logging.info("Frame labeling complete.")
    logging.info(f"Label distribution: {emg_df['Label'].value_counts().to_dict()}")
    # --------------------------------------------------
    # Step 10: Filter to only include frames within stride intervals (Label 1)
    # --------------------------------------------------
    emg_df = emg_df[emg_df["Label"] == 1]
    logging.info(f"Filtered EMG data shape (only stride intervals): {emg_df.shape}")
    # --------------------------------------------------
    # Step 11: Extract EMG Data from Blocks
    # --------------------------------------------------
    # The remaining columns are the EMG features.
    emg_data = emg_df.iloc[:, 2:-1]
    n_frames, n_features = emg_data.shape
    logging.info(f"Extracted EMG data with shape: {emg_data.shape}")
    
    # Rename columns for clarity and fill missing data
    emg_data.columns = [f"EMG_{i+1}" for i in range(n_features)]
    emg_data = emg_data.fillna(0)
    
    # Rectify EMG signals and convert to float32
    X_full = np.abs(emg_data.values).astype(np.float32)
    logging.info(f"Rectified full EMG data shape: {X_full.shape}")

    TARGET_VAR = 0.90  # 90 %
    MAX_SYNERGIES = 15  # safety cap
    logging.info(f"Using device: {device}")
    X_full_torch = torch.from_numpy(X_full).to(device)

    variance_summary = []

    for n_synergies in range(1, MAX_SYNERGIES + 1):
        W_torch, H_torch = torch_nmf(X_full_torch,
                                     n_synergies=n_synergies,
                                     max_iter=20000,
                                     device=device)
        synergy_activations = W_torch.cpu().numpy()
        synergy_weights = H_torch.cpu().numpy()
        X_reconstructed = synergy_activations.dot(synergy_weights)
        explained_percentage = compute_explained_variance(X_full, X_reconstructed)
        logging.info(f"{n_synergies} synergies → {explained_percentage:.2f}% variance explained")
        variance_summary.append((n_synergies, explained_percentage))
        if explained_percentage >= TARGET_VAR * 100:
            logging.info(f"Target reached with {n_synergies} synergies "
                         f"({explained_percentage:.2f}% variance).")
            break
    else:
        logging.warning(f"Reached MAX_SYNERGIES={MAX_SYNERGIES} "
                        f"but variance still < {TARGET_VAR * 100:.0f}%")

    print(f"Explained Variance by {n_synergies} synergies: {explained_percentage:.2f}%")

    df_varsummary = pd.DataFrame(
        variance_summary,
        columns=["n_synergies", "explained_variance_pct"]
    )
    df_varsummary.to_csv(
        "C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\nmf_variance_by_n_synergies.csv",
        index=False
    )
    logging.info("Saved overall variance summary to nmf_variance_by_n_synergies.csv")

    W = W_torch.cpu().numpy()
    H = H_torch.cpu().numpy()

    indiv_ev = []
    cum_ev = []
    for j in range(1, n_synergies + 1):
        Xj = W[:, :j].dot(H[:j, :])
        ev_j = compute_explained_variance(X_full, Xj)
        cum_ev.append(ev_j)
        if j == 1:
            indiv_ev.append(ev_j)
        else:
            indiv_ev.append(ev_j - cum_ev[-2])

    df_per_synergy = pd.DataFrame({
        "Synergy_index": list(range(1, n_synergies + 1)),
        "Individual_explained_pct": indiv_ev,
        "Cumulative_explained_pct": cum_ev
    })
    df_per_synergy.to_csv(
        "C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\nmf_per_synergy_explained_variance.csv",
        index=False
    )
    logging.info("Saved per-synergy explained variance to nmf_per_synergy_explained_variance.csv")

    # --------------------------------------------------
    # Step 12: Post-process Synergy Activations
    # --------------------------------------------------
    synergy_activations_fixed = post_process_activations(synergy_activations)
    pd.DataFrame(synergy_activations_fixed,columns=[f"Synergy_{i + 1}" for i in range(n_synergies)]).to_csv("C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\synergy_activations_p1.csv",index=False)
    # Save synergy activations
    synergy_df = pd.DataFrame(synergy_activations_fixed,
                              columns=[f"Synergy_{i+1}" for i in range(n_synergies)])
    synergy_df["Frame"] = emg_df["Frame"].values
    synergy_df["Time (Seconds)"] = emg_df["Time (Seconds)"].values
    synergy_df["Label"] = emg_df["Label"].values
    synergy_df.to_csv("C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\muscle_synergy_activations.csv", index=False)
    logging.info("Saved synergy activations to muscle_synergy_activations.csv")
    # --------------------------------------------------
    # Step 13: Define Block 2 indices
    # --------------------------------------------------
    #  Block 2 corresponds to columns 160 to 320 in X_full.
    block2_indices = list(range(160, 320))
    X_block2 = X_full[:, block2_indices]

    # --------------------------------------------------
    # Step 14: Channel Selection Method
    # --------------------------------------------------
    # Method 1: Anatomical mapping (predetermined channels)
    anatomical_indices = anatomical_selection(block2_indices)
    method1_indices = anatomical_indices  # absolute indices
    logging.info(f"Method 1 (Anatomical) selected channels: {method1_indices}")
    
    # Method 2: Variance-based selection (top 10 channels by variance)
    method2_indices = variance_based_selection(X_full, block2_indices, top_k=10)
    logging.info(f"Method 2 (Variance-based) selected channels: {method2_indices}")
    
    # Method 3: L1 feature selection (using linear model with L1 regularization)
    # Train on all channels in Block 2 first:
    top_local_indices = l1_feature_selection(X_block2, synergy_activations_fixed, top_k=10, device=device)
    method3_indices = [block2_indices[i] for i in top_local_indices]
    logging.info(f"Method 3 (L1 feature selection) selected channels: {method3_indices}")
    
    # Step 15: Train and Evaluate Enhanced Mapping for Each Method
    methods = {
        "Anatomical": method1_indices,
        "Variance": method2_indices,
        "L1": method3_indices
    }
    results = {}
    
    for method_name, indices in methods.items():
        logging.info(f"Training EnhancedMappingNet using {method_name} selection with channels: {indices}")
        # Extract reduced EMG data using the selected channels.
        X_reduced_method = X_full[:, indices]
        # Train the enhanced mapping network.
        model = train_enhanced_mapping_net(X_reduced_method, synergy_activations_fixed,
                                           max_epochs=20000, lr=1e-3, device=device)
        # Evaluate performance:
        model.eval()
        with torch.no_grad():
            X_reduced_tensor = torch.from_numpy(X_reduced_method).float().to(device)
            S_pred_nn = model(X_reduced_tensor).cpu().numpy()
            pd.DataFrame( S_pred_nn, columns=[f"Synergy_{i + 1}" for i in range(n_synergies)]).to_csv(f"C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\reconstructed_synergies_{method_name}_p1.csv",index=False)

        X_reconstructed_nn = S_pred_nn.dot(synergy_weights)
        explained_percentage_nn = compute_explained_variance(X_full, X_reconstructed_nn)
        logging.info(f"{method_name} selection - Explained Variance (full EMG from reduced set): {explained_percentage_nn:.2f}%")
        print(f"{method_name} selection - Explained Variance (full EMG from reduced set): {explained_percentage_nn:.2f}%")
        results[method_name] = {
            "explained_variance": explained_percentage_nn,
            "selected_indices": indices,
            "model": model,
            "X_reduced": X_reduced_method,
            "S_pred_nn": S_pred_nn
        }
        pd.DataFrame({"Method": [method_name],"Explained Variance": [explained_percentage_nn]
        }).to_csv(f"C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\reconstruction_eval_{method_name}_p1.csv",index=False)
    # --------------------------------------------------
    # Step 16: Select Best Method and Save Reconstructed EMG
    # --------------------------------------------------
    best_method = max(results, key=lambda k: results[k]["explained_variance"])
    best_result = results[best_method]
    logging.info(f"Best method: {best_method} with explained variance {best_result['explained_variance']:.2f}%")
    print(f"Best method: {best_method} with explained variance {best_result['explained_variance']:.2f}%")
    
    df_reconstructed = pd.DataFrame(best_result["S_pred_nn"].dot(synergy_weights),
                                    columns=[f"EMG_{i+1}" for i in range(n_features)])
    df_reconstructed["Frame"] = emg_df["Frame"].values
    df_reconstructed.to_csv("C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\reconstructed_full_emg_from_best_method.csv", index=False)
    logging.info("Saved reconstructed EMG to reconstructed_full_emg_from_best_method.csv")

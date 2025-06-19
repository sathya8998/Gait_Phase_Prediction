import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import numba
import logging
import warnings
import torch.optim as optim
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, roc_auc_score, f1_score)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.utils import compute_class_weight
from Gait_Phase.analysis_utils.model_analysis_utils import FeatureExtractor, UtilityManager, \
    ModelInterpretability
from Gait_Phase.hyperparameter_tuning.HyperParameter_Tuning import hyperparameter_tuning
from Gait_Phase.attention_lstm.Attention_Layer_Bi_Lstm_wrapped import LSTMModelBuilder
from Gait_Phase.cnn_model.CNN_wrapped import CNNModelBuilder
from Gait_Phase.evaluation_utils.evaluation_utils import adjust_pred_proba, prepare_roc_auc, F1Score
from Gait_Phase.transformers.Transformers_wrapped import TransformerModelBuilder
from Gait_Phase.config import CONFIG
from Gait_Phase.gpu_utils import configure_gpu
from datetime import datetime

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)
# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="expandable_segments not supported on this platform")
# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = configure_gpu()

# ----------------------------------------------------------------------------
# PyTorch Callback Equivalents
# ----------------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, monitor='val_loss', patience=3, restore_best_weights=True):
        self.monitor = monitor
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.wait = 0
        self.best_state = None

    def step(self, current_loss, model):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
            if self.restore_best_weights:
                self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            return False
        else:
            self.wait += 1
            if self.wait >= self.patience:
                if self.restore_best_weights and self.best_state is not None:
                    model.load_state_dict(self.best_state)
                return True
            return False


class ReduceLROnPlateau:
    def __init__(self, monitor='val_loss', factor=0.5, patience=1, verbose=1):
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.best_loss = float('inf')

    def step(self, current_loss, optimizer):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                for param_group in optimizer.param_groups:
                    old_lr = param_group['lr']
                    new_lr = old_lr * self.factor
                    param_group['lr'] = new_lr
                    if self.verbose:
                        logger.info(f"Reducing learning rate from {old_lr:.6f} to {new_lr:.6f}")
                self.wait = 0


def train_model(model, train_ds, val_ds, criterion, optimizer, epochs, device, F1Score):
    model = model.to(device)
    batch_size_train = CONFIG["batch_size"] if len(train_ds) >= CONFIG["batch_size"] else len(train_ds)
    batch_size_val = CONFIG["batch_size"] if len(val_ds) >= CONFIG["batch_size"] else len(val_ds)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size_train, shuffle=True,
        drop_last=True if batch_size_train > 1 else False
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size_val, shuffle=False,
        drop_last=True if batch_size_val > 1 else False
    )
    early_stop = EarlyStopping(monitor='val_loss', patience=CONFIG["early_stopping_patience"],
                               restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, verbose=1)

    # Instantiate F1Score on the correct device.
    f1_metric = F1Score(num_classes=model.num_classes, device=device)

    # Initialize the GradScaler for mixed precision
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        model.train()
        f1_metric.reset_states()
        train_loss = 0.0
        train_samples = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()

            # Forward pass using mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(xb)
                loss = criterion(outputs, yb)

            # Update metric using outputs from autocast context
            f1_metric.update_state(yb, outputs)
            train_loss += loss.item() * xb.size(0)
            train_samples += xb.size(0)

            # Backward pass with loss scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        train_loss = train_loss / train_samples
        train_f1 = f1_metric.result()

        model.eval()
        f1_metric.reset_states()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(xb)
                    loss = criterion(outputs, yb)
                f1_metric.update_state(yb, outputs)
                total_loss += loss.item() * xb.size(0)
                total_samples += xb.size(0)

        val_loss = total_loss / total_samples
        val_f1 = f1_metric.result()

        logger.info(f"Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f}, train_f1={train_f1:.4f}, "
                    f"val_loss={val_loss:.4f}, val_f1={val_f1:.4f}")

        reduce_lr.step(val_loss, optimizer)
        if early_stop.step(val_loss, model):
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    return model

# ----------------------------------------------------------------------------
# MAIN PIPELINE
# ----------------------------------------------------------------------------
def main_pipeline():
    # Configure GPU
    configure_gpu()
    logger.info("Starting main pipeline...")
    metrics = {}
    os.makedirs("visualizations", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # 1. Load CSV and Prepare Data
    logger.info("Reading CSV file and preparing data...")
    df = pd.read_csv(
        "C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\final_labels_cleaned_merged.csv")
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
    mask = np.zeros(len(df), dtype=bool)
    for low, high in STRIDE_INTERVALS:
        mask |= (df['Frame'] >= low) & (df['Frame'] <= high)
    df = df[mask]
    logger.info(f"Filtered data shape: {df.shape}")

    X = df[["Synergy_1", "Synergy_2", "Synergy_3", "Synergy_4", "Synergy_5"]].values
    y_raw = df["PredictedStage"].values
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(y_raw)
    num_classes = len(np.unique(labels_encoded))
    logger.info(f"Number of classes: {num_classes}")

    # 2. Create Sequences
    logger.info("Creating input sequences for the model...")
    X_sequences, seq_labels, center_indices = FeatureExtractor.create_sequences(
        X, window_size=CONFIG['window_size'], overlap=CONFIG['overlap'], labels=labels_encoded
    )
    if X_sequences is None or X_sequences.shape[0] < 5:
        logger.error("Insufficient sequences created.")
        return
    logger.info(f"Created input sequences: {X_sequences.shape}")
    try:
        X_aligned = FeatureExtractor.align_sequences_with_dtw(X_sequences)
        logger.info(f"Aligned sequences shape: {X_aligned.shape}")
    except Exception as e:
        logger.error(f"Error during DTW alignment: {e}")
        X_aligned = X_sequences
    X_final = X_aligned

    # 3. Select Samples for Interpretability (LIME/SHAP)
    logger.info("Selecting 3 samples per class for interpretability analyses...")
    X_selected_samples, y_selected_samples = UtilityManager.select_three_samples_per_class(X_final,seq_labels)
    if X_selected_samples is None or y_selected_samples is None or X_selected_samples.size == 0:
        logger.error("No samples selected for interpretability.")
        return
    logger.info(f"Selected samples: {X_selected_samples.shape}, labels: {y_selected_samples.shape}")

    # 4. Train-Test Split
    logger.info("Splitting data into training and testing sets...")
    try:
        X_train_nn, X_test_nn, y_train, y_test = train_test_split(
            X_final, seq_labels, test_size=0.2, stratify=seq_labels, random_state=CONFIG['random_seed']
        )
        logger.info(f"Training set: {X_train_nn.shape}, Testing set: {X_test_nn.shape}")
    except Exception as e:
        logger.error(f"Error during train-test split: {e}")
        return

    # 5. Compute Class Weights
    logger.info("Computing class weights...")
    try:
        classes = np.unique(labels_encoded)
        class_weights = compute_class_weight("balanced", classes=classes, y=labels_encoded)
        class_weight_dict = dict(zip(classes, class_weights))
        logger.info(f"Class weights: {class_weight_dict}")
    except Exception as e:
        logger.error(f"Error computing class weights: {e}")
        class_weight_dict = None

    # 6. Oversample Training Data
    logger.info("Oversampling training data for neural networks...")
    try:
        from collections import Counter
        class_counts = Counter(y_train)
        min_count = min(class_counts.values())
        if min_count > 1:
            n_neighbors = min(5, max(1, min_count - 1))
            logger.info(f"Applying SMOTE with n_neighbors={n_neighbors}")
            sm = SMOTE(random_state=CONFIG['random_seed'], k_neighbors=n_neighbors)
            X_os, y_train_res = sm.fit_resample(X_train_nn.reshape(X_train_nn.shape[0], -1), y_train)
            time_steps = X_train_nn.shape[1]
            n_feats = X_train_nn.shape[2]
            X_os = X_os.reshape(-1, time_steps, n_feats)
        else:
            logger.warning("Using RandomOverSampler instead of SMOTE.")
            from imblearn.over_sampling import RandomOverSampler
            ros = RandomOverSampler(random_state=CONFIG['random_seed'])
            X_os, y_train_res = ros.fit_resample(X_train_nn.reshape(X_train_nn.shape[0], -1), y_train)
            time_steps = X_train_nn.shape[1]
            n_feats = X_train_nn.shape[2]
            X_os = X_os.reshape(-1, time_steps, n_feats)
        logger.info(f"Oversampled data shape: {X_os.shape}, {y_train_res.shape}")
    except Exception as e:
        logger.error(f"Error during oversampling: {e}")
        return

    # 7. Hyperparameter Tuning via Cross-Validation
    logger.info("Starting hyperparameter tuning...")
    tuning_strategies = ["BayesianOptimization", "RandomSearch", "Hyperband"]
    lstm_cv_results = {s: [] for s in tuning_strategies}
    cnn_cv_results = {s: [] for s in tuning_strategies}
    transformer_cv_results = {s: [] for s in tuning_strategies}
    outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=CONFIG["random_seed"])
    logger.info("Tuning LSTM, CNN, Transformer via cross-validation...")
    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X_os, y_train_res), 1):
        logger.info(f"=== Fold {fold} ===")
        X_train_fold = X_os[train_idx]
        X_val_fold = X_os[val_idx]
        y_train_fold = y_train_res[train_idx]
        y_val_fold = y_train_res[val_idx]
        input_shape_model = (X_train_fold.shape[1], X_train_fold.shape[2])
        fold_num_classes = len(np.unique(y_train_fold))

        for strategy in tuning_strategies:
            # LSTM
            try:
                tuner, best_trial = hyperparameter_tuning(
                    model_builder=LSTMModelBuilder(),
                    tuner_type=strategy,
                    X_train=X_train_fold, y_train=y_train_fold,
                    X_val=X_val_fold, y_val=y_val_fold,
                    input_shape=input_shape_model, num_classes=fold_num_classes,
                    fold=fold, strategy=strategy
                )
                if best_trial:
                    hps = best_trial.params
                    model = LSTMModelBuilder().build_lstm_model_final(hps, input_shape_model, fold_num_classes)
                    train_ds = torch.utils.data.TensorDataset(
                        torch.from_numpy(X_train_fold).float().to(device),
                        torch.from_numpy(y_train_fold).long()
                    )
                    val_ds   = torch.utils.data.TensorDataset(
                        torch.from_numpy(X_val_fold).float().to(device),
                        torch.from_numpy(y_val_fold).long()
                    )
                    optimizer = optim.Adam(model.parameters(), lr=hps.get("learning_rate", 1e-3))
                    model = train_model(model, train_ds, val_ds, nn.CrossEntropyLoss(), optimizer, CONFIG["epochs"], device, F1Score)
                    model.eval()
                    with torch.no_grad():
                        out = model(torch.from_numpy(X_val_fold).float().to(device))
                    y_pred = torch.argmax(out, 1).cpu().numpy()
                    proba  = out.cpu().numpy()
                    y_val_bin = label_binarize(y_val_fold, classes=range(fold_num_classes))
                    lstm_cv_results[strategy].append({
                        "Fold": fold,
                        "Accuracy": accuracy_score(y_val_fold, y_pred),
                        "Precision": precision_score(y_val_fold, y_pred, average="weighted", zero_division=0),
                        "Recall": recall_score(y_val_fold, y_pred, average="weighted", zero_division=0),
                        "F1-Score": f1_score(y_val_fold, y_pred, average="weighted", zero_division=0),
                        "ROC_AUC": roc_auc_score(y_val_bin, proba, multi_class="ovr", average="weighted"),
                        "Best_Hyperparameters": hps
                    })
            except Exception as e:
                logger.error(f"Error tuning LSTM with {strategy} on fold {fold}: {e}")

            # CNN
            try:
                tuner, best_trial = hyperparameter_tuning(
                    model_builder=CNNModelBuilder(),
                    tuner_type=strategy,
                    X_train=X_train_fold, y_train=y_train_fold,
                    X_val=X_val_fold, y_val=y_val_fold,
                    input_shape=input_shape_model, num_classes=fold_num_classes,
                    fold=fold, strategy=strategy
                )
                if best_trial:
                    hps = best_trial.params
                    model = CNNModelBuilder().build_cnn_model_final(hps, input_shape_model, fold_num_classes)
                    train_ds = torch.utils.data.TensorDataset(
                        torch.from_numpy(X_train_fold).float().to(device),
                        torch.from_numpy(y_train_fold).long()
                    )
                    val_ds   = torch.utils.data.TensorDataset(
                        torch.from_numpy(X_val_fold).float().to(device),
                        torch.from_numpy(y_val_fold).long()
                    )
                    optimizer = optim.Adam(model.parameters(), lr=hps.get("learning_rate", 1e-3))
                    model = train_model(model, train_ds, val_ds, nn.CrossEntropyLoss(), optimizer, CONFIG["epochs"], device, F1Score)
                    model.eval()
                    with torch.no_grad():
                        out = model(torch.from_numpy(X_val_fold).float().to(device))
                    y_pred = torch.argmax(out, 1).cpu().numpy()
                    proba  = out.cpu().numpy()
                    y_val_bin = label_binarize(y_val_fold, classes=range(fold_num_classes))
                    cnn_cv_results[strategy].append({
                        "Fold": fold,
                        "Accuracy": accuracy_score(y_val_fold, y_pred),
                        "Precision": precision_score(y_val_fold, y_pred, average="weighted", zero_division=0),
                        "Recall": recall_score(y_val_fold, y_pred, average="weighted", zero_division=0),
                        "F1-Score": f1_score(y_val_fold, y_pred, average="weighted", zero_division=0),
                        "ROC_AUC": roc_auc_score(y_val_bin, proba, multi_class="ovr", average="weighted"),
                        "Best_Hyperparameters": hps
                    })
            except Exception as e:
                logger.error(f"Error tuning CNN with {strategy} on fold {fold}: {e}")

            # Transformer
            try:
                tuner, best_trial = hyperparameter_tuning(
                    model_builder=TransformerModelBuilder(),
                    tuner_type=strategy,
                    X_train=X_train_fold, y_train=y_train_fold,
                    X_val=X_val_fold, y_val=y_val_fold,
                    input_shape=input_shape_model, num_classes=fold_num_classes,
                    fold=fold, strategy=strategy
                )
                if best_trial:
                    hps = best_trial.params
                    model = TransformerModelBuilder().build_transformer_model_final(hps, input_shape_model, fold_num_classes)
                    train_ds = torch.utils.data.TensorDataset(
                        torch.from_numpy(X_train_fold).float().to(device),
                        torch.from_numpy(y_train_fold).long()
                    )
                    val_ds   = torch.utils.data.TensorDataset(
                        torch.from_numpy(X_val_fold).float().to(device),
                        torch.from_numpy(y_val_fold).long()
                    )
                    optimizer = optim.Adam(model.parameters(), lr=hps.get("learning_rate", 1e-3))
                    model = train_model(model, train_ds, val_ds, nn.CrossEntropyLoss(), optimizer, CONFIG["epochs"], device, F1Score)
                    model.eval()
                    with torch.no_grad():
                        out = model(torch.from_numpy(X_val_fold).float().to(device))
                    y_pred = torch.argmax(out, 1).cpu().numpy()
                    proba  = out.cpu().numpy()
                    y_val_bin = label_binarize(y_val_fold, classes=range(fold_num_classes))
                    transformer_cv_results[strategy].append({
                        "Fold": fold,
                        "Accuracy": accuracy_score(y_val_fold, y_pred),
                        "Precision": precision_score(y_val_fold, y_pred, average="weighted", zero_division=0),
                        "Recall": recall_score(y_val_fold, y_pred, average="weighted", zero_division=0),
                        "F1-Score": f1_score(y_val_fold, y_pred, average="weighted", zero_division=0),
                        "ROC_AUC": roc_auc_score(y_val_bin, proba, multi_class="ovr", average="weighted"),
                        "Best_Hyperparameters": hps
                    })
            except Exception as e:
                logger.error(f"Error tuning Transformer with {strategy} on fold {fold}: {e}")

    logger.info("Hyperparameter tuning complete.")
    #----------------------------------------------------------------------------
    # 8.Aggregation, Strategy Selection, Visualization, Final Model Retraining, Ensemble & Interpretability
    #----------------------------------------------------------------------------
    logger.info("START: Aggregation of tuning results")

    def aggregate_results(cv_results, model_type):
        logger.info(f"Aggregating results for {model_type} ...")
        for strategy, results in cv_results.items():
            if not results:
                logger.warning(f"No results for {model_type} using {strategy}.")
                continue
            avg_acc = np.mean([r["Accuracy"] for r in results])
            avg_f1 = np.mean([r["F1-Score"] for r in results])
            avg_roc = np.mean([r["ROC_AUC"] for r in results])
            combined = 0.5 * avg_f1 + 0.5 * avg_roc
            logger.info(
                f"{model_type} - {strategy}: Avg Acc={avg_acc:.4f}, Avg F1={avg_f1:.4f}, Avg ROC_AUC={avg_roc:.4f}, Combined={combined:.4f}")

    logger.info("Aggregating LSTM Results:")
    aggregate_results(lstm_cv_results, "LSTM")
    logger.info("Aggregating CNN Results:")
    aggregate_results(cnn_cv_results, "CNN")
    logger.info("Aggregating Transformer Results:")
    aggregate_results(transformer_cv_results, "Transformer")
    logger.info("END: Aggregation of tuning results")

    logger.info("START: Selecting best hyperparameter strategy for each model")
    best_lstm_strategy = None;
    best_cnn_strategy = None;
    best_transformer_strategy = None
    best_lstm_score = -np.inf;
    best_cnn_score = -np.inf;
    best_transformer_score = -np.inf

    for strategy, results in lstm_cv_results.items():
        if results:
            combined = 0.5 * np.mean([r["F1-Score"] for r in results]) + 0.5 * np.mean([r["ROC_AUC"] for r in results])
            if combined > best_lstm_score:
                best_lstm_score = combined
                best_lstm_strategy = strategy
    for strategy, results in cnn_cv_results.items():
        if results:
            combined = 0.5 * np.mean([r["F1-Score"] for r in results]) + 0.5 * np.mean([r["ROC_AUC"] for r in results])
            if combined > best_cnn_score:
                best_cnn_score = combined
                best_cnn_strategy = strategy
    for strategy, results in transformer_cv_results.items():
        if results:
            combined = 0.5 * np.mean([r["F1-Score"] for r in results]) + 0.5 * np.mean([r["ROC_AUC"] for r in results])
            if combined > best_transformer_score:
                best_transformer_score = combined
                best_transformer_strategy = strategy

    logger.info(f"Best LSTM Strategy: {best_lstm_strategy} (score={best_lstm_score:.4f})")
    logger.info(f"Best CNN Strategy: {best_cnn_strategy} (score={best_cnn_score:.4f})")
    logger.info(f"Best Transformer Strategy: {best_transformer_strategy} (score={best_transformer_score:.4f})")
    logger.info("END: Selecting best hyperparameter strategy for each model")
    #----------------------------------------------------------------------------
    # 9.Save aggregated tuning results to CSV
    #----------------------------------------------------------------------------
    aggregated_data = []
    for model_name, cv_results in (
    ("LSTM", lstm_cv_results), ("CNN", cnn_cv_results), ("Transformer", transformer_cv_results)):
        for strategy in ["BayesianOptimization", "RandomSearch", "Hyperband"]:
            results = cv_results.get(strategy, [])
            if results:
                avg_accuracy = np.mean([r["Accuracy"] for r in results])
                avg_f1 = np.mean([r["F1-Score"] for r in results])
                avg_roc = np.mean([r["ROC_AUC"] for r in results])
                combined = 0.5 * avg_f1 + 0.5 * avg_roc
                aggregated_data.append({
                    "Model": model_name,
                    "Strategy": strategy,
                    "Avg_Accuracy": avg_accuracy,
                    "Avg_F1": avg_f1,
                    "Avg_ROC_AUC": avg_roc,
                    "Combined_Score": combined
                })
    df_aggregated = pd.DataFrame(aggregated_data)
    aggregated_csv_path = r"C:\Users\ek23yboj\PycharmProjects\PythonProject\Gait_Phase\Data\aggregated_tuning_results.csv"
    df_aggregated.to_csv(aggregated_csv_path, index=False)
    logger.info(f"Saved aggregated tuning results to {aggregated_csv_path}")

    # Save best strategy info to JSON
    best_strategies = {
        "LSTM": {"Strategy": best_lstm_strategy, "Combined_Score": best_lstm_score},
        "CNN": {"Strategy": best_cnn_strategy, "Combined_Score": best_cnn_score},
        "Transformer": {"Strategy": best_transformer_strategy, "Combined_Score": best_transformer_score}
    }
    best_strategies_path = r"C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\best_strategies.json"
    with open(best_strategies_path, "w") as f:
        json.dump(best_strategies, f, indent=4)
    logger.info(f"Best strategies saved to {best_strategies_path}")
    logger.info(f"Saved best strategies information to {best_strategies_path}")


    logger.info("START: Retraining final models on entire oversampled dataset")
    lstm_models = {};
    cnn_best_model = None;
    transformer_best_model = None
    time_steps = X_os.shape[1]
    n_feats = X_os.shape[2]
    #----------------------------------------------------------------------------
    # 10. Retrain final models on entire oversampled dataset
    #----------------------------------------------------------------------------

    # ----- Retrain LSTM Model -----
    if best_lstm_strategy and lstm_cv_results[best_lstm_strategy]:
        try:
            best_lstm_fold = max(lstm_cv_results[best_lstm_strategy], key=lambda r: r["F1-Score"])
            best_hps_dict = best_lstm_fold["Best_Hyperparameters"]
        except Exception as e:
            logger.error("Error extracting best hyperparameters for LSTM.", exc_info=True)
            best_hps_dict = {}
        logger.info(f"LSTM best hyperparameters: {best_hps_dict}")
        lstm_model_builder = LSTMModelBuilder()
        final_lstm_model = lstm_model_builder.build_lstm_model_final(best_hps_dict, (time_steps, n_feats), num_classes)
        final_lstm_model = train_model(
            final_lstm_model,
            torch.utils.data.TensorDataset(torch.from_numpy(X_os).float(),
                                           torch.from_numpy(y_train_res).long()),
            torch.utils.data.TensorDataset(torch.from_numpy(X_os).float(),
                                           torch.from_numpy(y_train_res).long()),
            nn.CrossEntropyLoss(),
            optim.Adam(final_lstm_model.parameters(), lr=best_hps_dict.get("learning_rate", 1e-3)),
            CONFIG["epochs"],
            device,
            F1Score
        )
        lstm_models[best_lstm_strategy] = final_lstm_model
        torch.save(final_lstm_model.state_dict(), f"models/LSTM_{best_lstm_strategy}_best_model.pt")
        logger.info(f"Saved final LSTM model: models/LSTM_{best_lstm_strategy}_best_model.pt")

    # ----- Retrain CNN Model -----
    if best_cnn_strategy and cnn_cv_results[best_cnn_strategy]:
        try:
            best_cnn_fold = max(cnn_cv_results[best_cnn_strategy], key=lambda r: r["F1-Score"])
            best_hps_dict = best_cnn_fold["Best_Hyperparameters"]
        except Exception as e:
            logger.error("Error extracting best hyperparameters for CNN.", exc_info=True)
            best_hps_dict = {}
        cnn_model_builder = CNNModelBuilder()
        final_cnn_model = cnn_model_builder.build_cnn_model_final(best_hps_dict, (time_steps, n_feats), num_classes)
        final_cnn_model = train_model(
            final_cnn_model,
            torch.utils.data.TensorDataset(torch.from_numpy(X_os).float(),
                                           torch.from_numpy(y_train_res).long()),
            torch.utils.data.TensorDataset(torch.from_numpy(X_os).float(),
                                           torch.from_numpy(y_train_res).long()),
            nn.CrossEntropyLoss(),
            optim.Adam(final_cnn_model.parameters(), lr=best_hps_dict.get("learning_rate", 1e-3)),
            CONFIG["epochs"],
            device,
            F1Score
        )
        cnn_best_model = final_cnn_model
        torch.save(final_cnn_model.state_dict(), f"models/CNN_{best_cnn_strategy}_best_model.pt")
        logger.info(f"Saved final CNN model: models/CNN_{best_cnn_strategy}_best_model.pt")

    # ----- Retrain Transformer Model -----
    if best_transformer_strategy and transformer_cv_results[best_transformer_strategy]:
        try:
            best_trans_fold = max(transformer_cv_results[best_transformer_strategy], key=lambda r: r["F1-Score"])
            best_hps_dict = best_trans_fold["Best_Hyperparameters"]
        except Exception as e:
            logger.error("Error extracting best hyperparameters for Transformer.", exc_info=True)
            best_hps_dict = {}
        transformer_model_builder = TransformerModelBuilder()
        final_trans_model = transformer_model_builder.build_transformer_model_final(best_hps_dict,
                                                                                    (time_steps, n_feats), num_classes)
        final_trans_model = train_model(
            final_trans_model,
            torch.utils.data.TensorDataset(torch.from_numpy(X_os).float(),
                                           torch.from_numpy(y_train_res).long()),
            torch.utils.data.TensorDataset(torch.from_numpy(X_os).float(),
                                           torch.from_numpy(y_train_res).long()),
            nn.CrossEntropyLoss(),
            optim.Adam(final_trans_model.parameters(), lr=best_hps_dict.get("learning_rate", 1e-3)),
            CONFIG["epochs"],
            device,
            F1Score
        )
        transformer_best_model = final_trans_model
        torch.save(final_trans_model.state_dict(), f"models/Transformer_{best_transformer_strategy}_best_model.pt")
        logger.info(f"Saved final Transformer model: models/Transformer_{best_transformer_strategy}_best_model.pt")
    logger.info("END: Retraining final models")

    logger.info("START: Collecting predictions for ensemble from final models")
    y_pred_lstm = y_pred_prob_lstm = None
    if best_lstm_strategy in lstm_models:
        try:
            final_lstm_model = lstm_models[best_lstm_strategy]
            final_lstm_model.eval()
            with torch.no_grad():
                outputs = final_lstm_model(torch.from_numpy(X_test_nn).float().to(device))
            y_pred_lstm = torch.argmax(outputs, dim=1).cpu().numpy()
            y_pred_prob_lstm = outputs.cpu().numpy()
            logger.info("LSTM predictions computed successfully.")
        except Exception as e:
            logger.error(f"LSTM prediction error: {e}", exc_info=True)
    y_pred_cnn = y_pred_prob_cnn = None
    if cnn_best_model:
        try:
            cnn_best_model.eval()
            with torch.no_grad():
                outputs = cnn_best_model(torch.from_numpy(X_test_nn).float().to(device))
            y_pred_cnn = torch.argmax(outputs, dim=1).cpu().numpy()
            y_pred_prob_cnn = outputs.cpu().numpy()
            logger.info("CNN predictions computed successfully.")
        except Exception as e:
            logger.error(f"CNN prediction error: {e}", exc_info=True)
    y_pred_transformer = y_pred_prob_transformer = None
    if transformer_best_model:
        try:
            transformer_best_model.eval()
            with torch.no_grad():
                outputs = transformer_best_model(torch.from_numpy(X_test_nn).float().to(device))
            y_pred_transformer = torch.argmax(outputs, dim=1).cpu().numpy()
            y_pred_prob_transformer = outputs.cpu().numpy()
            logger.info("Transformer predictions computed successfully.")
        except Exception as e:
            logger.error(f"Transformer prediction error: {e}", exc_info=True)
    logger.info("END: Collecting predictions for ensemble")

    logger.info("START: Combining predictions for ensemble")
    pred_probs_list = []
    if y_pred_prob_lstm is not None:
        adj = adjust_pred_proba(y_pred_prob_lstm, num_classes)
        _, pad = prepare_roc_auc(y_test, adj, num_classes, label_encoder)
        pred_probs_list.append(pad)
    if y_pred_prob_cnn is not None:
        adj = adjust_pred_proba(y_pred_prob_cnn, num_classes)
        _, pad = prepare_roc_auc(y_test, adj, num_classes, label_encoder)
        pred_probs_list.append(pad)
    if y_pred_prob_transformer is not None:
        adj = adjust_pred_proba(y_pred_prob_transformer, num_classes)
        _, pad = prepare_roc_auc(y_test, adj, num_classes, label_encoder)
        pred_probs_list.append(pad)
    combined_pred = combined_pred_prob = None
    if pred_probs_list:
        combined_pred_prob = np.mean(pred_probs_list, axis=0)
        combined_pred = np.argmax(combined_pred_prob, axis=1)
        y_test_bin = label_binarize(y_test, classes=range(num_classes))
        acc_combined = accuracy_score(y_test, combined_pred)
        prec_combined = precision_score(y_test, combined_pred, average="weighted", zero_division=0)
        rec_combined = recall_score(y_test, combined_pred, average="weighted", zero_division=0)
        f1_combined = f1_score(y_test, combined_pred, average="weighted", zero_division=0)
        roc_combined = roc_auc_score(y_test_bin, combined_pred_prob, multi_class="ovr", average="weighted")
        logger.info(f"Combined Model: ACC={acc_combined:.4f}, F1={f1_combined:.4f}, ROC_AUC={roc_combined:.4f}")
        metrics["Combined_Model"] = {"Accuracy": acc_combined, "Precision": prec_combined, "Recall": rec_combined,
                                     "F1-Score": f1_combined, "ROC_AUC": roc_combined,
                                     "y_pred": combined_pred, "y_pred_prob": combined_pred_prob}
    else:
        logger.error("No model probabilities available for ensemble.")
    logger.info("END: Combining predictions for ensemble")

    logger.info("START: Evaluating individual models and ensemble to select best overall")
    metrics_evaluation = {}
    accuracies = {}
    f1_scores_eval = {}
    roc_aucs = {}
    if y_pred_lstm is not None:
        acc = accuracy_score(y_test, y_pred_lstm)
        f1_val = f1_score(y_test, y_pred_lstm, average="weighted", zero_division=0)
        y_test_bin = label_binarize(y_test, classes=range(num_classes))
        roc_val = roc_auc_score(y_test_bin, y_pred_prob_lstm, multi_class="ovr",
                                average="weighted") if y_pred_prob_lstm is not None else 0.0
        metrics_evaluation["LSTM"] = {"Accuracy": acc, "F1-Score": f1_val, "ROC_AUC": roc_val,
                                      "y_pred": y_pred_lstm, "y_pred_prob": y_pred_prob_lstm}
        accuracies["LSTM"] = acc;
        f1_scores_eval["LSTM"] = f1_val;
        roc_aucs["LSTM"] = roc_val
    if y_pred_cnn is not None:
        acc = accuracy_score(y_test, y_pred_cnn)
        f1_val = f1_score(y_test, y_pred_cnn, average="weighted", zero_division=0)
        y_test_bin = label_binarize(y_test, classes=range(num_classes))
        roc_val = roc_auc_score(y_test_bin, y_pred_prob_cnn, multi_class="ovr",
                                average="weighted") if y_pred_prob_cnn is not None else 0.0
        metrics_evaluation["CNN"] = {"Accuracy": acc, "F1-Score": f1_val, "ROC_AUC": roc_val,
                                     "y_pred": y_pred_cnn, "y_pred_prob": y_pred_prob_cnn}
        accuracies["CNN"] = acc;
        f1_scores_eval["CNN"] = f1_val;
        roc_aucs["CNN"] = roc_val
    if y_pred_transformer is not None:
        acc = accuracy_score(y_test, y_pred_transformer)
        f1_val = f1_score(y_test, y_pred_transformer, average="weighted", zero_division=0)
        y_test_bin = label_binarize(y_test, classes=range(num_classes))
        roc_val = roc_auc_score(y_test_bin, y_pred_prob_transformer, multi_class="ovr",
                                average="weighted") if y_pred_prob_transformer is not None else 0.0
        metrics_evaluation["Transformer"] = {"Accuracy": acc, "F1-Score": f1_val, "ROC_AUC": roc_val,
                                             "y_pred": y_pred_transformer, "y_pred_prob": y_pred_prob_transformer}
        accuracies["Transformer"] = acc;
        f1_scores_eval["Transformer"] = f1_val;
        roc_aucs["Transformer"] = roc_val
    if "Combined_Model" in metrics:
        cm = metrics["Combined_Model"]
        metrics_evaluation["Combined_Model"] = cm
        accuracies["Combined_Model"] = cm["Accuracy"]
        f1_scores_eval["Combined_Model"] = cm["F1-Score"]
        roc_aucs["Combined_Model"] = cm["ROC_AUC"]
    best_model_name = None;
    best_model_score = -np.inf
    for m in accuracies:
        combined_score = 0.5 * f1_scores_eval[m] + 0.5 * roc_aucs[m]
        if combined_score > best_model_score:
            best_model_score = combined_score
            best_model_name = m
    logger.info(f"Best Model Overall: {best_model_name} with Combined Score: {best_model_score:.4f}")

    # Save overall model performance evaluation to CSV
    df_metrics = pd.DataFrame(metrics_evaluation).transpose()
    metrics_csv_path = r"C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\model_performance_evaluation.csv"
    df_metrics.to_csv(metrics_csv_path, index=True)
    logger.info(f"Saved model performance evaluation to {metrics_csv_path}")

    logger.info("START: Running interpretability (LIME) analyses on all models...")
    all_models = {"LSTM": lstm_models.get(best_lstm_strategy),
                  "CNN": cnn_best_model,
                  "Transformer": transformer_best_model}
    for m_name, m_instance in all_models.items():
        if m_instance is None:
            logger.warning(f"{m_name} not available for interpretability.")
            continue
        if m_name in ["LSTM", "CNN", "Transformer"]:
            X_test_model = X_test_nn
            y_test_model = y_test
            X_train_model = X_train_nn
            y_train_model = y_train
            if X_test_model.ndim == 3:
                w_size = X_test_model.shape[1]
                feats = X_test_model.shape[2]
                feature_names = [f"Feature_{i + 1}" for i in range(w_size * feats)]
            else:
                feature_names = [f"Feature_{i + 1}" for i in range(X_test_model.shape[1])]
        else:
            X_test_model = X_test_nn
            y_test_model = y_test
            X_train_model = X_train_nn
            y_train_model = y_train
            feature_names = [f"Feature_{i + 1}" for i in range(X_test_model.shape[1])]
        try:
            logger.info(f"Starting LIME analysis for model: {m_name}")
            ModelInterpretability.lime_analysis(model=m_instance,
                                                X_train=X_train_model,
                                                y_train=y_train_model,
                                                X_test=X_test_model,
                                                y_test=y_test_model,
                                                feature_names=feature_names,
                                                model_type=m_name)
            logger.info(f"LIME analysis completed for model: {m_name}")
        except Exception as e:
            logger.error(f"Error during LIME analysis for {m_name}.", exc_info=True)
    logger.info("END: Interpretability (LIME) analyses")

    logger.info("START: Saving the best model for future use...")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if best_model_name == "LSTM" and best_lstm_strategy in lstm_models:
        torch.save(lstm_models[best_lstm_strategy].state_dict(), f"models/LSTM_best_{timestamp}.pt")
        logger.info(f"Saved best LSTM as models/LSTM_best_{timestamp}.pt")
    elif best_model_name == "CNN" and cnn_best_model:
        torch.save(cnn_best_model.state_dict(), f"models/CNN_best_{timestamp}.pt")
        logger.info(f"Saved best CNN as models/CNN_best_{timestamp}.pt")
    elif best_model_name == "Transformer" and transformer_best_model:
        torch.save(transformer_best_model.state_dict(), f"models/Transformer_best_{timestamp}.pt")
        logger.info(f"Saved best Transformer as models/Transformer_best_{timestamp}.pt")
    elif best_model_name == "Combined_Model":
        logger.info("Best model is the Combined Model (ensemble). Individual models are already saved.")
    else:
        logger.warning("No valid best model selected to save.")
    logger.info("END: Saving best model")

    logger.info("START: Saving predictions to a CSV file...")
    # Read the CSV file that was used for preparing predictions
    df_predictions = pd.read_csv(
        "C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\final_labels_cleaned_merged.csv")

    # Filter the rows based on the STRIDE_INTERVALS
    mask_pred = np.zeros(len(df_predictions), dtype=bool)
    for low, high in STRIDE_INTERVALS:
        mask_pred |= (df_predictions['Frame'] >= low) & (df_predictions['Frame'] <= high)
    df_predictions = df_predictions[mask_pred]

    # Extract the features for prediction
    X_features = df_predictions[["Synergy_1", "Synergy_2", "Synergy_3", "Synergy_4", "Synergy_5"]].values

    # Create sequences for prediction using the same window and overlap configuration
    X_seq_pred, _, center_inds = FeatureExtractor.create_sequences(
        X_features, window_size=CONFIG['window_size'], overlap=CONFIG['overlap'], labels=None
    )
    logger.info(f"Created sequences for prediction: {X_seq_pred.shape}")

    # batched inference to avoid out-of-memory errors.
    def batched_inference(model, X_seq, device, batch_size=16):
        model.eval()
        outputs_list = []
        with torch.no_grad():
            for i in range(0, len(X_seq), batch_size):
                batch = X_seq[i:i + batch_size]
                batch_tensor = torch.from_numpy(batch).float().to(device)
                output = model(batch_tensor)
                outputs_list.append(output)
        # Concatenate predictions from all batches into a single tensor
        return torch.cat(outputs_list, dim=0)

    # batched inference based for which model is the best according to selection
    if best_model_name == "LSTM":
        pred_seq = batched_inference(lstm_models[best_lstm_strategy], X_seq_pred, device)
    elif best_model_name == "CNN":
        pred_seq = batched_inference(cnn_best_model, X_seq_pred, device)
    elif best_model_name == "Transformer":
        pred_seq = batched_inference(transformer_best_model, X_seq_pred, device)
    elif best_model_name == "Combined_Model":
        outputs_list = []
        if best_lstm_strategy in lstm_models:
            outputs_list.append(batched_inference(lstm_models[best_lstm_strategy], X_seq_pred, device))
        if cnn_best_model is not None:
            outputs_list.append(batched_inference(cnn_best_model, X_seq_pred, device))
        if transformer_best_model is not None:
            outputs_list.append(batched_inference(transformer_best_model, X_seq_pred, device))
        if outputs_list:
            pred_seq = torch.mean(torch.stack(outputs_list), dim=0)
        else:
            logger.error("No models available in ensemble for prediction.")
            exit(1)
    else:
        logger.error("No valid best model found for prediction.")
        exit(1)

    # Get the predicted classes from the prediction probabilities
    df_predictions = pd.read_csv(
        "C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\predicted_stride_stages_final.csv"
    )

    # Predict classes
    y_pred_seq = torch.argmax(pred_seq, dim=1).cpu().numpy()

    # Fill predictions
    stride = int(CONFIG['window_size'] * (1 - CONFIG['overlap']))
    full_predictions = np.full(len(df_predictions), -1, dtype=int)

    start = 0
    for pred in y_pred_seq:
        end = start + CONFIG['window_size']
        if end > len(full_predictions):
            end = len(full_predictions)
        full_predictions[start:end] = pred
        start += stride

    # Forward/backward fill missing
    pred_series = pd.Series(full_predictions)
    pred_series.replace(-1, np.nan, inplace=True)
    pred_series = pred_series.ffill().bfill().astype(int)

    # Decode back to stage names
    predicted_labels = label_encoder.inverse_transform(pred_series.values)

    # Save to new column "PredictedStage_new"
    df_predictions["PredictedStage_new"] = predicted_labels

    # Compare original vs new
    original_labels_encoded = label_encoder.transform(df_predictions["PredictedStage"])
    new_labels_encoded = label_encoder.transform(df_predictions["PredictedStage_new"])

    # Calculate Accuracy
    accuracy = accuracy_score(original_labels_encoded, new_labels_encoded)
    print(f"Accuracy between original and new PredictedStage: {accuracy:.4f}")

    # 8. Save to CSV
    df_predictions.to_csv(
        "C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\final_labels_with_predictions.csv",
        index=False
    )
    logger.info(f"Saved updated predictions to final_labels_with_predictions.csv with accuracy {accuracy:.4f}")

if __name__ == "__main__":
    main_pipeline()

import logging
import os
import sys
import numpy as np
import pandas as pd
import ray
import warnings
from hmmlearn import hmm
from ray import tune
from ray.tune.schedulers import HyperBandScheduler, ASHAScheduler
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from Gait_Phase.gpu_utils import configure_gpu
from pandas.errors import PerformanceWarning
warnings.filterwarnings("ignore", category=PerformanceWarning)
warnings.filterwarnings("ignore", message="The `local_dir` argument is deprecated")

configure_gpu()

def run_fusion_hmm_pipeline():
    # --------------------------------------------------
    # GLOBAL INTERVALS and CONSTANTS
    # --------------------------------------------------
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
    THRESHOLD_FRAME = 159902  # Frames >= this value are outside our stride intervals


    # --------------------------------------------------
    # SETUP LOGGING
    # --------------------------------------------------
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )

    def step_log(step_number, message):
        logging.info(f"Step {step_number}: {message}")

    # --------------------------------------------------
    # HELPER FUNCTIONS
    # --------------------------------------------------
    def label_frames_custom(frame_indices, stride_intervals, threshold_frame):
        labels = np.zeros_like(frame_indices)
        for start, end in stride_intervals:
            if start < threshold_frame:
                actual_end = min(end, threshold_frame - 1)
                labels[(frame_indices >= start) & (frame_indices <= actual_end)] = 1
        return labels

    def enforce_symmetry(cov, min_covar=1e-0):
        cov_sym = 0.5 * (cov + cov.T)
        eigvals, eigvecs = np.linalg.eigh(cov_sym)
        eigvals[eigvals < min_covar] = min_covar
        cov_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        return cov_pd

    def stationary_distribution(transmat):
        eigvals, eigvecs = np.linalg.eig(transmat.T)
        idx = np.argmin(np.abs(eigvals - 1))
        stat = np.real(eigvecs[:, idx])
        stat = stat / np.sum(stat)
        return stat

    # --------------------------------------------------
    # DATA LOADING FUNCTIONS
    # --------------------------------------------------
    def load_marker_data(file_path):
        logging.debug("Loading gait marker CSV data.")
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()

        if 'Frame' not in df.columns:
            df.insert(0, 'Frame', np.arange(len(df)))

        if 'Time (Seconds)' not in df.columns:
            if 'Time_s' in df.columns:
                df = df.rename(columns={'Time_s': 'Time (Seconds)'})
            else:
                raise KeyError("'Time (Seconds)' column is missing.")

        logging.info(f"Gait marker data loaded with shape {df.shape}")
        return df

    def load_synergy_data(file_path):
        logging.debug("Loading muscle synergy CSV data.")
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        df = pd.read_csv(file_path, sep=None, engine='python')
        df.columns = df.columns.str.strip()
        required = ['Synergy_1', 'Synergy_2', 'Synergy_3', 'Synergy_4', 'Synergy_5', 'Frame', 'Time (Seconds)']
        for col in required:
            if col not in df.columns:
                raise KeyError(f"Required column '{col}' missing in synergy data.")
        logging.info(f"Muscle synergy data loaded with shape {df.shape}")
        return df

    # --------------------------------------------------
    # GAIT DATA PROCESSING FUNCTIONS (Markers)
    # --------------------------------------------------
    def compute_velocities(df, marker_prefixes):
        logging.debug("Computing velocities.")
        df = df.sort_values('Frame').reset_index(drop=True)
        df['dt'] = df['Time (Seconds)'].diff().fillna(0).replace(0, 1e-6)
        for prefix in marker_prefixes:
            for axis in ['X', 'Y', 'Z']:
                pos_col = f"{prefix}_{axis}"
                vel_col = f"{prefix}_V{axis}"
                if pos_col not in df.columns:
                    df[vel_col] = 0
                else:
                    df[vel_col] = df[pos_col].diff() / df['dt']
                    df[vel_col] = df[vel_col].fillna(0)
        df.drop(columns=['dt'], inplace=True)
        return df

    def compute_accelerations(df, marker_prefixes):
        logging.debug("Computing accelerations.")
        df['dt_acc'] = df['Time (Seconds)'].diff().fillna(0).replace(0, 1e-6)
        for prefix in marker_prefixes:
            for axis in ['X', 'Y', 'Z']:
                vel_col = f"{prefix}_V{axis}"
                acc_col = f"{prefix}_A{axis}"
                if vel_col not in df.columns:
                    df[acc_col] = 0
                else:
                    df[acc_col] = df[vel_col].diff() / df['dt_acc']
                    df[acc_col] = df[acc_col].fillna(0)
        df.drop(columns=['dt_acc'], inplace=True)
        return df

    def compute_angular_features(df):
        logging.debug("Computing angular features.")

        def compute_angle(p1, p2, p3):
            v1 = p1 - p2
            v2 = p3 - p2
            dot_product = np.einsum('ij,ij->i', v1, v2)
            norms = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-8
            angles = np.arccos(np.clip(dot_product / norms, -1.0, 1.0))
            return np.degrees(angles)

        angle_defs = {
            'R_Knee_Angle': ('RTHI', 'RKNE', 'RTIB'),
            'L_Knee_Angle': ('LTHI', 'LKNE', 'LTIB')
        }
        for angle_name, (p1, p2, p3) in angle_defs.items():
            req_cols = [f"{m}_{axis}" for m in [p1, p2, p3] for axis in ['X', 'Y', 'Z']]
            if not all(col in df.columns for col in req_cols):
                df[angle_name] = 0
            else:
                p1_coords = df[[f"{p1}_X", f"{p1}_Y", f"{p1}_Z"]].values
                p2_coords = df[[f"{p2}_X", f"{p2}_Y", f"{p2}_Z"]].values
                p3_coords = df[[f"{p3}_X", f"{p3}_Y", f"{p3}_Z"]].values
                df[angle_name] = compute_angle(p1_coords, p2_coords, p3_coords)
        return df

    def compute_angular_momentum(df, marker_prefixes):
        logging.debug("Computing angular momentum.")
        for marker in marker_prefixes:
            pos_cols = [f"{marker}_X", f"{marker}_Y", f"{marker}_Z"]
            vel_cols = [f"{marker}_V{axis}" for axis in ['X', 'Y', 'Z']]
            target_cols = [f"{marker}_Angular_Momentum_{axis}" for axis in ['X', 'Y', 'Z']]
            if not all(col in df.columns for col in pos_cols + vel_cols):
                for col in target_cols:
                    df[col] = 0
                continue
            pos = df[pos_cols].values
            vel = df[vel_cols].values
            omega = np.cross(pos, vel)
            df[target_cols] = omega
        return df

    def butter_lowpass_filter_multi(data, cutoff=5, fs=2000, order=6):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        logging.debug("Filtering data with shape: %s", data.shape)
        if data.shape[0] <= 21:
            logging.error("Input data length (%d) is too short for filtering.", data.shape[0])
        return filtfilt(b, a, data, axis=0)

    def extract_features(df, window_size, feature_combination, marker_prefixes):
        logging.debug(f"Extracting gait features with window_size={window_size} and combination={feature_combination}")
        expected = []
        for m in marker_prefixes:
            for ax in ['X', 'Y', 'Z']:
                expected.extend([f'{m}_{ax}', f'{m}_V{ax}', f'{m}_A{ax}'])
        stat_types = [st for st in ['var', 'poly'] if st in feature_combination]
        for st in stat_types:
            for m in marker_prefixes:
                for ax in ['X', 'Y', 'Z']:
                    expected.extend([f'{m}_{ax}_{st}', f'{m}_V{ax}_{st}', f'{m}_A{ax}_{st}'])
        distance_pairs = [
            ('LTHI', 'RTHI'),
            ('LKNE', 'RKNE'),
            ('LTIB', 'RTIB'),
            ('LANK', 'RANK'),
            ('LHEE', 'RHEE'),
            ('LTOE', 'RTOE')
        ]
        for pair in distance_pairs:
            expected.append(f'{pair[0]}_{pair[1]}_dist')
        for m in marker_prefixes:
            expected.append(f'{m}_A_mag')
            for axis in ['X', 'Y', 'Z']:
                expected.append(f'{m}_Angular_Momentum_{axis}')
        feats = pd.DataFrame(index=df.index, columns=expected, dtype=float)
        for col in expected:
            if col in df.columns:
                feats[col] = df[col].values
        for st in stat_types:
            for m in marker_prefixes:
                for ax in ['X', 'Y', 'Z']:
                    col_base = f'{m}_{ax}'
                    if col_base in df.columns:
                        if st == 'var':
                            feats[f'{col_base}_var'] = df[col_base].rolling(window_size, min_periods=1).var().fillna(0)
                        elif st == 'poly':
                            feats[f'{col_base}_poly'] = df[col_base].rolling(window_size, min_periods=3).apply(
                                lambda x: np.polyfit(np.arange(len(x)), x, 2)[-1] if len(x) >= 3 else 0, raw=True
                            ).fillna(0)
                    for prefix in [f'{m}_V{ax}', f'{m}_A{ax}']:
                        if prefix in df.columns:
                            if st == 'var':
                                feats[f'{prefix}_var'] = df[prefix].rolling(window_size, min_periods=1).var().fillna(0)
                            elif st == 'poly':
                                feats[f'{prefix}_poly'] = df[prefix].rolling(window_size, min_periods=3).apply(
                                    lambda x: np.polyfit(np.arange(len(x)), x, 2)[-1] if len(x) >= 3 else 0, raw=True
                                ).fillna(0)
        for pair in distance_pairs:
            m1, m2 = pair
            dist_col = f'{m1}_{m2}_dist'
            if all(col in df.columns for col in [f'{m1}_X', f'{m1}_Y', f'{m1}_Z', f'{m2}_X', f'{m2}_Y', f'{m2}_Z']):
                feats[dist_col] = np.linalg.norm(
                    df[[f'{m1}_X', f'{m1}_Y', f'{m1}_Z']].values -
                    df[[f'{m2}_X', f'{m2}_Y', f'{m2}_Z']].values, axis=1
                )
            else:
                feats[dist_col] = 0
        for m in marker_prefixes:
            acc_cols = [f'{m}_A{ax}' for ax in ['X', 'Y', 'Z']]
            if all(col in df.columns for col in acc_cols):
                feats[f'{m}_A_mag'] = np.linalg.norm(df[acc_cols].values, axis=1)
            else:
                feats[f'{m}_A_mag'] = 0
        feats.fillna(0, inplace=True)
        logging.debug(f"Extracted gait features shape: {feats.shape}")
        return feats

    def extract_synergy_features(df, window_size):
        logging.debug("Extracting EMG synergy features.")
        synergy_cols = [f"Synergy_{i}" for i in range(1, 6)]
        feats = df[synergy_cols].copy()
        for col in synergy_cols:
            feats[f"{col}_var"] = df[col].rolling(window_size, min_periods=1).var().fillna(0)
            feats[f"{col}_poly"] = df[col].rolling(window_size, min_periods=3).apply(
                lambda x: np.polyfit(np.arange(len(x)), x, 2)[-1] if len(x) >= 3 else 0, raw=True
            ).fillna(0)
        feats.fillna(0, inplace=True)
        logging.debug(f"Extracted EMG synergy features shape: {feats.shape}")
        return feats

    # --------------------------------------------------
    # HMM TRAINING FUNCTIONS
    # --------------------------------------------------
    def train_markers_hmm(X, n_components=5, covariance_type='full', n_iter=100, *,
                          alpha, beta, gamma, theta, delta):
        logging.info("Training markers HMM with 5 states using gait marker features.")

        # Scale features to improve numerical stability.
        X = StandardScaler().fit_transform(X)

        X += 1e-0 * np.random.randn(*X.shape)
        markers_model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=42,
            init_params="mc",
            min_covar=1e-3
        )
        markers_model.startprob_ = np.zeros(n_components)
        markers_model.startprob_[0] = 1.0
        transmat = np.array([
            [alpha, 1 - alpha, 0.0, 0.0, 0.0],
            [0.0, beta, 1 - beta, 0.0, 0.0],
            [0.0, 0.0, gamma, 1 - gamma, 0.0],
            [0.0, 0.0, 0.0, delta, 1 - delta],
            [1 - theta, 0.0, 0.0, 0.0, theta]
        ])
        markers_model.transmat_ = transmat

        # Attempt to fit the model and catch errors for further debugging
        try:
            markers_model.fit(X)
        except ValueError as e:
            logging.error(f"HMM fitting failed with: {e}")
            raise e

        if covariance_type == 'full':
            # Enforce symmetry and positive-definiteness on each covariance matrix.
            new_covars = []
            for cov in markers_model.covars_:
                sym_cov = enforce_symmetry(cov)
                eigvals = np.linalg.eigvals(sym_cov)
                if np.any(eigvals <= 1e-6):
                    logging.warning("Covariance not positive-definite. Adjusting by adding jitter.")
                    eigvals = np.clip(eigvals, 1e-6, None)
                    # Reconstruct the covariance matrix with the adjusted eigenvalues.
                    eigvals_full, eigvecs = np.linalg.eigh(sym_cov)
                    sym_cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
                new_covars.append(sym_cov)
            markers_model.covars_ = np.array(new_covars)

        if np.isnan(markers_model.means_).any():
            global_mean = np.mean(X, axis=0)
            markers_model.means_ = np.where(np.isnan(markers_model.means_), global_mean, markers_model.means_)

        return markers_model

    def train_synergy_hmm(X, n_components=5, covariance_type='full', n_iter=100, *,
                          alpha, beta, gamma, delta, theta):
        logging.info("Training synergy HMM with 5 states using EMG synergy features.")

        # Scale features to improve numerical stability.
        X = StandardScaler().fit_transform(X)

        X += 1e-0 * np.random.randn(*X.shape)

        synergy_model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=42,
            init_params="mc",
            min_covar=1e-3
        )
        synergy_model.startprob_ = np.zeros(n_components)
        synergy_model.startprob_[0] = 1.0
        transmat = np.array([
            [alpha, 1 - alpha, 0.0, 0.0, 0.0],
            [0.0, beta, 1 - beta, 0.0, 0.0],
            [0.0, 0.0, gamma, 1 - gamma, 0.0],
            [0.0, 0.0, 0.0, delta, 1 - delta],
            [1 - theta, 0.0, 0.0, 0.0, theta]
        ])
        synergy_model.transmat_ = transmat

        try:
            synergy_model.fit(X)
        except ValueError as e:
            logging.error(f"Synergy HMM fitting failed with: {e}")
            raise e

        if covariance_type == 'full':
            # Enforce symmetry and positive-definiteness on each covariance matrix.
            new_covars = []
            for cov in synergy_model.covars_:
                sym_cov = enforce_symmetry(cov)
                # Check if the covariance matrix is positive-definite.
                eigvals = np.linalg.eigvals(sym_cov)
                if np.any(eigvals <= 1e-6):
                    logging.warning("Covariance not positive-definite. Adjusting by adding jitter.")
                    eigvals = np.clip(eigvals, 1e-6, None)
                    eigvals_full, eigvecs = np.linalg.eigh(sym_cov)
                    sym_cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
                new_covars.append(sym_cov)
            synergy_model.covars_ = np.array(new_covars)

        if np.isnan(synergy_model.means_).any():
            global_mean = np.mean(X, axis=0)
            synergy_model.means_ = np.where(np.isnan(synergy_model.means_), global_mean, synergy_model.means_)

        return synergy_model

    def fuse_hmm_predictions(model_markers, model_synergy, X_markers, X_synergy,
                             weight_markers=0.5, weight_synergy=0.5):
        logging.info("Fusing HMM predictions from markers and synergy models.")
        _, posteriors_markers = model_markers.score_samples(X_markers)
        _, posteriors_synergy = model_synergy.score_samples(X_synergy)
        combined_posteriors = weight_markers * posteriors_markers + weight_synergy * posteriors_synergy
        fused_states = np.argmax(combined_posteriors, axis=1) + 1  # 1-indexed states
        return fused_states

    # --------------------------------------------------
    # PRECOMPUTATION: Process data once for tuning
    # --------------------------------------------------
    def precompute_markers_features():
        gait_csv_path = "C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\aligned_markers_data_cleaned.csv"
        df_gait = load_marker_data(gait_csv_path)
        df_gait['Label'] = label_frames_custom(df_gait['Frame'].values, STRIDE_INTERVALS, THRESHOLD_FRAME)
        df_gait = df_gait[df_gait['Label'] == 1].copy()
        marker_prefixes = [
            'RTHI', 'LTHI',
            'RKNE', 'LKNE',
            'RTIB', 'LTIB',
            'RANK', 'LANK',
            'RHEE', 'LHEE',
            'RTOE', 'LTOE'
        ]
        df_gait = compute_velocities(df_gait, marker_prefixes)
        df_gait = compute_accelerations(df_gait, marker_prefixes)
        df_gait = compute_angular_features(df_gait)
        df_gait = compute_angular_momentum(df_gait, marker_prefixes)
        window_size = 99
        feature_combination = ['var', 'poly']
        gait_feats = extract_features(df_gait, window_size, feature_combination, marker_prefixes)
        pca = PCA(n_components=3)
        X_gait_pca = pca.fit_transform(gait_feats)
        return X_gait_pca, df_gait['Frame'].values

    def precompute_synergy_features():
        emg_csv_path = "C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\muscle_synergy_activations.csv"
        df_emg = load_synergy_data(emg_csv_path)
        df_emg['Label'] = label_frames_custom(df_emg['Frame'].values, STRIDE_INTERVALS, THRESHOLD_FRAME)
        df_emg = df_emg[df_emg['Label'] == 1].copy()
        window_size = 99
        synergy_feats = extract_synergy_features(df_emg, window_size)
        X_emg = synergy_feats.values
        return X_emg, df_emg['Frame'].values

    # Precompute and store globally for tuning
    GLOBAL_MARKERS_DATA, MARKERS_FRAMES = precompute_markers_features()
    GLOBAL_SYNERGY_DATA, SYNERGY_FRAMES = precompute_synergy_features()

    # --------------------------------------------------
    # TUNING FUNCTION: Markers HMM
    # --------------------------------------------------
    def tune_pipeline_markers(config):
        X_gait_pca = GLOBAL_MARKERS_DATA
        # Train the markers HMM without any distribution penalty.
        markers_model = train_markers_hmm(
            X_gait_pca, n_components=5, n_iter=100,
            alpha=config["alpha"], beta=config["beta"],
            gamma=config["gamma"], delta=config["delta"], theta=config["theta"]
        )
        # Use the raw log-likelihood as the score.
        score = markers_model.score(X_gait_pca)
        tune.report({"score": score})

    def tune_pipeline_synergy(config):
        X_emg = GLOBAL_SYNERGY_DATA
        # Train the synergy HMM without any distribution penalty.
        synergy_model = train_synergy_hmm(
            X_emg, n_components=5, n_iter=100,
            alpha=config["alpha"], beta=config["beta"],
            gamma=config["gamma"], delta=config["delta"], theta=config["theta"]
        )
        score = synergy_model.score(X_emg)
        tune.report({"score": score})

    # --------------------------------------------------
    # FINAL TRAINING FUNCTION: Full Processing, Model Training, Fusion, and Saving Results
    # --------------------------------------------------
    def final_training(markers_tuned_params, synergy_tuned_params):
        step_log(1, "Loading gait marker data and EMG synergy data.")
        gait_csv_path = "C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\aligned_markers_data_cleaned.csv"
        emg_csv_path = "C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\muscle_synergy_activations.csv"
        try:
            df_gait = load_marker_data(gait_csv_path)
            df_emg = load_synergy_data(emg_csv_path)
        except Exception as e:
            logging.error(f"Data loading error: {e}")
            return
        step_log(1.5, "Merging gait and EMG synergy data on 'Frame' and 'Time (Seconds)'.")
        df_merged = pd.merge(df_gait, df_emg, on=['Frame', 'Time (Seconds)'], how='inner', suffixes=('_gait', '_emg'))
        logging.info(f"Merged data shape: {df_merged.shape}")
        df_merged['Label'] = label_frames_custom(df_merged['Frame'].values, STRIDE_INTERVALS, THRESHOLD_FRAME)
        df_train = df_merged[df_merged['Label'] == 1].copy()
        logging.info(f"Using {df_train.shape[0]} stride frames for training.")
        if df_train.empty:
            logging.error("No training data available after labeling. Check STRIDE_INTERVALS and THRESHOLD_FRAME.")
            return
        marker_prefixes = [
            'RTHI', 'LTHI',
            'RKNE', 'LKNE',
            'RTIB', 'LTIB',
            'RANK', 'LANK',
            'RHEE', 'LHEE',
            'RTOE', 'LTOE'
        ]
        step_log(3, "Computing velocities, accelerations, angular features and angular momentum on training gait data.")
        df_train = compute_velocities(df_train, marker_prefixes)
        df_train = compute_accelerations(df_train, marker_prefixes)
        df_train = compute_angular_features(df_train)
        df_train = compute_angular_momentum(df_train,marker_prefixes)
        step_log(4, "Applying low-pass filter to training gait data.")
        pos_cols = [col for col in df_train.columns if any(
            col.startswith(prefix + '_') and col.endswith(('_X', '_Y', '_Z')) for prefix in marker_prefixes)]
        vel_cols = [col for col in df_train.columns if any(
            col.startswith(prefix + '_V') and col.endswith(('_X', '_Y', '_Z')) for prefix in marker_prefixes)]
        acc_cols = [col for col in df_train.columns if any(
            col.startswith(prefix + '_A') and col.endswith(('_X', '_Y', '_Z')) for prefix in marker_prefixes)]
        ang_mom_cols = [col for col in df_train.columns if 'Angular_Momentum' in col]
        cols_to_filter = list(set(pos_cols + vel_cols + acc_cols + ang_mom_cols))
        if cols_to_filter:
            try:
                data_to_filter = df_train[cols_to_filter].values
                filtered_data = butter_lowpass_filter_multi(data_to_filter)
                df_train[cols_to_filter] = filtered_data
            except Exception as e:
                logging.error(f"Low-pass filtering error: {e}")
                return
        step_log(5, "Extracting gait and EMG synergy features from training data.")
        window_size = 99
        feature_combination = ['var', 'poly']
        gait_feats = extract_features(df_train, window_size, feature_combination, marker_prefixes)
        synergy_feats = extract_synergy_features(df_train, window_size)
        pca = PCA(n_components=5)
        X_gait_pca = pca.fit_transform(gait_feats)
        logging.info(f"PCA on gait features completed. Explained variance ratio: {pca.explained_variance_ratio_}")
        X_emg = synergy_feats.values
        markers_hmm = train_markers_hmm(
            X_gait_pca, n_components=5, n_iter=100,
            alpha=markers_tuned_params['alpha'],
            beta=markers_tuned_params['beta'],
            gamma=markers_tuned_params['gamma'],
            delta=markers_tuned_params['delta'],
            theta=markers_tuned_params['theta']
        )
        synergy_hmm = train_synergy_hmm(
            X_emg, n_components=5, n_iter=100,
            alpha=synergy_tuned_params['alpha'],
            beta=synergy_tuned_params['beta'],
            gamma=synergy_tuned_params['gamma'],
            delta=synergy_tuned_params['delta'],
            theta=synergy_tuned_params['theta']
        )

        step_log(10, "Predicting stride stages using pre-computed training features.")
        predictions = fuse_hmm_predictions(markers_hmm, synergy_hmm, X_gait_pca, X_emg)
        unique_states, counts = np.unique(predictions, return_counts=True)
        state_distribution = {state: count for state, count in zip(unique_states, counts)}
        logging.info(f"Predicted state distribution: {state_distribution}")
        mean_state = np.mean(predictions)
        logging.info(f"Mean predicted stride stage: {mean_state:.2f}")
        results = df_train[['Frame', 'Time (Seconds)']].copy()
        results['PredictedStage'] = predictions
        output_csv_path = "C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\predicted_stride_stages_final.csv"
        results.to_csv(output_csv_path, index=False)
        logging.info(f"Predicted stride stage segmentation saved to {output_csv_path}")
        step_log(11, "Script complete. Dynamic stride stage mapping is complete.")

    # --------------------------------------------------
    # EXECUTION: Tuning and Final Training
    # --------------------------------------------------
    os.makedirs("C:/Users/ek23yboj/RayResults", exist_ok=True)
    os.makedirs("C:/Users/ek23yboj/RayTemp", exist_ok=True)

    # Define a short trial name creator
    def short_name_creator(trial):
        return f"trial_{trial.trial_id[:5]}"

    # Initialize Ray
    ray.init(ignore_reinit_error=True, include_dashboard=False)

    # Define hyperparameter search space
    search_space = {
        "alpha": tune.uniform(0.05, 0.95),
        "beta": tune.uniform(0.05, 0.95),
        "gamma": tune.uniform(0.05, 0.95),
        "delta": tune.uniform(0.05, 0.95),
        "theta": tune.uniform(0.05, 0.95)
    }

    # Tuning for markers using HyperBand
    markers_scheduler = HyperBandScheduler(
        time_attr="training_iteration",
        max_t=100,
        metric="score",
        mode="max"

    )

    analysis_markers = tune.run(
        tune_pipeline_markers,
        config=search_space,
        scheduler=markers_scheduler,
        storage_path="C:/Users/ek23yboj/RayResults",
        trial_dirname_creator=short_name_creator,
        num_samples=20,
        reuse_actors=True,
        verbose=1
    )
    markers_tuned_params = analysis_markers.get_best_config(metric="score", mode="max")
    print("Best markers hyperparameters (HyperBand):", markers_tuned_params)

    # Tuning for synergy using ASHA
    synergy_scheduler = ASHAScheduler(
        time_attr="training_iteration",
        max_t=100,
        metric="score",
        mode="max"
    )

    analysis_synergy = tune.run(
        tune_pipeline_synergy,
        config=search_space,
        scheduler=synergy_scheduler,
        storage_path="C:/Users/ek23yboj/RayResults",
        trial_dirname_creator=short_name_creator,
        num_samples=20,
        reuse_actors=True,
        verbose=1

    )
    synergy_tuned_params = analysis_synergy.get_best_config(metric="score", mode="max")
    print("Best synergy hyperparameters (ASHA):", synergy_tuned_params)

    # Run final training using the tuned hyperparameters
    final_training(markers_tuned_params, synergy_tuned_params)

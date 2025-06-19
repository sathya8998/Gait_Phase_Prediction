import logging
import numpy as np
import torch
import lime
import lime.lime_tabular
from pathlib import Path
from scipy.interpolate import interp1d
from tqdm import tqdm
from tslearn.metrics import dtw_path
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Base Manager
# -------------------------------------------------------------------
class BaseManager:
    pass

# -------------------------------------------------------------------
# 1. Model Interpretability (LIME Analysis)
# -------------------------------------------------------------------
class ModelInterpretability:
    @staticmethod
    def lime_analysis(model, X_train, y_train, X_test, y_test, feature_names, model_type):
        try:
            save_dir = Path(r"C:\Users\ek23yboj\PycharmProjects\PythonProject\Gait_Phase\Data")

            if model_type in ['LSTM', 'CNN', 'Transformer']:
                #  prediction function
                def predict_fn_3d(data_2d):
                    n_samples = data_2d.shape[0]
                    win_size = X_test.shape[1]
                    n_feats = X_test.shape[2]
                    # Reshape back to 3D: (n_samples, win_size, n_feats)
                    X_input = data_2d.reshape(n_samples, win_size, n_feats)
                    model.eval()
                    with torch.no_grad():
                        tensor_input = torch.from_numpy(X_input).float()
                        # Move to model device
                        device = next(model.parameters()).device
                        tensor_input = tensor_input.to(device)
                        predictions = model(tensor_input)
                        return predictions.cpu().numpy()

                X_train_lime = X_train.reshape(X_train.shape[0], -1)
                X_test_lime = X_test.reshape(X_test.shape[0], -1)
                # generate feature names
                feature_names = [f'Feature_{i+1}' for i in range(X_train_lime.shape[1])]
                explain_func = predict_fn_3d
            else:
                logger.warning(f"Model type '{model_type}' is not supported for LIME analysis.")
                return

            if len(feature_names) != X_train_lime.shape[1]:
                logger.warning("Mismatch between feature_names and data shape. Adjusting automatically.")
                feature_names = [f'Feature_{i+1}' for i in range(X_train_lime.shape[1])]

            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=X_train_lime,
                feature_names=feature_names,
                class_names=[f"Class {cls}" for cls in np.unique(y_train)],
                mode='classification'
            )

            sample_size = min(3, len(X_test_lime))
            if sample_size == 0:
                logger.warning("No samples available for LIME analysis.")
                return

            np.random.seed(42)
            indices = np.random.choice(len(X_test_lime), size=sample_size, replace=False)

            for sample_idx in indices:
                explanation = explainer.explain_instance(
                    X_test_lime[sample_idx],
                    explain_func,
                    num_features=10,
                    top_labels=1
                )
                filename = save_dir / f"LIME_Explanation_{model_type}_Sample_{sample_idx}.html"
                explanation.save_to_file(str(filename))
                logger.info(f"Saved LIME explanation for {model_type}, Sample {sample_idx} -> {filename}")
        except Exception as e:
            logger.error(f"Error during LIME analysis for {model_type}: {e}", exc_info=True)
# -------------------------------------------------------------------
# 2. Utility Manager for Sample Selection
# -------------------------------------------------------------------
class UtilityManager(BaseManager):
    @staticmethod
    def select_three_samples_per_class(X, y, samples_per_class=3, num_classes=None):
        try:
            selected_X = []
            selected_y = []
            unique_classes = np.unique(y)

            if num_classes is not None and len(unique_classes) != num_classes:
                logger.warning(f"Expected {num_classes} classes, found {len(unique_classes)} unique classes.")

            np.random.seed(42)

            for cls in unique_classes:
                cls_indices = np.where(y == cls)[0]
                if len(cls_indices) >= samples_per_class:
                    selected_indices = np.random.choice(cls_indices, samples_per_class, replace=False)
                elif len(cls_indices) > 0:
                    selected_indices = cls_indices
                    logger.warning(f"Class {cls} has fewer than {samples_per_class} samples. Selecting all.")
                else:
                    logger.warning(f"Class {cls} has no samples. Skipping.")
                    continue

                selected_X.append(X[selected_indices])
                selected_y.extend([cls] * len(selected_indices))
                logger.info(f"Selected {len(selected_indices)} samples for class {cls}.")

            if selected_X:
                return np.vstack(selected_X), np.array(selected_y)
            else:
                logger.error("No samples were selected. Please check the input data.")
                return np.array([]), np.array([])
        except Exception as e:
            logger.error(f"Error selecting samples: {e}")
            return None, None

# -------------------------------------------------------------------
# 3. Feature Extraction with Sequence Creation and DTW Alignment
# -------------------------------------------------------------------
class FeatureExtractor:
    @staticmethod
    def create_sequences(data, window_size, overlap=0.99, labels=None):
        try:
            step = max(int(window_size * (1 - overlap)), 1)
            sequences = []
            sequence_labels = [] if labels is not None else None
            center_indices = []
            for i in tqdm(range(0, len(data) - window_size + 1, step), desc="Creating Sequences"):
                seq = data[i : i + window_size]
                sequences.append(seq)
                center_index = i + window_size // 2
                center_indices.append(center_index)
                if labels is not None:
                    sequence_labels.append(labels[center_index])
            sequences = np.array(sequences)
            if labels is not None:
                sequence_labels = np.array(sequence_labels)
                logger.info(f"Created sequences: {sequences.shape} with sequence labels: {sequence_labels.shape}")
                return sequences, sequence_labels, center_indices
            else:
                logger.info(f"Created sequences: {sequences.shape}")
                return sequences, None, center_indices
        except Exception as e:
            logger.error(f"Error during sequence creation: {e}")
            return None, None, None

    @staticmethod
    def align_sequences_with_dtw(sequences, target_length=None):
        try:
            if sequences is None or sequences.ndim != 3:
                logger.error("Invalid input to align_sequences_with_dtw. Must be 3D array.")
                return sequences

            scaler = TimeSeriesScalerMeanVariance()
            num_sequences, window_size, num_features = sequences.shape
            sequences = scaler.fit_transform(sequences)
            reference = sequences[0]
            reference_length = reference.shape[0]
            if target_length is None:
                target_length = reference_length

            aligned = []
            for idx, seq in enumerate(sequences):
                if idx == 0:
                    aligned.append(seq)
                    continue
                if seq.ndim != 2 or reference.ndim != 2:
                    logger.error(f"Sequence {idx} or reference is not 2D. Skipping.")
                    aligned.append(seq)
                    continue
                try:
                    alignment_path, distance = dtw_path(seq, reference, global_constraint="sakoe_chiba", sakoe_chiba_radius=1)
                except Exception as e:
                    logger.error(f"DTW error on sequence {idx}: {e}")
                    aligned.append(seq)
                    continue

                seq_indices, ref_indices = zip(*alignment_path)
                seq_indices = np.array(seq_indices)
                ref_indices = np.array(ref_indices)
                weight_matrix = np.ones_like(seq_indices)
                diagonal_moves = (np.diff(seq_indices) == 1) & (np.diff(ref_indices) == 1)
                weight_matrix[1:][diagonal_moves] = 2

                unique_ref, unique_inds = np.unique(ref_indices, return_index=True)
                unique_seq = seq_indices[unique_inds]
                weight_matrix = weight_matrix[unique_inds]
                if len(unique_ref) < 2:
                    logger.warning(f"Not enough unique points for interpolation in seq {idx}. Using original.")
                    aligned.append(seq)
                    continue

                try:
                    f = interp1d(unique_ref, unique_seq, kind='linear', fill_value="extrapolate")
                except Exception as e:
                    logger.error(f"Error creating interpolation function for seq {idx}: {e}")
                    aligned.append(seq)
                    continue

                new_ref_indices = np.linspace(0, reference_length - 1, target_length)
                try:
                    new_seq_indices = f(new_ref_indices)
                except Exception as e:
                    logger.error(f"Interpolation error for seq {idx}: {e}")
                    aligned.append(seq)
                    continue

                new_seq_indices = np.round(new_seq_indices).astype(int)
                new_seq_indices = np.clip(new_seq_indices, 0, window_size - 1)
                warped_seq = seq[new_seq_indices]
                aligned.append(warped_seq)

            aligned = np.array(aligned)
            logger.info(f"Aligned sequences shape: {aligned.shape}")
            return aligned
        except Exception as e:
            logger.error(f"Error during DTW alignment: {e}")
            return sequences


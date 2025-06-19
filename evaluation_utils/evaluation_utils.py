import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import logging
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# 1. Adjust Predicted Probabilities
# -------------------------------------------------------------------
def adjust_pred_proba(y_pred_proba, num_classes):
    current_classes = y_pred_proba.shape[1]
    if current_classes < num_classes:
        padding = np.zeros((y_pred_proba.shape[0], num_classes - current_classes))
        y_pred_proba = np.hstack([y_pred_proba, padding])
    elif current_classes > num_classes:
        y_pred_proba = y_pred_proba[:, :num_classes]
    return y_pred_proba


# -------------------------------------------------------------------
# 2. Prepare ROC AUC
# -------------------------------------------------------------------
def prepare_roc_auc(y_true, y_pred_proba, num_classes, label_encoder=None):
    # Binarize y_true
    y_true_binarized = label_binarize(y_true, classes=range(num_classes))

    # Adjust y_pred_proba if needed
    if y_pred_proba.shape[1] < num_classes:
        padding = np.zeros((y_pred_proba.shape[0], num_classes - y_pred_proba.shape[1]))
        y_pred_proba_padded = np.hstack([y_pred_proba, padding])
    elif y_pred_proba.shape[1] > num_classes:
        y_pred_proba_padded = y_pred_proba[:, :num_classes]
    else:
        y_pred_proba_padded = y_pred_proba

    return y_true_binarized, y_pred_proba_padded


# -------------------------------------------------------------------
# 3. Compute ROC AUC
# -------------------------------------------------------------------
def compute_roc_auc(y_true, y_pred_prob, num_classes, save_path):
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        y_true_binarized = label_binarize(y_true, classes=range(num_classes))
        if y_true_binarized.shape[1] == 1:
            y_true_binarized = np.hstack([1 - y_true_binarized, y_true_binarized])
            logger.warning("Only one class present in y_true. ROC AUC might not be informative.")

        roc_auc = roc_auc_score(
            y_true_binarized,
            y_pred_prob,
            average='weighted',
            multi_class='ovr'
        )
        logger.info(f"ROC AUC Score: {roc_auc:.4f}")

        fpr = dict()
        tpr = dict()
        roc_auc_dict = dict()
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_pred_prob[:, i])
            roc_auc_dict[i] = auc(fpr[i], tpr[i])

        plt.figure(figsize=(10, 8))
        for i in range(num_classes):
            plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc_dict[i]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC AUC - Multi-class')
        plt.legend(loc='lower right')
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved ROC AUC plot as '{save_path}'")

        return roc_auc
    except Exception as e:
        logger.error(f"Error computing ROC AUC: {e}")
        return None


# -------------------------------------------------------------------
# 4. Custom PyTorch Metric for F1 Score
# -------------------------------------------------------------------
class F1Score(nn.Module):
    def __init__(self, num_classes=5, device=torch.device("cpu")):
        super(F1Score, self).__init__()
        self.num_classes = num_classes
        self.epsilon = 1e-7
        self.device = device
        self.reset_states()

    def update_state(self, y_true, y_pred):
        # Ensure the inputs are on the correct device.
        y_true = y_true.to(self.device)
        y_pred = y_pred.to(self.device)

        y_pred_labels = torch.argmax(y_pred, dim=1)
        y_true_one_hot = F.one_hot(y_true, num_classes=self.num_classes).float().to(self.device)
        y_pred_one_hot = F.one_hot(y_pred_labels, num_classes=self.num_classes).float().to(self.device)

        tp = (y_true_one_hot * y_pred_one_hot).sum(dim=0)
        fp = ((1 - y_true_one_hot) * y_pred_one_hot).sum(dim=0)
        fn = (y_true_one_hot * (1 - y_pred_one_hot)).sum(dim=0)

        self.true_positives += tp
        self.false_positives += fp
        self.false_negatives += fn

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + self.epsilon)
        recall = self.true_positives / (self.true_positives + self.false_negatives + self.epsilon)
        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        weights = self.true_positives + self.false_negatives
        sum_weights = torch.sum(weights) + self.epsilon
        weights = weights / sum_weights
        weighted_f1 = torch.sum(f1 * weights)
        return weighted_f1.item()

    def reset_states(self):
        self.true_positives = torch.zeros(self.num_classes, device=self.device)
        self.false_positives = torch.zeros(self.num_classes, device=self.device)
        self.false_negatives = torch.zeros(self.num_classes, device=self.device)

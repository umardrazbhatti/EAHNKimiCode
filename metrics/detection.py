"""
metrics/detection.py — AUC-ROC, AUC-PR, F1 for binary deepfake detection.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import warnings


def compute_detection_metrics(probs, labels) -> dict:
    labels = np.array(labels, dtype=int)
    probs  = np.array(probs,  dtype=float)

    unique = np.unique(labels)
    if len(unique) < 2:
        warnings.warn(
            f"Only class(es) {unique.tolist()} present in labels. "
            "AUC-ROC and AUC-PR are undefined; returning NaN."
        )
        return {"auc_roc": float("nan"), "auc_pr": float("nan"), "f1": 0.0}

    preds = (probs >= 0.5).astype(int)
    return {
        "auc_roc": float(roc_auc_score(labels, probs)),
        "auc_pr":  float(average_precision_score(labels, probs)),
        "f1":      float(f1_score(labels, preds, zero_division=0)),
    }


class DetectionMetrics:
    @staticmethod
    def compute(probs, labels) -> dict:
        return compute_detection_metrics(probs, labels)

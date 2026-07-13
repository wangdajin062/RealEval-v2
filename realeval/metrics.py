"""realeval/metrics.py — Classification Metrics

Standard sklearn-based metrics for fraud detection evaluation.
"""
from __future__ import annotations
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def classification_metrics(y_true: list, y_pred: list, y_score: list = None) -> dict:
    """Compute standard classification metrics. Returns accuracy, f1, precision, recall, fpr, auc.

    AUC requires continuous scores: pass y_score (probabilities/logits) to get a meaningful AUC.
    If only hard 0/1 predictions are available, AUC is set to None (a hard-prediction ROC degenerates
    to a single point, so roc_auc_score on 0/1 labels would be misleading).
    """
    # All metrics use average="binary" because fraud detection is a binary
    # classification problem. For multi-class extensions, this would need to
    # be parameterized to the appropriate average strategy.
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred, average="binary", zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, average="binary", zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, average="binary", zero_division=0)), 4),
    }
    # False-positive rate = FP / (FP + TN), computed from the confusion matrix.
    # Auto-detect class labels (handles int/str/float) instead of hardcoding [0, 1].
    try:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["fpr"] = round(float(fp / (fp + tn)), 4) if (fp + tn) > 0 else 0.0
        else:
            metrics["fpr"] = None  # multi-class or degenerate
    except Exception:
        metrics["fpr"] = None
    # AUC only from continuous scores; None (not a degenerate value) when only hard preds are given.
    if y_score is not None:
        try:
            metrics["auc"] = round(float(roc_auc_score(y_true, y_score)), 4)
        except Exception:
            metrics["auc"] = None
    else:
        metrics["auc"] = None
    return metrics

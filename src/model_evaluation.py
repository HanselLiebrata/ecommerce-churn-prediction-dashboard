from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn import metrics


def evaluate_predictions(y_true, y_prob) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    acc = metrics.accuracy_score(y_true, y_pred)
    roc_auc = metrics.roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    pr_auc = metrics.average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
    }


def confusion_matrix(y_true, y_prob) -> Tuple[int, int, int, int]:
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    return int(tn), int(fp), int(fn), int(tp)



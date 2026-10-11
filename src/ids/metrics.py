"""Evaluation metrics recommended in the paper review: beyond accuracy (macro-F1, MCC, PR-AUC, FPR, latency)."""
import time

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)


def evaluate(y_true, proba, n_classes):
    pred = proba.argmax(1)
    m = {"accuracy": accuracy_score(y_true, pred),
         "precision_macro": precision_score(y_true, pred, average="macro", zero_division=0),
         "recall_macro": recall_score(y_true, pred, average="macro", zero_division=0),
         "f1_macro": f1_score(y_true, pred, average="macro"),
         "mcc": matthews_corrcoef(y_true, pred)}
    if n_classes == 2:
        tn = np.sum((y_true == 0) & (pred == 0))
        fp = np.sum((y_true == 0) & (pred == 1))
        m["fpr"] = fp / (fp + tn)
        m["roc_auc"] = roc_auc_score(y_true, proba[:, 1])
        m["pr_auc"] = average_precision_score(y_true, proba[:, 1])
    else:
        # False-positive rate on Normal traffic = Normal flows flagged as any attack (class 0 = Normal).
        normal = y_true == 0
        m["fpr"] = np.mean(pred[normal] != 0)
        m["roc_auc"] = roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
    return m


def latency_us_per_flow(predict_fn, X, n=20000, repeats=3):
    """Median per-flow latency in microseconds for batch inference, measured after a warm-up call."""
    Xs = X[:n]
    predict_fn(Xs[:256])
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        predict_fn(Xs)
        times.append(time.perf_counter() - t0)
    return 1e6 * float(np.median(times)) / len(Xs)

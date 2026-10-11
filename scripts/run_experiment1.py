"""Experiment 1: model comparison on UNSW-NB15 (binary and 10-class).

Usage:  python scripts/run_experiment1.py [--task binary|multiclass|both] [--epochs 30]
Writes: results/exp1_<task>.csv, results/exp1_multiclass_per_class_f1.csv, figures/*.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402

from ids.data import load_splits  # noqa: E402
from ids.metrics import evaluate, latency_us_per_flow  # noqa: E402
from ids.models import CLASSIFIERS, Autoencoder  # noqa: E402
from ids.train import autoencoder_scores, fit_autoencoder, fit_classifier, predict_proba  # noqa: E402

RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
SEED = 42


def sklearn_models():
    return {
        "LogReg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1,
                                               class_weight="balanced_subsample", random_state=SEED),
        "GBDT (HistGB)": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                                        class_weight="balanced", random_state=SEED),
    }


def run_task(task, epochs):
    print(f"\n=== Task: {task} ===")
    s = load_splits(task, seed=SEED)
    k = len(s.classes)
    print(f"train {s.X_train.shape}, val {s.X_val.shape}, test {s.X_test.shape}, classes {s.classes}")
    rows, probas = [], {}

    def record(name, family, proba, train_s, predict_fn):
        m = evaluate(s.y_test, proba, k)
        m.update(model=name, family=family, train_time_s=round(train_s, 1),
                 latency_us_per_flow=round(latency_us_per_flow(predict_fn, s.X_test), 2))
        rows.append(m)
        probas[name] = proba
        print(f"  {name:14s} acc {m['accuracy']:.4f}  macro-F1 {m['f1_macro']:.4f}  MCC {m['mcc']:.4f}  FPR {m['fpr']:.4f}")

    import time
    for name, clf in sklearn_models().items():
        print(f"  training {name} ...")
        t0 = time.time()
        clf.fit(s.X_train, s.y_train)
        record(name, "Classical ML", clf.predict_proba(s.X_test), time.time() - t0, clf.predict_proba)

    for name, cls in CLASSIFIERS.items():
        print(f"  training {name} ...")
        model, t = fit_classifier(cls(s.X_train.shape[1], k), s, epochs=epochs, seed=SEED,
                                  log=lambda msg: None)
        model.cpu()  # latency measured on CPU, the realistic deployment target
        record(name, "Neural", predict_proba(model, s.X_test), t, lambda X, m=model: predict_proba(m, X))
        torch.save(model.state_dict(), RESULTS / f"{task}_{name}.pt")

    ensemble_members = ["GBDT (HistGB)", "RandomForest", "CNN-LSTM", "MLP"]
    proba = np.mean([probas[n] for n in ensemble_members], 0)
    m = evaluate(s.y_test, proba, k)
    m.update(model="Ensemble (soft vote)", family="Hybrid ensemble",
             train_time_s=round(sum(r["train_time_s"] for r in rows if r["model"] in ensemble_members), 1),
             latency_us_per_flow=round(sum(r["latency_us_per_flow"] for r in rows if r["model"] in ensemble_members), 2))
    rows.append(m)
    probas[m["model"]] = proba
    print(f"  {'Ensemble':14s} acc {m['accuracy']:.4f}  macro-F1 {m['f1_macro']:.4f}  MCC {m['mcc']:.4f}  FPR {m['fpr']:.4f}")

    if task == "binary":
        rows.append(run_autoencoder(s))

    df = pd.DataFrame(rows)
    cols = ["model", "family", "accuracy", "precision_macro", "recall_macro", "f1_macro", "mcc", "fpr",
            "roc_auc", "pr_auc", "train_time_s", "latency_us_per_flow"]
    df = df[[c for c in cols if c in df.columns]].round(4)
    df.to_csv(RESULTS / f"exp1_{task}.csv", index=False)

    if task == "multiclass":
        per_class = {n: f1_score(s.y_test, p.argmax(1), average=None, labels=range(k)) for n, p in probas.items()}
        pc = pd.DataFrame(per_class, index=s.classes).round(4)
        pc.insert(0, "test_support", np.bincount(s.y_test, minlength=k))
        pc.to_csv(RESULTS / "exp1_multiclass_per_class_f1.csv")
        best = df.sort_values("f1_macro").iloc[-1]["model"]
        plot_confusion(s.y_test, probas[best].argmax(1), s.classes, best)
    return df


def run_autoencoder(s):
    """Unsupervised anomaly detection: train on Normal flows only, never sees an attack label."""
    print("  training Autoencoder (normal traffic only) ...")
    ae, t = fit_autoencoder(Autoencoder(s.X_train.shape[1]), s.X_train[s.y_train == 0],
                            s.X_val[s.y_val == 0], seed=SEED)
    ae.cpu()
    thr = np.percentile(autoencoder_scores(ae, s.X_val[s.y_val == 0]), 95)  # 5% FPR budget on validation
    score = autoencoder_scores(ae, s.X_test)
    pred = (score > thr).astype(int)
    proba = np.stack([1 - pred, pred], 1).astype(float)
    m = evaluate(s.y_test, proba, 2)
    from sklearn.metrics import average_precision_score, roc_auc_score
    m["roc_auc"], m["pr_auc"] = roc_auc_score(s.y_test, score), average_precision_score(s.y_test, score)
    m.update(model="Autoencoder (unsupervised)", family="Anomaly detection", train_time_s=round(t, 1),
             latency_us_per_flow=round(latency_us_per_flow(lambda X: autoencoder_scores(ae, X), s.X_test), 2))
    torch.save(ae.state_dict(), RESULTS / "binary_Autoencoder.pt")
    print(f"  {'Autoencoder':14s} acc {m['accuracy']:.4f}  macro-F1 {m['f1_macro']:.4f}  ROC-AUC {m['roc_auc']:.4f}")
    return m


def plot_confusion(y, pred, classes, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y, pred, labels=range(len(classes)), normalize="true")
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > 0.6 else "#1f2933")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"UNSW-NB15 10-class confusion matrix (row-normalised) — {name}", fontsize=10)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIGURES / "exp1_confusion_multiclass.png", dpi=200)
    plt.close(fig)


def plot_comparison(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4.2), squeeze=False)
    for ax, (task, df) in zip(axes[0], results.items()):
        d = df.sort_values("f1_macro")
        color = d["family"].map({"Classical ML": "#8a9bb0", "Neural": "#1d4e89",
                                 "Hybrid ensemble": "#067647", "Anomaly detection": "#b54708"})
        ax.barh(d["model"], d["f1_macro"], color=color)
        for y, v in enumerate(d["f1_macro"]):
            ax.text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=8)
        ax.set_xlim(max(0, d["f1_macro"].min() - 0.1), 1.0)
        ax.set_xlabel("Macro-F1 on official test set")
        ax.set_title(f"UNSW-NB15 — {task}", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "exp1_model_comparison.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="both", choices=["binary", "multiclass", "both"])
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
    np.random.seed(SEED)
    tasks = ["binary", "multiclass"] if args.task == "both" else [args.task]
    results = {t: run_task(t, args.epochs) for t in tasks}
    plot_comparison(results)
    for t, df in results.items():
        print(f"\n{t}\n{df.to_string(index=False)}")
    (RESULTS / "exp1_config.json").write_text(json.dumps(
        {"seed": SEED, "epochs_max": args.epochs, "dataset": "UNSW-NB15 (official train/test partitions)",
         "val_split": "15% stratified from training set", "torch": torch.__version__}, indent=2))

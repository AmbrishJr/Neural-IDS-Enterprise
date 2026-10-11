"""Training loops: supervised classifiers (early stopping on validation macro-F1) and autoencoder."""
import copy
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _loader(X, y=None, batch=512, shuffle=False):
    tensors = [torch.from_numpy(X)] + ([torch.from_numpy(y)] if y is not None else [])
    return DataLoader(TensorDataset(*tensors), batch_size=batch, shuffle=shuffle)


@torch.no_grad()
def predict_proba(model, X, batch=4096):
    model.eval()
    dev = next(model.parameters()).device
    out = [torch.softmax(model(xb.to(dev)), 1).cpu() for (xb,) in _loader(X, batch=batch)]
    return torch.cat(out).numpy()


def class_weights(y, n_classes):
    """Inverse-sqrt frequency weights: softens imbalance without over-boosting tiny classes (Worms)."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    w = 1.0 / np.sqrt(np.maximum(counts, 1))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def fit_classifier(model, s, epochs=30, lr=1e-3, patience=5, batch=512, seed=42, log=print):
    torch.manual_seed(seed)
    dev = device()
    model.to(dev)
    n_classes = len(s.classes)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(s.y_train, n_classes).to(dev))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    best, best_state, bad = -1.0, None, 0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for xb, yb in _loader(s.X_train, s.y_train, batch, shuffle=True):
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        f1 = f1_score(s.y_val, predict_proba(model, s.X_val).argmax(1), average="macro")
        sched.step(f1)
        if f1 > best + 1e-4:
            best, best_state, bad = f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        log(f"    epoch {ep + 1:2d}  val macro-F1 {f1:.4f}")
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    return model, time.time() - t0


def fit_autoencoder(model, X_normal, X_val_normal, epochs=40, lr=1e-3, patience=5, batch=512, seed=42):
    torch.manual_seed(seed)
    dev = device()
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, bad = float("inf"), None, 0
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        for (xb,) in _loader(X_normal, batch=batch, shuffle=True):
            xb = xb.to(dev)
            opt.zero_grad()
            model.score(xb).mean().backward()
            opt.step()
        val = autoencoder_scores(model, X_val_normal).mean()
        if val < best - 1e-5:
            best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, time.time() - t0


@torch.no_grad()
def autoencoder_scores(model, X, batch=4096):
    model.eval()
    dev = next(model.parameters()).device
    return torch.cat([model.score(xb.to(dev)).cpu() for (xb,) in _loader(X, batch=batch)]).numpy()

"""Data loading and leakage-safe preprocessing for UNSW-NB15.

Note on file naming: the Hugging Face mirror (wwydmanski/UNSW-NB15, from Kaggle dhoogla/unswnb15)
labels the 82,332-row file "train" and the 175,341-row file "test". The original authors
(Moustafa & Slay, 2015) define the 175,341-row partition as the TRAINING set and the 82,332-row
partition as the TESTING set. We follow the original definition.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "unsw_nb15"

CATEGORICAL = ["proto", "service", "state"]
LABEL, ATTACK_CAT = "label", "attack_cat"
# Identifier-like TCP sequence numbers carry no behavioural signal.
DROP = ["stcpb", "dtcpb"]
TOP_K_CATEGORIES = 12


def load_raw():
    train = pd.read_parquet(RAW / "test.parquet")   # 175,341 rows = official training set
    test = pd.read_parquet(RAW / "train.parquet")   # 82,332 rows = official testing set
    for df in (train, test):
        for c in CATEGORICAL + [ATTACK_CAT]:
            df[c] = df[c].astype(str).str.strip()
    return train, test


@dataclass
class Preprocessor:
    """Fit on training data only; applied identically to val/test."""
    numeric: list = None
    categories: dict = None
    mean: np.ndarray = None
    std: np.ndarray = None

    def fit(self, df):
        self.numeric = [c for c in df.columns if c not in CATEGORICAL + DROP + [LABEL, ATTACK_CAT]]
        self.categories = {c: df[c].value_counts().index[:TOP_K_CATEGORIES].tolist() for c in CATEGORICAL}
        x = self._numeric(df)
        self.mean, self.std = x.mean(0), x.std(0) + 1e-6
        return self

    def _numeric(self, df):
        # Heavy-tailed byte/packet/rate counts -> signed log1p before standardisation.
        x = df[self.numeric].to_numpy(dtype=np.float64)
        return np.sign(x) * np.log1p(np.abs(x))

    def transform(self, df):
        num = (self._numeric(df) - self.mean) / self.std
        onehots = []
        for c, cats in self.categories.items():
            col = df[c].where(df[c].isin(cats), "other")
            onehots.append(np.stack([(col == k).to_numpy() for k in cats + ["other"]], 1))
        return np.hstack([num] + onehots).astype(np.float32)

    @property
    def feature_names(self):
        return self.numeric + [f"{c}={k}" for c, cats in self.categories.items() for k in cats + ["other"]]


@dataclass
class Splits:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    classes: list
    feature_names: list
    n_numeric: int


def load_splits(task="binary", val_size=0.15, seed=42):
    """task='binary' (normal vs attack) or 'multiclass' (10 attack categories)."""
    train, test = load_raw()
    if task == "binary":
        classes = ["Normal", "Attack"]
        y_tr, y_te = train[LABEL].to_numpy(), test[LABEL].to_numpy()
    else:
        classes = sorted(train[ATTACK_CAT].unique(), key=lambda c: (c != "Normal", c))
        idx = {c: i for i, c in enumerate(classes)}
        y_tr, y_te = train[ATTACK_CAT].map(idx).to_numpy(), test[ATTACK_CAT].map(idx).to_numpy()

    tr_df, va_df, y_tr, y_va = train_test_split(train, y_tr, test_size=val_size,
                                                stratify=train[ATTACK_CAT], random_state=seed)
    pre = Preprocessor().fit(tr_df)
    return Splits(pre.transform(tr_df), pre.transform(va_df), pre.transform(test),
                  y_tr.astype(np.int64), y_va.astype(np.int64), y_te.astype(np.int64),
                  classes, pre.feature_names, len(pre.numeric))

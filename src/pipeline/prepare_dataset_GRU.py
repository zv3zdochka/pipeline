from __future__ import annotations

import pathlib
import joblib
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

EVENT_COLS = [
    "whale_position_unrealizedPnl",
    "whale_position_size",
    "whale_position_liquidationPx",
    "whale_position_marginUsed",
    "whale_position_leverage",
    "whale_position_entryPx",
    "whale_position_positionValue",
    "whale_position_cumFunding",
    "liquidation_history_longTurnover",
    "liquidation_history_shortTurnover"
]

MISSING_COLS = [f"{c}_was_missing" for c in EVENT_COLS]

FEATURE_COLS = EVENT_COLS + MISSING_COLS


class GRUSequenceDataset(Dataset):
    """
    A PyTorch Dataset that provides fixed-length time series sequences
    and their corresponding labels for GRU training or evaluation.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray, seq_len: int):
        self.features = features
        self.labels = labels
        self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len + 1

    def __getitem__(self, idx):
        x = self.features[idx: idx + self.seq_len]
        y = self.labels[idx + self.seq_len - 1]
        return torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.long)


def prepare_gru_dataset(
        events_pkl: str | pathlib.Path,
        emb_path: str | pathlib.Path,
        seq_len: int = 96,
        train_frac: float = 0.8,
        scaler_path: str | pathlib.Path | None = None,
        dataset_train_path: str | pathlib.Path | None = None,
        dataset_test_path: str | pathlib.Path | None = None
):
    """
    Load event data and CNN embeddings, split chronologically into train and test sets,
    fit a StandardScaler on the training features, transform both sets, build
    GRUSequenceDataset instances, and optionally save scaler and datasets.

    Returns
    -------
    train_dataset : GRUSequenceDataset
        Dataset for training.
    test_dataset : GRUSequenceDataset
        Dataset for testing.
    scaler : StandardScaler
        Fitted scaler on training data.
    """
    print(f"[GRU] Loading events from {events_pkl}")
    events = joblib.load(events_pkl)
    events = events.set_index("ts").sort_index()
    print(f"[GRU] Events loaded, total records: {len(events)}")

    # Ensure all columns exist
    for c in EVENT_COLS:
        if c not in events.columns:
            events[c] = 0.0
    for c in MISSING_COLS:
        if c not in events.columns:
            events[c] = 0

    needed = FEATURE_COLS + ["microtrend_label"]
    df_ev = events[needed].copy()
    print(f"[GRU] Filtered events to needed columns, shape: {df_ev.shape}")

    emb_path = pathlib.Path(emb_path)
    if emb_path.suffix == ".parquet":
        print(f"[GRU] Loading embeddings from parquet: {emb_path}")
        emb = pd.read_parquet(emb_path)
    else:
        print(f"[GRU] Loading embeddings from CSV: {emb_path}")
        emb = pd.read_csv(emb_path, index_col=0, parse_dates=True)
    emb = emb.sort_index()
    print(f"[GRU] Embeddings loaded, total records: {len(emb)}")

    df_all = df_ev.join(emb, how="inner").dropna()
    print(f"[GRU] Joined events and embeddings, resulting shape: {df_all.shape}")

    n_total = len(df_all)
    split_idx = int(n_total * train_frac)
    df_train = df_all.iloc[:split_idx]
    df_test = df_all.iloc[split_idx:]
    print(f"[GRU] Split into train ({len(df_train)}) / test ({len(df_test)})")

    feature_columns = EVENT_COLS + MISSING_COLS + emb.columns.tolist()
    scaler = StandardScaler()
    scaler.fit(df_train[feature_columns].values)

    if scaler_path:
        joblib.dump(scaler, scaler_path)

        print(f"[GRU] Saved scaler to {scaler_path}")

    X_train = scaler.transform(df_train[feature_columns].values)
    X_test = scaler.transform(df_test[feature_columns].values)

    y_train = (df_train["microtrend_label"] + 1).astype("int64").values
    y_test = (df_test["microtrend_label"] + 1).astype("int64").values

    train_dataset = GRUSequenceDataset(X_train, y_train, seq_len)
    test_dataset = GRUSequenceDataset(X_test, y_test, seq_len)

    if dataset_train_path:
        joblib.dump(train_dataset, dataset_train_path)
        print(f"[GRU] Saved training dataset to {dataset_train_path}")
    if dataset_test_path:
        joblib.dump(test_dataset, dataset_test_path)
        print(f"[GRU] Saved test dataset to {dataset_test_path}")

    print("[GRU] Dataset preparation complete")
    return train_dataset, test_dataset, scaler

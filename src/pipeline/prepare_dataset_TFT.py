# src/pipeline/prepare_dataset_TFT.py
from __future__ import annotations

import pathlib
import warnings
from typing import Sequence, Mapping

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class TFTDataset(Dataset):
    """
    A PyTorch Dataset for TFT that generates sliding windows on-the-fly.
    Each item is a tuple (X_window, y_target):
      - X_window: Tensor of shape [seq_len, D]
      - y_target: Tensor scalar (int/float)
    """
    def __init__(self,
                 features: np.ndarray,
                 targets: np.ndarray,
                 seq_len: int):
        self.features = torch.from_numpy(features).float()
        self.targets = torch.from_numpy(targets).float()
        self.seq_len = seq_len
        self.n_windows = len(self.targets) - seq_len

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
        return x, y


def prepare_tft_dataset(
        df_events: pd.DataFrame | str | pathlib.Path,
        cnn_emb_path: str | pathlib.Path,
        gru_emb_path: str | pathlib.Path,
        timesnet_emb_path: str | pathlib.Path,
        timesnet_pred_path: str | pathlib.Path,
        *,
        seq_len: int = 96,
        target_col: str = "microtrend_label",
        scaler_path: str | pathlib.Path = "tft_scaler.pkl",
        dataset_path: str | pathlib.Path = "tft_dataset.pt",
        train_size: float = 0.8,
        random_state: int = 42,
        train_features_path: str | pathlib.Path = "tft_train_features.npy",
        train_targets_path: str | pathlib.Path = "tft_train_targets.npy",
        test_features_path: str | pathlib.Path = "tft_test_features.npy",
        test_targets_path: str | pathlib.Path = "tft_test_targets.npy",
        strict: bool = False,
) -> tuple[TFTDataset, Mapping[str, Sequence[str]]]:
    print("[TFT] DATA PREP STARTED")
    print("[TFT] Loading events data...")
    if isinstance(df_events, (str, pathlib.Path)):
        df_events = (
            joblib.load(df_events)
            if str(df_events).endswith(".pkl")
            else pd.read_csv(df_events, parse_dates=["ts"])
        )
    df_events = df_events.set_index("ts").sort_index()

    print("[TFT] Loading embeddings and forecasts...")
    def _read_parquet_any(path):
        p = pathlib.Path(path)
        return (pd.read_parquet(p) if p.suffix == ".parquet"
                else pd.read_csv(p, index_col=0, parse_dates=True))

    df_cnn     = _read_parquet_any(cnn_emb_path).add_prefix("cnn_emb_")
    df_gru     = _read_parquet_any(gru_emb_path).add_prefix("gru_emb_")
    df_tn_emb  = _read_parquet_any(timesnet_emb_path).add_prefix("timesnet_emb_")
    df_tn_pred = _read_parquet_any(timesnet_pred_path)

    print("[TFT] Joining dataframes (left-join)...")
    df_all = df_events.copy()
    for extra in (df_cnn, df_gru, df_tn_emb, df_tn_pred):
        df_all = df_all.join(extra, how="left")
    df_all = df_all.fillna(0.0)

    if target_col not in df_all.columns:
        raise KeyError(f"target column '{target_col}' not found")

    print("[TFT] Defining feature groups...")
    categorical_cols = [c for c in df_all.columns if c.endswith("_code")]
    binary_cols      = [c for c in df_all.columns if c.endswith("_na") or c.endswith("_was_missing")]
    all_numeric      = df_all.select_dtypes(include="number").columns.tolist()
    continuous_cols  = [
        c for c in all_numeric
        if c not in categorical_cols and c not in binary_cols and c != target_col
    ]

    missing = [c for c in continuous_cols if c not in df_all.columns]
    if missing:
        msg = f"[prepare_tft_dataset] missing columns: {missing}"
        if strict:
            raise KeyError(msg)
        warnings.warn(msg)

    print("[TFT] Stratified train/test split...")
    y_all = df_all[target_col].values
    idx = np.arange(len(df_all))
    idx_train, idx_test = train_test_split(
        idx,
        train_size=train_size,
        stratify=y_all,
        random_state=random_state,
    )
    df_train = df_all.iloc[idx_train].copy().sort_index()
    df_test  = df_all.iloc[idx_test].copy().sort_index()

    print("[TFT] Class balance (train):", df_train[target_col].value_counts().to_dict())
    print("[TFT] Class balance (test) :", df_test[target_col].value_counts().to_dict())

    print("[TFT] Fitting scaler on train and transforming splits...")
    scaler = StandardScaler()
    df_train.loc[:, continuous_cols] = scaler.fit_transform(df_train[continuous_cols].astype("float32"))
    df_test.loc[:,  continuous_cols] = scaler.transform(df_test[continuous_cols].astype("float32"))
    joblib.dump(scaler, scaler_path)

    feature_cols = continuous_cols + categorical_cols + binary_cols

    print("[TFT] Building sliding windows for train data...")
    np.save(train_features_path, df_train[feature_cols].values.astype("float32"))
    np.save(train_targets_path,  df_train[target_col].values.astype("float32"))

    print("[TFT] Building sliding windows for test data...")
    np.save(test_features_path,  df_test[feature_cols].values.astype("float32"))
    np.save(test_targets_path,   df_test[target_col].values.astype("float32"))

    print("[TFT] Saving full dataset for inspection...")
    torch.save({
        "features": df_all[feature_cols].values.astype("float32"),
        "targets":  df_all[target_col].values.astype("float32"),
    }, dataset_path)

    groups = {
        "continuous": continuous_cols,
        "categorical": categorical_cols,
        "binary": binary_cols,
    }

    full_dataset = TFTDataset(
        df_all[feature_cols].values.astype("float32"),
        df_all[target_col].values.astype("float32"),
        seq_len=seq_len
    )
    print("[TFT] DATA PREP COMPLETED")
    return full_dataset, groups

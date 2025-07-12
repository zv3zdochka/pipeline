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


class TFTDataset(Dataset):
    """
    A PyTorch Dataset for TFT.
    Each item is a tuple (X_window, y_target):
      - X_window: Tensor of shape [seq_len, D]
      - y_target: Tensor scalar (int/float)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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
        train_dataset_path: str | pathlib.Path = "tft_train.pt",
        test_dataset_path: str | pathlib.Path = "tft_test.pt",
        strict: bool = False,
) -> tuple[TFTDataset, Mapping[str, Sequence[str]]]:
    """
    Assemble features for TFT, split into train/test, build sliding windows,
    fit scaler on train, transform both splits, and save all datasets & scaler.

    Returns
    -------
    full_dataset : TFTDataset
    feature_groups : dict[str, list[str]]
    """

    # 1. load events table
    if isinstance(df_events, (str, pathlib.Path)):
        df_events = (
            joblib.load(df_events)
            if str(df_events).endswith(".pkl")
            else pd.read_csv(df_events, parse_dates=["ts"])
        )
    df_events = df_events.set_index("ts").sort_index()

    # 2. load embeddings / forecasts
    def _read_parquet_any(path):
        p = pathlib.Path(path)
        return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, index_col=0, parse_dates=True)

    df_cnn = _read_parquet_any(cnn_emb_path).add_prefix("cnn_emb_")
    df_gru = _read_parquet_any(gru_emb_path).add_prefix("gru_emb_")
    df_tn_emb = _read_parquet_any(timesnet_emb_path).add_prefix("timesnet_emb_")
    df_tn_pred = _read_parquet_any(timesnet_pred_path)

    # 3. join all
    df_all = (
        df_events
        .join(df_cnn, how="inner")
        .join(df_gru, how="inner")
        .join(df_tn_emb, how="inner")
        .join(df_tn_pred, how="inner")
        .sort_index()
    )

    if target_col not in df_all.columns:
        raise KeyError(f"target column '{target_col}' not found in dataframe")

    # 4. define feature groups
    categorical_cols = [c for c in df_all.columns if c.endswith("_code")]
    binary_cols = [c for c in df_all.columns if c.endswith("_na") or c.endswith("_was_missing")]
    all_numeric = df_all.select_dtypes(include="number").columns.tolist()
    continuous_cols = [
        c for c in all_numeric
        if c not in categorical_cols and c not in binary_cols and c != target_col
    ]

    missing = [c for c in continuous_cols if c not in df_all.columns]
    if missing:
        msg = f"[prepare_tft_dataset] missing columns: {missing}"
        if strict:
            raise KeyError(msg)
        warnings.warn(msg)

    # 5. split into train/test by time
    n_total = len(df_all)
    split_idx = int(n_total * train_size)
    # train windows use only indices [seq_len, split_idx)
    # test windows may use past history (including train) for sliding window
    df_train = df_all.iloc[:split_idx]
    df_test = df_all.iloc[split_idx - seq_len:]  # include overlap for initial windows

    # 6. fit scaler on train, transform both
    scaler = StandardScaler()
    df_train[continuous_cols] = scaler.fit_transform(df_train[continuous_cols].astype("float32"))
    df_test[continuous_cols] = scaler.transform(df_test[continuous_cols].astype("float32"))

    # 7. build sliding windows for train
    feature_cols = continuous_cols + categorical_cols + binary_cols
    X_train_windows, y_train = [], []
    for i in range(seq_len, len(df_train)):
        window = df_train.iloc[i - seq_len: i][feature_cols].values.astype("float32")
        X_train_windows.append(window)
        y_train.append(df_train[target_col].iloc[i])
    X_train = np.stack(X_train_windows, axis=0)
    y_train = np.asarray(y_train)

    # 8. build sliding windows for test
    X_test_windows, y_test = [], []
    # note: i indexes into df_test; original df_all index = split_idx - seq_len + i
    for i in range(seq_len, len(df_test)):
        window = df_test.iloc[i - seq_len: i][feature_cols].values.astype("float32")
        X_test_windows.append(window)
        # target taken from df_all at global index
        global_i = split_idx - seq_len + i
        y_test.append(df_all[target_col].iloc[global_i])
    X_test = np.stack(X_test_windows, axis=0)
    y_test = np.asarray(y_test)

    # 9. save scaler and datasets
    joblib.dump(scaler, scaler_path)
    torch.save({"X": df_all[feature_cols].astype("float32").values.reshape(-1, len(feature_cols)),
                "y": df_all[target_col].values}, dataset_path)
    torch.save({"X": X_train, "y": y_train}, train_dataset_path)
    torch.save({"X": X_test, "y": y_test}, test_dataset_path)

    groups = {
        "continuous": continuous_cols,
        "categorical": categorical_cols,
        "binary": binary_cols,
    }
    return TFTDataset(df_all[feature_cols].values.reshape(-1, len(feature_cols)), df_all[target_col].values), groups

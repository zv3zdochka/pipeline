# prepare_dataset_TimesNet.py
import pathlib

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import joblib


class TimesNetDataset(Dataset):
    """
    PyTorch Dataset for TimesNet: returns (X, y) pairs where
    X is a sequence of shape [seq_len, D] and y is a scalar target.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X: [N, seq_len, D], y: [N]
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Return (seq_len, D) and scalar
        return self.X[idx], self.y[idx]


def prepare_timesnet_dataset(
        df: pd.DataFrame,
        seq_len: int = 288,
        horizon: int = 288,
        feature_cols: list = None,
        scaler_path: pathlib.Path = "timesnet_scaler.pkl",
        dataset_path: pathlib.Path = "timesnet_dataset.pt"
):
    """
    Prepare data for TimesNet:
    - Selects numeric feature columns
    - Computes target as pct_change of close over `horizon` steps
    - Fits StandardScaler on features
    - Applies sliding window of length `seq_len`
    - Saves scaler and dataset

    Returns:
        TimesNetDataset, fitted StandardScaler
    """
    # Default feature columns if none provided
    if feature_cols is None:
        feature_cols = [
            'ohlcv_5m_open', 'ohlcv_5m_high', 'ohlcv_5m_low', 'ohlcv_5m_close', 'ohlcv_5m_vol',
            'open_interest_kline_open', 'open_interest_kline_close',
            'open_interest_kline_low', 'open_interest_kline_high',
            'funding_rate_kline_open', 'funding_rate_kline_close',
            'funding_rate_kline_low', 'funding_rate_kline_high',
            'fund_flow_history_m5net',
            'funding_rate_weighted_openFundingRate',
            'funding_rate_weighted_turnoverFundingRate',
            'longshort_global_ratio', 'longshort_top_account_ratio', 'longshort_top_position_ratio'
        ]

    df_proc = df.copy()
    # Compute target: % change of close over horizon
    df_proc['timesnet_target'] = (
            df_proc['ohlcv_5m_close'].shift(-horizon) / df_proc['ohlcv_5m_close'] - 1.0
    )
    # Drop rows where any feature or target is NaN
    df_proc = df_proc.dropna(subset=feature_cols + ['timesnet_target'])

    # Extract feature matrix and target vector
    X_all = df_proc[feature_cols].values  # shape [T, D]
    y_all = df_proc['timesnet_target'].values  # shape [T]

    # Fit scaler and transform features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    # Generate sliding windows
    sequences = []
    targets = []
    for i in range(seq_len, len(X_scaled)):
        seq = X_scaled[i - seq_len:i]
        sequences.append(seq)
        targets.append(y_all[i])
    X_arr = np.stack(sequences, axis=0)  # [N, seq_len, D]
    y_arr = np.array(targets, dtype=np.float32)  # [N]

    # Create dataset
    dataset = TimesNetDataset(X_arr, y_arr)

    # Save scaler and dataset
    joblib.dump(scaler, scaler_path)
    torch.save({'X': X_arr, 'y': y_arr}, dataset_path)

    return dataset, scaler

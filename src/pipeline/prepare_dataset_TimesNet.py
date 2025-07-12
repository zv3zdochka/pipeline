from __future__ import annotations

import pathlib
import warnings
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

EPS = 0.001


class TimesNetDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        # хранит только 2D-массивы и лениво отдает окна
        self.X = X.astype("float32")
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        # число возможных окон
        return len(self.y) - self.seq_len

    def __getitem__(self, idx):
        # окно [idx : idx+seq_len], метка в позиции idx+seq_len
        i = idx + self.seq_len
        seq = torch.from_numpy(self.X[idx:i]).float()
        label = torch.tensor(int(self.y[i])).long()
        return seq, label


def prepare_timesnet_dataset(
        df: pd.DataFrame,
        *,
        seq_len: int = 288,
        horizon: int = 288,
        feature_cols: list | None = None,
        scaler_path: str | pathlib.Path = "timesnet_scaler.pkl",
        train_dataset_path: str | pathlib.Path | None = None,
        test_dataset_path: str | pathlib.Path | None = None,
        train_ratio: float = 0.8,
        strict: bool = False,
):

    print("[TIMESNET] DATA PREP STARTED")

    default_cols = [
        "ohlcv_5m_open", "ohlcv_5m_high", "ohlcv_5m_low",
        "ohlcv_5m_close", "ohlcv_5m_vol",
        "open_interest_kline_open", "open_interest_kline_close",
        "open_interest_kline_low", "open_interest_kline_high",
        "funding_rate_kline_open", "funding_rate_kline_close",
        "funding_rate_kline_low", "funding_rate_kline_high",
        "longshort_global_ratio", "longshort_top_account_ratio",
        "longshort_top_position_ratio",
        "fund_flow_history_m5net",
        "funding_rate_weighted_openFundingRate",
        "funding_rate_weighted_turnoverFundingRate",
    ]
    feature_cols = feature_cols or default_cols

    df_proc = df.sort_values("ts").reset_index(drop=True)

    df_proc["pct_future"] = (
        df_proc["ohlcv_5m_close"].shift(-horizon) /
        df_proc["ohlcv_5m_close"] - 1.0
    )
    df_proc["timesnet_target"] = np.select(
        [df_proc["pct_future"] > EPS, df_proc["pct_future"] < -EPS],
        [1, -1],
        default=0
    ).astype(np.int8)
    print("[TIMESNET] Assigned target classes")

    missing = [c for c in feature_cols if c not in df_proc.columns]
    if missing:
        msg = f"[TIMESNET] Missing columns: {missing}"
        if strict:
            raise KeyError(msg)
        warnings.warn(msg)
        feature_cols = [c for c in feature_cols if c in df_proc.columns]

    df_proc[feature_cols] = df_proc[feature_cols].ffill().bfill()
    df_proc = df_proc.dropna(subset=feature_cols + ["timesnet_target"])

    split_idx = int(len(df_proc) * train_ratio)
    train_df, test_df = df_proc.iloc[:split_idx], df_proc.iloc[split_idx:]
    print(f"[TIMESNET] Split into train ({len(train_df)}) / test ({len(test_df)})")

    scaler = StandardScaler().fit(train_df[feature_cols])
    print(f"[TIMESNET] Fitted scaler, saving to {scaler_path}")
    joblib.dump({"scaler": scaler, "features": feature_cols}, scaler_path)

    def to_numpy(dfr: pd.DataFrame):
        X = scaler.transform(dfr[feature_cols].astype("float32"))
        y = dfr["timesnet_target"].values
        return X, y

    X_train, y_train = to_numpy(train_df)
    X_test, y_test = to_numpy(test_df)

    # посчитаем число окон, чтобы вывести аналогичный лог
    n_tr = len(y_train) - seq_len
    n_te = len(y_test) - seq_len
    print(f"[TIMESNET] Created windows: train {n_tr} / test {n_te}")

    # создаем ленивые датасеты
    train_ds = TimesNetDataset(X_train, y_train, seq_len)
    test_ds  = TimesNetDataset(X_test,  y_test,  seq_len)

    # по желанию можно сохранять сами np-массивы для оффлайн-загрузки
    if train_dataset_path and test_dataset_path:
        torch.save({"X": X_train, "y": y_train}, train_dataset_path)
        torch.save({"X": X_test,  "y": y_test},  test_dataset_path)
        print(f"[TIMESNET] Saved train dataset to {train_dataset_path}")
        print(f"[TIMESNET] Saved test dataset to {test_dataset_path}")
    else:
        torch.save(
            {"X": np.concatenate([X_train, X_test]),
             "y": np.concatenate([y_train, y_test])},
            "timesnet_dataset.pt"
        )
        print("[TIMESNET] Saved combined dataset to timesnet_dataset.pt")

    print("[TIMESNET] DATA PREP COMPLETED")
    return train_ds, test_ds, scaler

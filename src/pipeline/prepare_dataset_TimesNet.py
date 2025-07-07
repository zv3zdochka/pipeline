# src/pipeline/prepare_dataset_TimesNet.py
import pathlib
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import joblib


class TimesNetDataset(Dataset):
    """Torch-Dataset -> (seq_len, D), scalar-target."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):           return len(self.y)
    def __getitem__(self, idx):  return self.X[idx], self.y[idx]


def prepare_timesnet_dataset(
        df: pd.DataFrame,
        *,
        seq_len: int = 288,
        horizon: int = 288,
        feature_cols: list | None = None,
        scaler_path: str | pathlib.Path = "timesnet_scaler.pkl",
        dataset_path: str | pathlib.Path = "timesnet_dataset.pt",
        strict: bool = False         # если True — падаем, когда колонки отсутствуют
):
    """
    • Берёт числовые фичи ➜ нормализует ➜ делает скользящие окна для TimesNet
    • Цель: %-изменение close через `horizon`
    • Автоматически сохраняет scaler и torch-tensor датасет
    """

    default_cols = [
        # --- рыночные OHLCV -----------
        "ohlcv_5m_open", "ohlcv_5m_high", "ohlcv_5m_low", "ohlcv_5m_close", "ohlcv_5m_vol",
        # --- открытый интерес ----------
        "open_interest_kline_open", "open_interest_kline_close",
        "open_interest_kline_low",  "open_interest_kline_high",
        # --- funding-rate --------------
        "funding_rate_kline_open", "funding_rate_kline_close",
        "funding_rate_kline_low",  "funding_rate_kline_high",
        # --- LS-ratio ------------------
        "longshort_global_ratio", "longshort_top_account_ratio", "longshort_top_position_ratio",
        # --- OPTIONAL ---
        "fund_flow_history_m5net",
        "funding_rate_weighted_openFundingRate",
        "funding_rate_weighted_turnoverFundingRate",
    ]

    feature_cols = feature_cols or default_cols
    df_proc = df.copy()

    # ---- таргет: % изменения close через horizon ----
    df_proc["timesnet_target"] = df_proc["ohlcv_5m_close"].shift(-horizon) / df_proc["ohlcv_5m_close"] - 1.0

    # ---- проверяем доступность колонок ----
    missing = [c for c in feature_cols if c not in df_proc.columns]
    if missing:
        msg = f"[TimesNet] отсутствуют колонки: {missing}"
        if strict:
            raise KeyError(msg)
        warnings.warn(msg + " — будут пропущены")
        feature_cols = [c for c in feature_cols if c in df_proc.columns]

    # ---- отбрасываем строки с NaN в выбранных фичах/таргете ----
    df_proc = df_proc.dropna(subset=feature_cols + ["timesnet_target"])

    # ---- извлекаем X, y ----
    X_all = df_proc[feature_cols].astype("float32").values
    y_all = df_proc["timesnet_target"].astype("float32").values

    # ---- масштабируем ----
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    # ---- окна длиной seq_len ----
    seqs, tgts = [], []
    for i in range(seq_len, len(X_scaled)):
        seqs.append(X_scaled[i - seq_len : i])
        tgts.append(y_all[i])
    X_arr = np.stack(seqs, axis=0)          # [N, seq_len, D]
    y_arr = np.array(tgts, dtype=np.float32)

    # ---- сохраняем ----
    joblib.dump(scaler, scaler_path)
    torch.save({"X": X_arr, "y": y_arr}, dataset_path)

    return TimesNetDataset(X_arr, y_arr), scaler

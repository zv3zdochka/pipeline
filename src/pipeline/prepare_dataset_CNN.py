from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import pywt
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def _fit_scaler_partial(scaler: StandardScaler, arr: np.ndarray) -> None:
    mask = ~np.isnan(arr).any(axis=1)
    scaler.fit(arr[mask])


def prepare_1dcnn_df(
    df: pd.DataFrame,
    *,
    wavelet: str = "db4",
    level: int = 3,
    window_size: int = 24,
    train_frac: float = 0.8,
    raw_scaler_path: Path = Path("scaler_raw.pkl"),
    wave_scaler_path: Path = Path("scaler_wave.pkl"),
    dataset_train_path: Path = Path("wavecnn_dataset_train.pkl"),
    dataset_test_path: Path = Path("wavecnn_dataset_test.pkl"),
    class_freq_path: Path = Path("class_freqs.pt"),
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler, StandardScaler]:

    if "ts" not in df.columns:
        raise ValueError("'ts' column required")

    df = df.copy().set_index("ts").sort_index()

    features = [
        "ohlcv_5m_open", "ohlcv_5m_high", "ohlcv_5m_low", "ohlcv_5m_close",
        "ohlcv_5m_vol",
        "open_interest_kline_open", "open_interest_kline_high",
        "open_interest_kline_low",  "open_interest_kline_close",
        "longshort_global_ratio", "longshort_top_account_ratio",
        "longshort_top_position_ratio",
    ]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")

    raw_arr = df[features].astype("float32").values
    scaler_raw = StandardScaler()
    split = int(len(df) * train_frac)
    scaler_raw.fit(raw_arr[:split])
    raw_scaled = scaler_raw.transform(raw_arr)
    df_raw = pd.DataFrame(raw_scaled, index=df.index, columns=features)

    wavelet_coeffs = {}
    wav = pywt.Wavelet(wavelet)
    shift_base = (wav.dec_len - 1) // 2
    for feat in features:
        series = df_raw[feat].values
        swt = pywt.swt(series, wavelet, level=level, start_level=0, norm=True)
        for lvl, (cA, cD) in enumerate(swt, start=1):
            shift = (2 ** (lvl - 1)) * shift_base
            if shift:
                pad = np.full(shift, np.nan, dtype="float32")
                cA = np.concatenate([pad, cA[:-shift]])
                cD = np.concatenate([pad, cD[:-shift]])
            wavelet_coeffs[f"{feat}_ca{lvl}"] = cA
            wavelet_coeffs[f"{feat}_cd{lvl}"] = cD

    df_wave = pd.DataFrame(wavelet_coeffs, index=df.index)

    scaler_wave = StandardScaler()
    _fit_scaler_partial(scaler_wave, df_wave.values[:split].astype("float32"))
    wave_scaled = scaler_wave.transform(np.nan_to_num(df_wave.values.astype("float32")))
    df_wave_scaled = pd.DataFrame(wave_scaled, index=df.index, columns=df_wave.columns)

    df_out = pd.concat([df_raw, df_wave_scaled], axis=1)
    if "microtrend_label" in df.columns:
        df_out["microtrend_label"] = df["microtrend_label"].astype("int8")

    train_df, test_df = df_out.iloc[:split], df_out.iloc[split:]

    joblib.dump(scaler_raw,  raw_scaler_path)
    joblib.dump(scaler_wave, wave_scaler_path)
    joblib.dump(train_df, dataset_train_path)
    joblib.dump(test_df,  dataset_test_path)

    if "microtrend_label" in train_df.columns:
        freqs = train_df["microtrend_label"].value_counts().to_dict()
        torch.save(freqs, class_freq_path)

    return train_df, test_df, scaler_raw, scaler_wave


class CNNWindowDataset(Dataset):
    def __init__(self, df: pd.DataFrame, window_size: int = 24):
        self.window = window_size
        self.X = df.drop(columns=["microtrend_label"], errors="ignore").values.astype("float32")
        self.has_label = "microtrend_label" in df.columns
        self.y = df["microtrend_label"].values.astype("int64") if self.has_label else None

    def __len__(self):
        return len(self.X) - self.window + 1

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx : idx + self.window].T)
        if self.has_label:
            y = int(self.y[idx + self.window - 1])
            return x, torch.tensor(y, dtype=torch.long)
        return x

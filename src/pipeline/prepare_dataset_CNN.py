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
    """
    Fit scaler on the non-NaN portion of arr.
    """
    mask = ~np.isnan(arr).any(axis=1)
    scaler.fit(arr[mask])


def prepare_1dcnn_df(
        df: pd.DataFrame,
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
    """
    Build causal WaveNet-CNN dataset with train/test split.

    Args:
        df: Input DataFrame containing 'ts' column.
        wavelet: Wavelet name for SWT.
        level: Decomposition level.
        window_size: Sliding window size.
        train_frac: Fraction for training split.
        raw_scaler_path: Path to save raw features scaler.
        wave_scaler_path: Path to save wavelet scaler.
        dataset_train_path: Path to save train dataset.
        dataset_test_path: Path to save test dataset.
        class_freq_path: Path to save class frequency info.

    Returns:
        df_train_out: Processed train DataFrame.
        df_test_out: Processed test DataFrame.
        scaler_raw: Fitted scaler for raw features.
        scaler_wave: Fitted scaler for wavelet features.
    """
    if "ts" not in df.columns:
        raise ValueError("DataFrame must contain 'ts' column")

    df = df.copy().set_index("ts").sort_index()

    features = [
        "ohlcv_5m_open",
        "ohlcv_5m_high",
        "ohlcv_5m_low",
        "ohlcv_5m_close",
        "ohlcv_5m_vol",
        "open_interest_kline_open",
        "open_interest_kline_high",
        "open_interest_kline_low",
        "open_interest_kline_close",
        "longshort_global_ratio",
        "longshort_top_account_ratio",
        "longshort_top_position_ratio",
    ]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    raw_arr = df[features].astype("float32").values
    scaler_raw = StandardScaler()
    train_len = int(len(df) * train_frac)
    scaler_raw.fit(raw_arr[:train_len])
    raw_scaled = scaler_raw.transform(raw_arr)
    df_raw = pd.DataFrame(raw_scaled, index=df.index, columns=features)

    wavelet_coeffs = {}
    wav = pywt.Wavelet(wavelet)
    filt_shift = (wav.dec_len - 1) // 2
    for feat in features:
        series = df_raw[feat].values
        swt = pywt.swt(series, wavelet, level=level, start_level=0, norm=True)
        for lvl, (cA, cD) in enumerate(swt, start=1):
            shift = (2 ** (lvl - 1)) * filt_shift
            if shift > 0:
                pad = np.full(shift, np.nan, dtype="float32")
                cA = np.concatenate([pad, cA[:-shift]]).astype("float32")
                cD = np.concatenate([pad, cD[:-shift]]).astype("float32")
            wavelet_coeffs[f"{feat}_ca{lvl}"] = cA
            wavelet_coeffs[f"{feat}_cd{lvl}"] = cD

    df_wave = pd.DataFrame(wavelet_coeffs, index=df.index)

    scaler_wave = StandardScaler()
    _fit_scaler_partial(scaler_wave, df_wave.values[:train_len].astype("float32"))
    wave_scaled = scaler_wave.transform(np.nan_to_num(df_wave.values.astype("float32")))
    df_wave_scaled = pd.DataFrame(wave_scaled, index=df.index, columns=df_wave.columns)

    df_out = pd.concat([df_raw, df_wave_scaled], axis=1)

    if "microtrend_label" in df.columns:
        df_out["microtrend_label"] = df["microtrend_label"].astype("int8")

    df_train_out = df_out.iloc[:train_len]
    df_test_out = df_out.iloc[train_len:]

    joblib.dump(scaler_raw, raw_scaler_path)
    joblib.dump(scaler_wave, wave_scaler_path)
    joblib.dump(df_train_out, dataset_train_path)
    joblib.dump(df_test_out, dataset_test_path)

    if "microtrend_label" in df_train_out.columns:
        class_freqs = df_train_out["microtrend_label"].value_counts().to_dict()
        torch.save(class_freqs, class_freq_path)

    return df_train_out, df_test_out, scaler_raw, scaler_wave


class CNNWindowDataset(Dataset):
    def __init__(self, df: pd.DataFrame, window_size: int = 24):
        self.window = window_size
        self.features = df.drop(columns=["microtrend_label"], errors="ignore")
        self.X = self.features.values.astype("float32")
        self.has_label = "microtrend_label" in df.columns
        if self.has_label:
            self.y = df["microtrend_label"].values.astype("int64")

    def __len__(self) -> int:
        return len(self.X) - self.window + 1

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx: idx + self.window].T)
        if self.has_label:
            y = int(self.y[idx + self.window - 1])
            return x, torch.tensor(y, dtype=torch.long)
        return x

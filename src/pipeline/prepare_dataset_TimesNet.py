# pipeline/prepare_dataset_TimesNet.py
from __future__ import annotations

import pathlib
import warnings
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class TimesNetDataset(Dataset):
    """
    "Ленивая" обёртка вокруг ndarray-ов.
    Возвращает пару (окно длиной seq_len, метка на шаге idx+seq_len).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int) -> None:
        assert X.ndim == 2 and len(X) == len(y)
        self.X = X.astype("float32")
        self.y = y.astype("int8")
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.y) - self.seq_len

    def __getitem__(self, idx: int):
        j = idx + self.seq_len
        return (
            torch.from_numpy(self.X[idx:j]).float(),   # (seq_len, F)
            torch.tensor(int(self.y[j])).long()        # scalar
        )


def prepare_timesnet_dataset(
    df: pd.DataFrame,
    *,
    seq_len: int = 288,
    horizon: int = 288,
    feature_cols: list[str] | None = None,
    # --- целевая метка -------------------------------------------------
    target_col: str = "microtrend_label",
    shift_target: bool = True,
    # -------------------------------------------------------------------
    train_ratio: float = 0.8,
    scaler_path: str | pathlib.Path = "timesnet_scaler.pkl",
    train_dataset_path: str | pathlib.Path | None = None,
    test_dataset_path: str | pathlib.Path | None = None,
    strict: bool = False,
):
    """
    Готовит train/test наборы для TimesNet и при желании сохраняет их
    (а также скейлер) на диск.

    Parameters
    ----------
    df : исходный DataFrame (уже с колонкой `target_col`)
    seq_len : длина входного окна для TimesNet
    horizon : на сколько шагов вперёд сдвигаем метку (если shift_target=True)
    target_col : имя исходной метки (по умолчанию microtrend_label)
    shift_target : True ⇒ target(t) = label(t + horizon)
    train_ratio : доля строк в train-части
    """

    default_cols = [
        # --- рыночные (пример — можете расширить) ----------------------
        "ohlcv_5m_open", "ohlcv_5m_high", "ohlcv_5m_low",
        "ohlcv_5m_close", "ohlcv_5m_vol",
        "open_interest_kline_open", "open_interest_kline_close",
        "open_interest_kline_low", "open_interest_kline_high",
        "funding_rate_kline_open", "funding_rate_kline_close",
        "funding_rate_kline_low", "funding_rate_kline_high",
        "longshort_global_ratio", "longshort_top_account_ratio",
        "longshort_top_position_ratio",
        # --- фундаментал / ончейн --------------------------------------
        "fund_flow_history_m5net",
        "funding_rate_weighted_openFundingRate",
        "funding_rate_weighted_turnoverFundingRate",
    ]
    feature_cols = feature_cols or default_cols

    # ------------------------------------------------------------------
    # 1) сортируем и формируем целевой столбец
    # ------------------------------------------------------------------
    df = df.sort_values("ts").reset_index(drop=True)

    if target_col not in df.columns:
        raise KeyError(f"{target_col} not found in dataframe")

    if shift_target:
        df["timesnet_target"] = df[target_col].shift(-horizon)
    else:
        df["timesnet_target"] = df[target_col]

    # ------------------------------------------------------------------
    # 2) проверяем/убираем отсутствующие фичи
    # ------------------------------------------------------------------
    miss = [c for c in feature_cols if c not in df.columns]
    if miss:
        msg = f"[TIMESNET] Missing columns: {miss}"
        if strict:
            raise KeyError(msg)
        warnings.warn(msg)
        feature_cols = [c for c in feature_cols if c in df.columns]

    # ------------------------------------------------------------------
    # 3) заполняем пропуски + удаляем строки без метки
    # ------------------------------------------------------------------
    df[feature_cols] = df[feature_cols].ffill().bfill()
    df = df.dropna(subset=feature_cols + ["timesnet_target"])

    # ------------------------------------------------------------------
    # 4) train / test сплит
    # ------------------------------------------------------------------
    split = int(len(df) * train_ratio)
    train_df, test_df = df.iloc[:split], df.iloc[split:]
    print(f"[TIMESNET] Split: train {len(train_df)}  /  test {len(test_df)}")

    # ------------------------------------------------------------------
    # 5) скейлер только по train
    # ------------------------------------------------------------------
    scaler = StandardScaler().fit(train_df[feature_cols].astype("float32"))
    joblib.dump({"scaler": scaler, "features": feature_cols}, scaler_path)
    print(f"[TIMESNET] Scaler saved → {scaler_path}")

    def _to_np(dfr: pd.DataFrame):
        x = scaler.transform(dfr[feature_cols].astype("float32"))
        y = dfr["timesnet_target"].values         # -1 / 0 / 1
        return x, y

    X_train, y_train = _to_np(train_df)
    X_test,  y_test  = _to_np(test_df)

    print(f"[TIMESNET] Windows: train {len(y_train)-seq_len} / test {len(y_test)-seq_len}")

    # ------------------------------------------------------------------
    # 6) ленивые датасеты
    # ------------------------------------------------------------------
    train_ds = TimesNetDataset(X_train, y_train, seq_len)
    test_ds  = TimesNetDataset(X_test,  y_test,  seq_len)

    # ------------------------------------------------------------------
    # 7) сохраняем numpy-пары (экономит RAM при повторном запуске)
    # ------------------------------------------------------------------
    if train_dataset_path:
        torch.save({"X": X_train, "y": y_train}, train_dataset_path)
        print(f"[TIMESNET] train.pt  → {train_dataset_path}")
    if test_dataset_path:
        torch.save({"X": X_test,  "y": y_test},  test_dataset_path)
        print(f"[TIMESNET] test.pt   → {test_dataset_path}")

    return train_ds, test_ds, scaler

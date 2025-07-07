# src/pipeline/prepare_dataset_TFT.py
import pathlib
import warnings
from typing import Sequence, Mapping

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


# ----------------------------- #
# 1.  Dataset-класс
# ----------------------------- #
class TFTDataset(Dataset):
    """
    (X, y) где
      X.shape == [seq_len, D]  — окно признаков
      y.shape == []            — скаляр-таргет (int/float)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ----------------------------- #
# 2.  Основная обёртка
# ----------------------------- #
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
        strict: bool = False,
) -> tuple[TFTDataset, Mapping[str, Sequence[str]]]:
    """
    Собирает полный набор признаков для TFT, строит окна и возвращает `TFTDataset`.

    Parameters
    ----------
    df_events : DataFrame | path
        Результат `prepare_tft_features()` + метки (`label_microtrend`),
        либо путь к pickle/CSV.
    cnn_emb_path / gru_emb_path / timesnet_emb_path / timesnet_pred_path :
        Пути к parquet/CSV с эмбеддингами и прогнозами.
    seq_len : int
        Длина временного окна.
    target_col : str
        Имя колонки-таргета.
    strict : bool
        Если True — бросать ошибку при отсутствии нужных колонок,
        иначе — лишь предупреждать.

    Returns
    -------
    ds : TFTDataset
    feature_groups : dict
        {'continuous': [...], 'categorical': [...], 'binary': [...]}
    """

    # ---------- 2.1  загружаем базу событий ----------
    if isinstance(df_events, (str, pathlib.Path)):
        df_events = (
            joblib.load(df_events)
            if str(df_events).endswith(".pkl")
            else pd.read_csv(df_events, parse_dates=["ts"])
        )
    df_events = df_events.set_index("ts").sort_index()

    # ---------- 2.2  подгружаем эмбеддинги / прогнозы ----------
    def _read_parquet_any(path):
        p = pathlib.Path(path)
        if p.suffix == ".parquet":
            return pd.read_parquet(p)
        return pd.read_csv(p, index_col=0, parse_dates=True)

    df_cnn = _read_parquet_any(cnn_emb_path)
    df_gru = _read_parquet_any(gru_emb_path)
    df_tn_emb = _read_parquet_any(timesnet_emb_path)
    df_tn_pred = _read_parquet_any(timesnet_pred_path)

    # именуем столбцы уникально
    df_cnn = df_cnn.add_prefix("cnn_emb_")
    df_gru = df_gru.add_prefix("gru_emb_")
    df_tn_emb = df_tn_emb.add_prefix("timesnet_emb_")

    # ---------- 2.3  объединяем ----------
    df_all = (
        df_events.join(df_cnn, how="inner")
        .join(df_gru, how="inner")
        .join(df_tn_emb, how="inner")
        .join(df_tn_pred, how="inner")
        .sort_index()
    )

    # ---------- 2.4  проверяем наличие колонки-таргета ----------
    if target_col not in df_all.columns:
        raise KeyError(f"target column '{target_col}' not found in dataframe")

    # ----------------------------------------------------------------------------
    # 3.  Определяем группы признаков
    # ----------------------------------------------------------------------------
    # categorical
    categorical_cols = [
        c for c in df_all.columns if c.endswith("_code")
    ]
    # binary flags
    binary_cols = [
        c
        for c in df_all.columns
        if c.endswith("_na") or c.endswith("_was_missing")
    ]
    # все остальное, кроме таргета
    all_numeric = df_all.select_dtypes(include="number").columns.tolist()
    continuous_cols = [
        c
        for c in all_numeric
        if c not in categorical_cols
           and c not in binary_cols
           and c != target_col
    ]

    # ---------- 3.1  проверяем пропущенные обязательные continuous ----------
    must_exist = continuous_cols  # их очень много — проверим позже, если захотим
    missing = [c for c in must_exist if c not in df_all.columns]
    if missing:
        msg = f"[prepare_tft_dataset] missing columns: {missing}"
        if strict:
            raise KeyError(msg)
        warnings.warn(msg)

    # ----------------------------------------------------------------------------
    # 4.  Масштабируем continuous-часть
    # ----------------------------------------------------------------------------
    scaler = StandardScaler()
    df_all[continuous_cols] = scaler.fit_transform(df_all[continuous_cols].astype("float32"))

    # (категориальные и бинарные оставляем без изменений)

    # ----------------------------------------------------------------------------
    # 5.  формируем X, y, делаем окна
    # ----------------------------------------------------------------------------
    feature_cols = continuous_cols + categorical_cols + binary_cols
    X_all = df_all[feature_cols].astype("float32").values
    y_all = df_all[target_col].values  # dtype сохраняем как есть (int8 / float)

    sequences, targets = [], []
    for i in range(seq_len, len(df_all)):
        sequences.append(X_all[i - seq_len: i])
        targets.append(y_all[i])

    X_arr = np.stack(sequences, axis=0)  # shape [N, seq_len, D]
    y_arr = np.asarray(targets)

    # ----------------------------------------------------------------------------
    # 6.  сохраняем
    # ----------------------------------------------------------------------------
    joblib.dump(scaler, scaler_path)
    torch.save({"X": X_arr, "y": y_arr}, dataset_path)

    groups = {
        "continuous": continuous_cols,
        "categorical": categorical_cols,
        "binary": binary_cols,
    }
    return TFTDataset(X_arr, y_arr), groups

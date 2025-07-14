from __future__ import annotations

import pathlib
import warnings
from collections import Counter
from typing import Tuple, List

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
#                              DATASET-ОБЁРТКА                                #
# --------------------------------------------------------------------------- #
class TimesNetDataset(Dataset):
    """Ленивая обёртка: возвращает (окно, метка)."""

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        assert X.ndim == 2 and len(X) == len(y)
        self.X = X.astype("float32")
        self.y = y.astype("int8")
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.y) - self.seq_len

    def __getitem__(self, idx: int):
        j = idx + self.seq_len
        return (
            torch.from_numpy(self.X[idx:j]).float(),      # (seq_len, F)
            torch.tensor(int(self.y[j])).long(),          # scalar
        )


# --------------------------------------------------------------------------- #
#                    ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ                                  #
# --------------------------------------------------------------------------- #
def _balance_by_oversampling(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Делает классы ровными за счёт дублирования строк."""
    counts = df[label_col].value_counts()
    max_cnt = counts.max()

    parts = []
    for lbl, cnt in counts.items():
        repeat = int(np.ceil(max_cnt / cnt))
        part = (
            pd.concat([df[df[label_col] == lbl]] * repeat, ignore_index=True)
            .iloc[:max_cnt]
        )
        parts.append(part)

    return (
        pd.concat(parts, ignore_index=True)
        .sort_values("ts")
        .reset_index(drop=True)
    )


def _print_dist(name: str, y: np.ndarray):
    dist = Counter(y)
    tot = sum(dist.values())
    print(
        f"[TIMESNET] {name} class distribution → "
        f"{ {k: f'{v} ({v/tot:.1%})' for k, v in dist.items()} }"
    )


# --------------------------------------------------------------------------- #
#                              ГЛАВНАЯ ФУНКЦИЯ                                #
# --------------------------------------------------------------------------- #
def prepare_timesnet_dataset(
    df: pd.DataFrame,
    *,
    seq_len: int = 288,
    feature_cols: List[str] | None = None,
    # --- целевая метка -------------------------------------------------
    target_col: str = "microtrend_label",
    shift_target: bool = False,
    horizon: int = 0,
    # ------------------------------------------------------------------
    train_ratio: float = 0.8,
    scaler_path: str | pathlib.Path = "timesnet_scaler.pkl",
    train_dataset_path: str | pathlib.Path | None = None,
    test_dataset_path: str | pathlib.Path | None = None,
    strict: bool = False,
) -> Tuple[TimesNetDataset, TimesNetDataset, StandardScaler]:
    """
    Полная подготовка данных для TimesNet c ровным балансом классов.
    """

    # 1. Хронологический порядок
    df = df.sort_values("ts").reset_index(drop=True)

    if target_col not in df.columns:
        raise KeyError(f"{target_col!r} not found")

    df["timesnet_target"] = (
        df[target_col].shift(-horizon) if shift_target else df[target_col]
    )

    # 2. Признаки
    if feature_cols is None:
        numeric = df.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric if c != target_col]

    miss = [c for c in feature_cols if c not in df.columns]
    if miss:
        msg = f"[TIMESNET] Missing columns: {miss}"
        if strict:
            raise KeyError(msg)
        warnings.warn(msg)
        feature_cols = [c for c in feature_cols if c in df.columns]

    if not feature_cols:
        raise ValueError("[TIMESNET] No usable feature columns left")

    # 3. Пропуски
    df[feature_cols] = df[feature_cols].ffill().bfill()
    df = df.dropna(subset=feature_cols + ["timesnet_target"])

    # 4. **Стратифицированный** сплит
    train_parts, test_parts = [], []
    for lbl, grp in df.groupby("timesnet_target"):
        grp = grp.sort_values("ts")
        split = max(1, int(len(grp) * train_ratio))  # мин. 1 строка в test
        train_parts.append(grp.iloc[:split])
        test_parts.append(grp.iloc[split:])

    train_df = pd.concat(train_parts).sort_values("ts").reset_index(drop=True)
    test_df = pd.concat(test_parts).sort_values("ts").reset_index(drop=True)

    print(f"[TIMESNET] Split: train {len(train_df)}  /  test {len(test_df)}")

    # 5. Балансировка
    train_df = _balance_by_oversampling(train_df, "timesnet_target")
    test_df = _balance_by_oversampling(test_df, "timesnet_target")

    # 6. Скейлер
    scaler = StandardScaler().fit(train_df[feature_cols].astype("float32"))
    joblib.dump({"scaler": scaler, "features": feature_cols}, scaler_path)
    print(f"[TIMESNET] Scaler saved → {scaler_path}")

    def _to_np(dfr):
        x = scaler.transform(dfr[feature_cols].astype("float32"))
        y = dfr["timesnet_target"].values
        return x, y

    X_train, y_train = _to_np(train_df)
    X_test, y_test = _to_np(test_df)

    _print_dist("Train", y_train)
    _print_dist("Test ", y_test)
    print(
        f"[TIMESNET] Windows: train {len(y_train)-seq_len} / "
        f"test {len(y_test)-seq_len}"
    )

    # 7. PyTorch Datasets
    train_ds = TimesNetDataset(X_train, y_train, seq_len)
    test_ds = TimesNetDataset(X_test, y_test, seq_len)

    # 8. (опц.) сохраняем
    if train_dataset_path:
        torch.save({"X": X_train, "y": y_train}, train_dataset_path)
        print(f"[TIMESNET] train.pt  → {train_dataset_path}")

    if test_dataset_path:
        torch.save({"X": X_test, "y": y_test}, test_dataset_path)
        print(f"[TIMESNET] test.pt   → {test_dataset_path}")

    return train_ds, test_ds, scaler

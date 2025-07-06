"""data_timesnet.py
===================
Модуль полностью закрывает подготовку **и** обучение TimesNet‑блока, чтобы его можно было безболезненно вписать в существующий конвейер (`main.py`).

Содержит три ключевых части:
1. **TimesNetDataset**  – отдаёт кортежи `(seq_len×D,  target)`
2. **prepare_timesnet_dataset(...)** – формирует dataset + scaler и сохраняет их.
3. **train_timesnet(...)** – скрипт обучения/валидации, сохраняет Torch‑чекпойнт.

Дефолтные параметры ориентированы на 5‑мин данные с горизонтом 24 ч (288 тиков).
"""

from __future__ import annotations

import pathlib
import joblib
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# 1. Dataset
# ----------------------------------------------------------------------------

class TimesNetDataset(Dataset):
    """Держит X‑последовательности и y‑таргет."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()  # (N, T, D)
        self.y = torch.from_numpy(y).float()  # (N,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

# ----------------------------------------------------------------------------
# 2. Подготовка данных
# ----------------------------------------------------------------------------

def _default_feature_cols() -> List[str]:
    return [
        "ohlcv_5m_open", "ohlcv_5m_high", "ohlcv_5m_low", "ohlcv_5m_close", "ohlcv_5m_vol",
        "open_interest_kline_open", "open_interest_kline_close",
        "open_interest_kline_low", "open_interest_kline_high",
        "funding_rate_kline_open", "funding_rate_kline_close",
        "funding_rate_kline_low", "funding_rate_kline_high",
        "fund_flow_history_m5net",
        "funding_rate_weighted_openFundingRate",
        "funding_rate_weighted_turnoverFundingRate",
        "longshort_global_ratio", "longshort_top_account_ratio", "longshort_top_position_ratio",
    ]

def prepare_timesnet_dataset(
    df: pd.DataFrame,
    seq_len: int = 288,
    horizon: int = 288,
    feature_cols: List[str] | None = None,
    scaler_path: str | pathlib.Path = "timesnet_scaler.pkl",
    dataset_path: str | pathlib.Path = "timesnet_dataset.pt",
) -> Tuple[TimesNetDataset, StandardScaler]:
    """Готовит данные под TimesNet.

    * **target** = *(close[t+horizon] / close[t] − 1)* – проц. изменение за сутки.
    * Чистит NaN, стандартизует только выбранные фичи.
    * Скользящее окно длиной *seq_len*.
    """

    feature_cols = feature_cols or _default_feature_cols()

    work_df = df.copy()
    work_df["timesnet_target"] = (
        work_df["ohlcv_5m_close"].shift(-horizon) / work_df["ohlcv_5m_close"] - 1.0
    )

    work_df = work_df.dropna(subset=feature_cols + ["timesnet_target"])

    X_raw = work_df[feature_cols].to_numpy(dtype="float32")
    y_raw = work_df["timesnet_target"].to_numpy(dtype="float32")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    sequences, targets = [], []
    for t in range(seq_len, len(X_scaled)):
        sequences.append(X_scaled[t - seq_len : t])
        targets.append(y_raw[t])

    X_arr = np.stack(sequences)  # (N, seq_len, D)
    y_arr = np.asarray(targets, dtype="float32")

    dataset = TimesNetDataset(X_arr, y_arr)

    joblib.dump(scaler, str(scaler_path))
    torch.save({"X": X_arr, "y": y_arr}, str(dataset_path))

    return dataset, scaler

# ----------------------------------------------------------------------------
# 3. Лёгкая обёртка TimesNet
# ----------------------------------------------------------------------------

class _FallbackTCN(nn.Module):
    """На случай, если настоящая TimesNet не установлена. Стоит заменить при наличии libs."""

    def __init__(self, in_dim: int, h_dim: int = 64, levels: int = 4):
        super().__init__()
        layers = []
        dil = 1
        for _ in range(levels):
            layers += [nn.Conv1d(in_dim, h_dim, kernel_size=3, dilation=dil, padding=dil),
                       nn.ReLU()]
            in_dim = h_dim
            dil *= 2
        self.net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor):  # x: (B, T, D)
        x = x.transpose(1, 2)            # (B, D, T)
        h = self.pool(self.net(x)).squeeze(-1)  # (B, h_dim)
        return h

try:
    from timesnet.models.torch_model import Model as _TimesNetOfficial  # type: ignore
except ModuleNotFoundError:
    _TimesNetOfficial = None  # noqa: N818

class TimesNetWrapper(nn.Module):
    """Unified interface: returns (pred, latent)."""

    def __init__(self, in_dim: int, horizon: int, d_model: int = 64):
        super().__init__()
        if _TimesNetOfficial is not None:
            self.backbone = _TimesNetOfficial(in_dim, d_model=d_model, out_len=horizon)
            self.latent_dim = d_model
        else:
            self.backbone = _FallbackTCN(in_dim, h_dim=d_model)
            self.latent_dim = d_model
            self.reg_head = nn.Linear(d_model, 1)
        self.horizon = horizon

    def forward(self, x: torch.Tensor):  # x: (B, T, D)
        if _TimesNetOfficial is not None:
            # official returns (B, horizon, D_out) – берём D_out=1
            out = self.backbone(x)  # shape (B, horizon, 1)
            pred = out[:, -1, 0]    # последний горизонт (24ч) – (B,)
            latent = out.mean(dim=1)  # (B, d_model)
        else:
            latent = self.backbone(x)
            pred = self.reg_head(latent).squeeze(-1)
        return pred, latent

# ----------------------------------------------------------------------------
# 4. Training helper
# ----------------------------------------------------------------------------

def train_timesnet(
    dataset_path: str | pathlib.Path = "timesnet_dataset.pt",
    model_out: str | pathlib.Path = "timesnet_model.pt",
    scaler_path: str | pathlib.Path = "timesnet_scaler.pkl",
    seq_len: int = 288,
    horizon: int = 288,
    batch: int = 128,
    epochs: int = 15,
    lr: float = 1e-3,
    val_ratio: float = 0.2,
    d_model: int = 64,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Полный цикл обучения/валидации."""

    payload = torch.load(dataset_path, weights_only=False)
    dataset = TimesNetDataset(payload["X"], payload["y"])

    n_total = len(dataset)
    n_val = int(val_ratio * n_total)
    train_ds, val_ds = random_split(dataset, [n_total - n_val, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False)

    model = TimesNetWrapper(in_dim=payload["X"].shape[-1], horizon=horizon, d_model=d_model).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    patience, patience_left = 5, 5

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        tr_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred, _ = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * len(yb)
        tr_loss /= len(train_ds)

        # ---- val ----
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                pred, _ = model(Xb)
                val_loss += criterion(pred, yb).item() * len(yb)
            val_loss /= len(val_ds)

        print(f"[TimesNet] Epoch {epoch:02d}: train {tr_loss:.5f} | val {val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            patience_left = patience
            torch.save({
                "model_state_dict": model.state_dict(),
                "scaler_path": str(scaler_path),
                "seq_len": seq_len,
                "horizon": horizon,
                "d_model": d_model,
                "feature_dim": payload["X"].shape[-1],
            }, str(model_out))
        else:
            patience_left -= 1
            if patience_left == 0:
                print("Early stopping triggered.")
                break

    print(f"[TimesNet] Best val MSE: {best_val:.5f}. Model saved to {model_out}")

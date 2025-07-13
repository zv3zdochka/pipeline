# pipeline/train_timesnet.py
from __future__ import annotations

import pathlib
import warnings
import joblib
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F

from .prepare_dataset_TimesNet import TimesNetDataset
from .models.TimesNet import TimesNetModel


# ---------- loss ---------------------------------------------------------------
def focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    gamma: float = 2.0,
    alpha: list[float] | None = None,
) -> torch.Tensor:
    """Multiclass focal-loss с весами классов `alpha=[w_-1, w_0, w_+1]`."""
    ce = F.cross_entropy(logits, target, reduction="none")
    pt = torch.exp(-ce)
    a = 1.0 if alpha is None else torch.tensor(alpha, device=logits.device)[target]
    return (a * (1.0 - pt).pow(gamma) * ce).mean()


# ---------- helpers ------------------------------------------------------------
def _loader(ds, batch, shuffle):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle,
                      num_workers=4, pin_memory=True)


def _class_weights(y: np.ndarray) -> list[float]:
    """inverse-frequency weights w_k / mean(w)  for k∈{-1,0,1}"""
    cnt = np.bincount(y + 1, minlength=3)
    w   = cnt.max() / np.clip(cnt, 1, None)
    return w.tolist()


# ---------- main trainer -------------------------------------------------------
def train_timesnet(
    train_pt: str | pathlib.Path,
    test_pt: str | pathlib.Path | None = None,
    *,
    events_pkl: str | pathlib.Path | None = None,
    model_out: str | pathlib.Path = "timesnet_model.pt",
    embed_out: str | pathlib.Path = "timesnet_embeddings.parquet",
    forecast_out: str | pathlib.Path = "timesnet_forecast.parquet",
    seq_len: int = 288,
    epochs: int = 25,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 6,
    device: str | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------ #
    # 1. загрузка подготовленных массивов (.pt)
    # ------------------------------------------------------------------ #
    tr_raw = torch.load(train_pt, weights_only=False)
    X_tr, y_tr = tr_raw["X"], tr_raw["y"].astype(np.int64)

    if test_pt and pathlib.Path(test_pt).exists():
        te_raw = torch.load(test_pt, weights_only=False)
        X_val, y_val = te_raw["X"], te_raw["y"].astype(np.int64)
    else:  # fallback — сплит 90/10 без стратификации
        split = int(0.9 * len(X_tr))
        X_tr, X_val = X_tr[:split], X_tr[split:]
        y_tr, y_val = y_tr[:split], y_tr[split:]

    if np.unique(y_tr).size < 2:
        raise RuntimeError("train set contains <2 classes — проверьте prepare_timesnet_dataset()")

    # datasets / loaders
    tr_ds = TimesNetDataset(X_tr, y_tr, seq_len)
    va_ds = TimesNetDataset(X_val, y_val, seq_len)
    tr_ld = _loader(tr_ds, batch_size, shuffle=True)
    va_ld = _loader(va_ds, batch_size, shuffle=False)

    # ------------------------------------------------------------------ #
    # 2. модель, оптимизатор, scheduler
    # ------------------------------------------------------------------ #
    alpha = _class_weights(y_tr)
    print(f"[TIMESNET] focal α → {alpha}")

    model = TimesNetModel(
        seq_len=seq_len,
        n_features=X_tr.shape[-1],
        d_model=128,
        n_blocks=4,
        num_classes=3,
    ).to(device)

    optim     = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="max", patience=3, factor=0.5, verbose=True
    )

    best_f1, wait = 0.0, 0

    # ------------------------------------------------------------------ #
    # 3. цикл обучения
    # ------------------------------------------------------------------ #
    for ep in range(1, epochs + 1):
        # ---- train ----------------------------------------------------
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), (yb + 1).to(device)  # shift [-1,0,1]→[0,1,2]
            optim.zero_grad()
            logits, _ = model(xb)
            loss = focal_loss(logits, yb, gamma=2.0, alpha=alpha)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            tr_loss += loss.item() * yb.size(0)
        tr_loss /= len(tr_ds)

        # ---- val ------------------------------------------------------
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in va_ld:
                xb, yb = xb.to(device), (yb + 1).to(device)
                logit, _ = model(xb)
                preds.append(logit.argmax(1).cpu().numpy())
                trues.append(yb.cpu().numpy())
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)

        val_f1 = f1_score(trues, preds,
                          labels=[0, 1, 2], average="macro", zero_division=0)

        # ---- log ------------------------------------------------------
        if ep == 1:
            binc = np.bincount(trues, minlength=3)
            print(f"[TIMESNET] val class dist: {{-1:{binc[0]}, 0:{binc[1]}, 1:{binc[2]}}}")

        print(f"[TIMESNET] ep {ep:02d}/{epochs}  "
              f"train_loss={tr_loss:.4f}  val_f1={val_f1:.3f}")

        scheduler.step(val_f1)

        if val_f1 > best_f1:
            best_f1, wait = val_f1, 0
            torch.save(model.state_dict(), model_out)
            print(f"  ↳ new best F1={best_f1:.3f}  (saved → {model_out})")
        else:
            wait += 1
            print(f"  ↳ no improve, patience {wait}/{patience}")
            if wait >= patience:
                print("[TIMESNET] Early stopping")
                break

    # ------------------------------------------------------------------ #
    # 4. эмбеддинги + прогнозы за весь период
    # ------------------------------------------------------------------ #
    print(f"[TIMESNET] Training done, best F1={best_f1:.3f}")
    model.load_state_dict(torch.load(model_out, map_location=device))
    model.eval()

    full_X = np.concatenate([X_tr, X_val], axis=0)
    full_y = np.concatenate([y_tr, y_val], axis=0)
    full_ds = TimesNetDataset(full_X, full_y, seq_len)
    full_ld = _loader(full_ds, batch_size, shuffle=False)

    emb_list, log_list = [], []
    with torch.no_grad():
        for xb, _ in full_ld:
            xb = xb.to(device)
            logit, emb = model(xb)
            emb_list.append(emb.cpu())
            log_list.append(logit.cpu())

    embeds = torch.cat(emb_list).numpy()
    probs  = torch.softmax(torch.cat(log_list), dim=1).numpy()[:, 2]  # P(class=+1)

    # привязываем к тайм-индексу
    if events_pkl and pathlib.Path(events_pkl).exists():
        ev = joblib.load(events_pkl).set_index("ts").sort_index()
        idx = ev.index[seq_len : seq_len + len(probs)]
    else:
        idx = pd.RangeIndex(len(probs))

    pd.DataFrame(embeds, index=idx).to_parquet(embed_out)
    pd.DataFrame({"timesnet_pred": probs}, index=idx).to_parquet(forecast_out)
    print(f"[TIMESNET] Embeddings → {embed_out}")
    print(f"[TIMESNET] Forecasts  → {forecast_out}")

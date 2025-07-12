import pathlib
import warnings
import joblib
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from .prepare_dataset_TimesNet import TimesNetDataset
from .models.TimesNet import TimesNetModel


def _loader(ds, batch, shuffle, sampler=None):
    return DataLoader(
        ds, batch_size=batch, shuffle=shuffle if sampler is None else False,
        sampler=sampler, num_workers=0, pin_memory=True
    )


def _class_weights(y):
    binc = torch.bincount(torch.tensor(y, dtype=torch.long) + 1)
    w = 1.0 / torch.clamp(binc.float(), min=1)
    return (w / w.mean()).tolist()


def train_timesnet(
        train_pt: str,
        test_pt: str | None = None,
        events_pkl: str | None = None,
        model_out: str = "timesnet_model.pt",
        embed_out: str = "timesnet_embeddings.parquet",
        forecast_out: str = "timesnet_forecast.parquet",
        seq_len: int = 288,
        epochs: int = 20,
        batch_size: int = 256,
        lr: float = 3e-4,
        device: str | None = None,
):
    train_raw = torch.load(train_pt, weights_only=False)
    X_train, y_train = train_raw["X"], train_raw["y"]
    print(f"[TIMESNET] Loaded train set: {len(X_train)} samples")

    if test_pt and pathlib.Path(test_pt).exists():
        test_raw = torch.load(test_pt, weights_only=False)
        X_val, y_val = test_raw["X"], test_raw["y"]
        print(f"[TIMESNET] Loaded test set: {len(X_val)} samples")
    else:
        split = int(0.9 * len(X_train))
        X_train, X_val = X_train[:split], X_train[split:]
        y_train, y_val = y_train[:split], y_train[split:]

    train_ds = TimesNetDataset(X_train, y_train)
    val_ds = TimesNetDataset(X_val, y_val)

    w = _class_weights(y_train)
    sample_weights = torch.tensor([w[y + 1] for y in y_train])
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = _loader(train_ds, batch_size, shuffle=False, sampler=sampler)
    val_loader = _loader(val_ds, batch_size, shuffle=False)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TIMESNET] Using device: {device}")

    model = TimesNetModel(
        seq_len=seq_len,
        n_features=X_train.shape[-1],
        d_model=128,
        n_blocks=4,
        num_classes=3,
    ).to(device)

    weights = torch.tensor(w, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", patience=3, factor=0.5, verbose=True
    )
    scaler = torch.amp.GradScaler(enabled=device.startswith("cuda"))

    best_val = float("inf")
    patience = 6

    for ep in range(1, epochs + 1):
        print(f"[TIMESNET] Epoch {ep}/{epochs} starting")
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), (yb + 1).to(device)
            optim.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=device.startswith("cuda")):
                logits, _ = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            train_loss += loss.item() * yb.size(0)

        model.eval()
        val_loss = 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), (yb + 1).to(device)
                logits, _ = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * yb.size(0)
                y_true.extend(yb.cpu().tolist())
                y_pred.extend(logits.argmax(1).cpu().tolist())

        train_loss /= len(train_ds)
        val_loss /= len(val_ds)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        print(f"[TIMESNET] ep {ep:02d}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}  f1={f1:.3f}")
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            patience = 6
            torch.save(model.state_dict(), model_out)
            print(f"[TIMESNET] New best val={best_val:.4f}, model saved")
        else:
            patience -= 1
            print(f"[TIMESNET] No improvement, patience={patience}")
            if patience == 0:
                print("[TIMESNET] Early stopping")
                break

    print(f"[TIMESNET] Training completed, best val={best_val:.4f}")

    model.load_state_dict(torch.load(model_out, map_location=device))
    model.eval()
    print("[TIMESNET] Generating embeddings and forecasts")

    full_ds = TimesNetDataset(
        np.concatenate([X_train, X_val], axis=0),
        np.concatenate([y_train, y_val], axis=0),
    )
    full_loader = _loader(full_ds, batch_size, shuffle=False)

    embeds_list, logits_list = [], []
    with torch.no_grad():
        for xb, _ in full_loader:
            xb = xb.to(device)
            logits, emb = model(xb)
            embeds_list.append(emb.cpu())
            logits_list.append(logits.cpu())

    embeds = torch.cat(embeds_list).numpy()
    probs = torch.softmax(torch.cat(logits_list), dim=1).numpy()[:, 2]

    if events_pkl and pathlib.Path(events_pkl).exists():
        ev = joblib.load(events_pkl).set_index("ts").sort_index()
        ts_idx = ev.index[seq_len: seq_len + len(probs)]
    else:
        ts_idx = pd.RangeIndex(len(probs))

    pd.DataFrame(embeds, index=ts_idx).to_parquet(embed_out)
    pd.DataFrame({"timesnet_pred": probs}, index=ts_idx).to_parquet(forecast_out)
    print(f"[TIMESNET] Embeddings saved to {embed_out}")
    print(f"[TIMESNET] Forecast saved to {forecast_out}")

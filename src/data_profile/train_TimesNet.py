# src/data_profile/train_TimesNet.py
import warnings, pathlib, joblib, torch, pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader

from .prepare_dataset_TimesNet import TimesNetDataset
from .models.TimesNet import TimesNetModel


def _make_loader(dataset, batch, shuffle):
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=0)


def train_timesnet(
    dataset_pt: str,
    events_pkl: str,
    model_out: str = "timesnet_model.pt",
    embed_out: str = "timesnet_embeddings.parquet",
    forecast_out: str = "timesnet_forecast.parquet",
    seq_len: int = 288,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 3e-4,
    device: str | None = None,
):
    # -------------------------------------------------------------- #
    # 1. Датасет: X (N, L, F) , y (N) = pct_change; конвертируем в 0/1
    # -------------------------------------------------------------- #
    print(f"[TimesNet] Starting training for {epochs} epochs on device {device}", flush=True)
    raw = torch.load(dataset_pt, weights_only=False)
    X_arr, y_arr = raw["X"], raw["y"]
    y_bin = (y_arr > 0).astype("float32")          # рост = 1, падение / ноль = 0
    ds = TimesNetDataset(X_arr, y_bin)
    train_loader = _make_loader(ds, batch_size, shuffle=True)
    eval_loader = _make_loader(ds, batch_size, shuffle=False)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = TimesNetModel(
        seq_len=seq_len,
        n_features=X_arr.shape[-1],
        d_model=128,
        n_blocks=3,
        num_classes=2,
    ).to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    # -------------------------------------------------------------- #
    # 2. Обучение
    # -------------------------------------------------------------- #
    for ep in range(1, epochs + 1):
        model.train()
        loss_total, n_total = 0.0, 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optim.zero_grad()
            logits, _ = model(Xb)
            loss = loss_fn(logits[:, 1], yb)   # используем логит «рост»
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            loss_total += loss.item() * yb.size(0)
            n_total += yb.size(0)
        print(f"[TimesNet] epoch {ep}/{epochs}  BCE={loss_total/n_total:.4f}")

    torch.save(model.state_dict(), model_out)
    print(f"[TimesNet] saved weights → {model_out}")

    # -------------------------------------------------------------- #
    # 3. Сохраняем эмбеддинги и прогнозы
    # -------------------------------------------------------------- #
    model.eval()
    embeds, preds = [], []
    with torch.no_grad():
        for Xb, _ in eval_loader:
            Xb = Xb.to(device)
            logits, emb = model(Xb)
            prob = torch.sigmoid(logits[:, 1])      # вероятность роста
            embeds.append(emb.cpu())
            preds.append(prob.cpu())
    embeds = torch.cat(embeds).numpy()
    preds = torch.cat(preds).numpy()

    # ---------- индексы ----------
    try:
        ev = joblib.load(events_pkl).set_index("ts").sort_index()
        ts_idx = ev.index[seq_len : seq_len + len(preds)]
    except Exception as e:
        warnings.warn(f"index fallback: {e}")
        ts_idx = pd.RangeIndex(len(preds))

    # ---------- parquet / csv ----------
    embed_df = pd.DataFrame(embeds, index=ts_idx)
    pred_df = pd.DataFrame({"timesnet_pred": preds}, index=ts_idx)

    def _safe(df, path):
        path = pathlib.Path(path)
        try:
            df.to_parquet(path)
        except Exception:
            df.to_csv(path.with_suffix(".csv"))

    _safe(embed_df, embed_out)
    _safe(pred_df, forecast_out)
    print(f"[TimesNet] embeddings → {embed_out} | forecast → {forecast_out}")

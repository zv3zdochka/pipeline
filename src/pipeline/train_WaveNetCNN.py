# src/pipeline/train_WaveNetCNN.py
from __future__ import annotations
import functools, pathlib, warnings, joblib
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from torch.nn.utils import clip_grad_norm_
from torchmetrics import Accuracy, F1Score
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import cross_entropy
from .prepare_dataset_CNN import CNNWindowDataset
from .models.WaveNetCNN import WaveCNN


# ---------- losses ---------------------------------------------------- #
def focal_loss(logits: torch.Tensor, target: torch.Tensor, *,
               gamma: float = 2.0,
               alpha: list[float] | tuple[float, ...] | None = None) -> torch.Tensor:
    if alpha is None:
        alpha = [1.0] * logits.size(1)
    ce = cross_entropy(logits, target, reduction="none")
    pt = torch.exp(-ce)
    w = torch.tensor(alpha, device=logits.device)[target]
    return (w * (1. - pt).pow(gamma) * ce).mean()


# ---------- helpers --------------------------------------------------- #
def _save_embeddings(mat: np.ndarray,
                     df_ref: pd.DataFrame,
                     window: int,
                     path: str | pathlib.Path):
    """mat.shape[0] == len(df_ref) - window + 1"""
    try:
        idx = df_ref.index[window - 1: window - 1 + len(mat)]
        if len(idx) != len(mat):
            raise ValueError
    except Exception:
        idx = pd.RangeIndex(len(mat))
        warnings.warn("Не удалось выровнять индексы – использую RangeIndex")
    out = pd.DataFrame(mat, index=idx)
    out.to_parquet(path)


# ---------- train ----------------------------------------------------- #
def train_wavecnn(*,
                  train_pkl: str | pathlib.Path,
                  test_pkl: str | pathlib.Path,
                  model_out: str | pathlib.Path,
                  emb_train_out: str | pathlib.Path | None = None,
                  emb_test_out: str | pathlib.Path,
                  window: int = 48,
                  epochs: int = 30,
                  batch: int = 256,
                  lr: float = 1e-4,
                  log_dir: str | pathlib.Path | None = None,
                  device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[WAVECNN] device → {device}")

    # ---------------- data ---------------- #
    train_df = joblib.load(train_pkl)
    test_df = joblib.load(test_pkl)
    train_ds = CNNWindowDataset(train_df, window_size=window)
    test_ds = CNNWindowDataset(test_df, window_size=window)

    # inverse-frequency α
    y_win = train_df["microtrend_label"].values.astype(np.int8)[window - 1:]
    freq = np.bincount(y_win + 1, minlength=3)
    alpha = (freq.max() / np.clip(freq, 1, None)).tolist()
    print(f"[WAVECNN] focal alpha → {alpha}")

    train_ld = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=2,
                          pin_memory=torch.cuda.is_available())
    test_ld = DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=2,
                         pin_memory=torch.cuda.is_available())

    # ---------------- model ---------------- #
    model = WaveCNN(in_channels=train_df.shape[1] - 1, window_size=window).to(device)
    print(f"[WAVECNN] RF covers window={window}")
    loss_fn = functools.partial(focal_loss, gamma=2.0, alpha=alpha)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    warm = max(1, int(epochs * 0.05))
    scheduler = SequentialLR(
        optim,
        [LinearLR(optim, 0.01, 1.0, warm), CosineAnnealingLR(optim, epochs - warm, 1e-6)],
        [warm]
    )

    acc_m = Accuracy(task="multiclass", num_classes=3).to(device)
    f1_m = F1Score(task="multiclass", num_classes=3, average="macro").to(device)

    writer = SummaryWriter(log_dir or "cache/runs/wavecnn")

    best_f1, no_imp, patience = 0., 0, 3
    for ep in range(1, epochs + 1):
        # ---- train
        model.train();
        run_loss = 0.
        for xb, yb in train_ld:
            xb = xb.to(device);
            yb = (yb + 1).to(device)
            optim.zero_grad(set_to_none=True)
            logits, _ = model(xb)
            loss = loss_fn(logits, yb);
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0);
            optim.step()
            run_loss += loss.item() * xb.size(0)
        scheduler.step()

        # ---- val
        model.eval();
        acc_m.reset();
        f1_m.reset()
        with torch.no_grad():
            for xb, yb in test_ld:
                xb = xb.to(device);
                yb = (yb + 1).to(device)
                preds = model(xb)[0].argmax(1)
                acc_m.update(preds, yb);
                f1_m.update(preds, yb)
        tr_loss = run_loss / len(train_ds)
        acc, f1 = acc_m.compute().item(), f1_m.compute().item()

        writer.add_scalars("metrics", {"f1_val": f1, "acc_val": acc}, ep)
        writer.add_scalar("loss/train", tr_loss, ep)
        print(f"[ep {ep:02d}/{epochs}] loss={tr_loss:.4f}  acc={acc:.3f}  f1={f1:.3f}")

        if f1 > best_f1:
            best_f1, no_imp = f1, 0
            torch.save(model.state_dict(), model_out)
            print(f"  ↳ new best F1={best_f1:.3f}  (model saved)")
        else:
            no_imp += 1
            if no_imp >= patience:
                print("  ↳ early-stopping");
                break
    writer.close()
    print(f"[WAVECNN] best F1 = {best_f1:.3f}")

    # ------------- embeddings ------------- #
    model.load_state_dict(torch.load(model_out, map_location=device))
    model.eval()
    for name, loader, df_ref, path in [
        ("train", train_ld, train_df, emb_train_out),
        ("test", test_ld, test_df, emb_test_out)
    ]:
        if path is None: continue
        embs = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(device);
                embs.append(model(xb)[1].cpu())
        _save_embeddings(torch.cat(embs).numpy(), df_ref, window, path)
        print(f"[WAVECNN] {name} embeddings → {path}")

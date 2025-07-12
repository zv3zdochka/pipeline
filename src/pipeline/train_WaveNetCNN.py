# src/pipeline/train_WaveNetCNN.py
from __future__ import annotations

import functools
import pathlib
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from torch.nn.utils import clip_grad_norm_
from torchmetrics import Accuracy, F1Score
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import cross_entropy

from .prepare_dataset_CNN import CNNWindowDataset
from .models.WaveNetCNN import WaveCNN


# --------------------------------------------------------------------- #
#                               LOSSES                                  #
# --------------------------------------------------------------------- #
def focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    gamma: float = 2.0,
    alpha: list[float] | tuple[float, ...] | None = None,
) -> torch.Tensor:
    """
    Focal loss со взвешиванием классов (alpha) и фокус-фактором gamma.
    При alpha=None используется равный вес для всех классов.
    """
    if alpha is None:
        alpha = [1.0] * logits.size(1)
    alpha_t = torch.tensor(alpha, device=logits.device)[target]  # (B,)
    ce = cross_entropy(logits, target, reduction="none")         # (B,)
    pt = torch.exp(-ce)
    loss = alpha_t * (1.0 - pt).pow(gamma) * ce
    return loss.mean()


# --------------------------------------------------------------------- #
#                              TRAIN LOOP                               #
# --------------------------------------------------------------------- #
def train_wavecnn(
    *,
    train_pkl: str | pathlib.Path,
    test_pkl: str | pathlib.Path,
    model_out: str | pathlib.Path,
    emb_out: str | pathlib.Path,
    window: int = 48,
    epochs: int = 30,
    batch: int = 256,
    lr: float = 1e-4,
    log_dir: str | pathlib.Path | None = None,
    device: str | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[WAVECNN] device → {device}")

    log_dir = log_dir or "cache/runs/wavecnn"
    writer = SummaryWriter(log_dir)
    print(f"[WAVECNN] TensorBoard → {log_dir}")

    # ------------------------------------------------------------------ #
    #                    ЗАГРУЗКА ДАННЫХ И ВЫЧИСЛЕНИЕ α                  #
    # ------------------------------------------------------------------ #
    train_df = joblib.load(train_pkl)
    test_df  = joblib.load(test_pkl)

    train_ds = CNNWindowDataset(train_df, window_size=window)
    test_ds  = CNNWindowDataset(test_df,  window_size=window)

    # Частоты *окон* (метка = правому краю окна)
    y_window = train_df["microtrend_label"].values.astype(np.int8)[window - 1 :]
    freqs    = np.bincount(y_window + 1, minlength=3)  # counts for [-1,0,1]
    raw_alpha = freqs.max() / np.clip(freqs, 1, None)
    # Ограничим веса, чтобы не было чрезмерного доминирования
    alpha    = np.clip(raw_alpha, 1.0, 5.0).tolist()
    print(f"[WAVECNN] focal alpha → {alpha}")

    # ------------------------------------------------------------------ #
    #                         DATALOADERS                               #
    # ------------------------------------------------------------------ #
    train_loader = DataLoader(
        train_ds,
        batch_size=batch,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.startswith("cuda")),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.startswith("cuda")),
    )

    # ------------------------------------------------------------------ #
    #                           MODEL + OPT                            #
    # ------------------------------------------------------------------ #
    model = WaveCNN(
        in_channels=train_df.shape[1] - 1,  # минус столбец microtrend_label
        window_size=window,
    ).to(device)
    print(f"[WAVECNN] RF covers window={window}")

    loss_fn = functools.partial(focal_loss, gamma=2.0, alpha=alpha)
    optim   = torch.optim.AdamW(model.parameters(), lr=lr)

    # Более длительный warm-up: 5% от total epochs
    warmup_epochs = max(1, int(epochs * 0.05))
    cosine  = CosineAnnealingLR(optim, T_max=epochs - warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(
        optim,
        schedulers=[
            LinearLR(optim, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs),
            cosine,
        ],
        milestones=[warmup_epochs],
    )

    acc_metric = Accuracy(task="multiclass", num_classes=3).to(device)
    f1_metric  = F1Score(task="multiclass", num_classes=3, average="macro").to(device)

    best_f1 = 0.0
    patience = 3
    no_improve = 0

    # ------------------------------------------------------------------ #
    #                             EPOCH LOOP                           #
    # ------------------------------------------------------------------ #
    for epoch in range(1, epochs + 1):
        # -------- TRAIN -------- #
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = (yb + 1).to(device)  # -1/0/1 → 0/1/2

            optim.zero_grad(set_to_none=True)
            logits, _ = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            running_loss += loss.item() * xb.size(0)

        scheduler.step()

        # -------- VALIDATION -------- #
        model.eval()
        acc_metric.reset(); f1_metric.reset()
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = (yb + 1).to(device)
                logits, _ = model(xb)
                preds = logits.argmax(1)
                acc_metric.update(preds, yb)
                f1_metric.update(preds, yb)

        train_loss = running_loss / len(train_ds)
        val_acc    = acc_metric.compute().item()
        val_f1     = f1_metric.compute().item()

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("acc/val",    val_acc,   epoch)
        writer.add_scalar("f1/val",     val_f1,    epoch)

        print(f"[ep {epoch:02d}/{epochs}] loss={train_loss:.4f}  acc={val_acc:.3f}  f1={val_f1:.3f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            no_improve = 0
            torch.save(model.state_dict(), model_out)
            print(f"  ↳ new best F1={best_f1:.3f}  (model saved)")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("  ↳ early-stopping")
                break

    writer.close()
    print(f"[WAVECNN] best F1 = {best_f1:.3f}")

    # ------------------------------------------------------------------ #
    #                          SAVE EMBEDDINGS                          #
    # ------------------------------------------------------------------ #
    model.load_state_dict(torch.load(model_out, map_location=device))
    model.eval()
    emb_list = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            _, emb = model(xb)
            emb_list.append(emb.cpu())
    emb_mat = torch.cat(emb_list).numpy()
    emb_df  = joblib.load(test_pkl).index[window - 1 :].to_frame().reset_index(drop=True)
    emb_df  = torch.tensor(emb_mat)  # или pd.DataFrame(emb_mat, index=test_df.index[window-1:])
    try:
        pd.DataFrame(emb_mat, index=train_df.index[window - 1 :]).to_parquet(emb_out)
    except Exception:
        csv_path = pathlib.Path(emb_out).with_suffix(".csv")
        pd.DataFrame(emb_mat, index=train_df.index[window - 1 :]).to_csv(csv_path)
        warnings.warn(f"Parquet save failed, wrote CSV → {csv_path}")

    print(f"[WAVECNN] embeddings → {emb_out}")

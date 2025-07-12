# src/pipeline/train_WaveNetCNN.py
from __future__ import annotations

import pathlib
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from torch.nn.utils import clip_grad_norm_
from sklearn.utils.class_weight import compute_class_weight
from torchmetrics import Accuracy, F1Score
from torch.utils.tensorboard import SummaryWriter

from .prepare_dataset_CNN import CNNWindowDataset
from .models.WaveNetCNN import WaveCNN


def make_loader(df, window, batch, shuffle):
    ds = CNNWindowDataset(df, window_size=window)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle,
                      pin_memory=torch.cuda.is_available(), num_workers=2)


def get_class_weights(y_labels: np.ndarray) -> torch.Tensor:
    classes = np.array([-1, 0, 1])
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_labels
    )
    return torch.tensor(weights, dtype=torch.float32)


def train_wavecnn(
        train_pkl: str | pathlib.Path,
        test_pkl: str | pathlib.Path,
        class_freqs_pt: str | pathlib.Path,
        model_out: str | pathlib.Path,
        emb_out: str | pathlib.Path,
        window: int = 24,
        epochs: int = 10,
        batch: int = 256,
        lr: float = 3e-4,
        warmup_epochs: int = 3,
        log_dir: str | pathlib.Path = None,
        device: str = None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[WAVECNN] Using device: {device}")

    log_dir = log_dir or "cache/runs/wavecnn"
    writer = SummaryWriter(log_dir)
    print(f"[WAVECNN] TensorBoard logs -> {log_dir}")

    print(f"[WAVECNN] Loading train dataset from {train_pkl}")
    train_df = joblib.load(train_pkl)
    print(f"[WAVECNN] Loading test  dataset from {test_pkl}")
    test_df = joblib.load(test_pkl)

    train_loader = make_loader(train_df, window, batch, shuffle=True)
    test_loader = make_loader(test_df, window, batch, shuffle=False)

    model = WaveCNN(
        in_channels=train_df.shape[1] - 1,
        window_size=window
    ).to(device)
    print(f"[WAVECNN] Model initialized with window_size={window}")

    y_train = train_df["microtrend_label"].to_numpy()
    weights = get_class_weights(y_train).to(device)
    print(f"[WAVECNN] Class weights: {weights.tolist()}")

    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    cosine = CosineAnnealingLR(optim, T_max=epochs - warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(
        optim,
        schedulers=[
            LinearLR(optim, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs),
            cosine
        ],
        milestones=[warmup_epochs]
    )
    print(f"[WAVECNN] Scheduler: {warmup_epochs}-epoch warmup + CosineAnnealingLR")

    acc_metric = Accuracy(task="multiclass", num_classes=3).to(device)
    f1_metric = F1Score(task="multiclass", num_classes=3, average="macro").to(device)

    best_f1 = 0.0
    early_stop_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = (y + 1).to(device)  # -1/0/1 -> 0/1/2

            optim.zero_grad()
            logits, _ = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            train_loss += float(loss) * x.size(0)

        scheduler.step()

        model.eval()
        acc_metric.reset()
        f1_metric.reset()
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y_true = (y + 1).to(device)
                logits, _ = model(x)
                preds = logits.argmax(dim=1)
                acc_metric.update(preds, y_true)
                f1_metric.update(preds, y_true)

        val_acc = acc_metric.compute().item()
        val_f1 = f1_metric.compute().item()
        avg_loss = train_loss / len(train_df)

        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("val/accuracy", val_acc, epoch)
        writer.add_scalar("val/f1", val_f1, epoch)

        print(f"[WAVECNN] [Epoch {epoch}/{epochs}] "
              f"loss={avg_loss:.4f} acc={val_acc:.3f} f1={val_f1:.3f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            early_stop_counter = 0
            torch.save(model.state_dict(), model_out)
            print(f"[WAVECNN] Saved best model to {model_out}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= 3:
                print(f"[WAVECNN] Early stopping triggered at epoch {epoch}")
                break

    writer.close()
    print(f"[WAVECNN] Best test F1: {best_f1:.3f}")

    model.load_state_dict(torch.load(model_out))
    model.eval()
    emb_list = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            _, emb = model(x)
            emb_list.append(emb.cpu())
    emb_mat = torch.cat(emb_list).numpy()
    emb_df = pd.DataFrame(emb_mat, index=test_df.index[window - 1:])
    try:
        emb_df.to_parquet(emb_out)
    except Exception:
        emb_df.to_csv(pathlib.Path(emb_out).with_suffix(".csv"))
        warnings.warn("Saved embeddings as CSV instead of parquet")
    print(f"[WAVECNN] Embeddings saved to {emb_out}")

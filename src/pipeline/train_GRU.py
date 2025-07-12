from __future__ import annotations

import pathlib
import warnings
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score
from .models.GRU import MicroTrendGRU


def _compute_class_weights_from_labels(labels: np.ndarray) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from an array of integer labels [0,1,2].
    """
    counts = np.bincount(labels, minlength=3)
    total = counts.sum()
    weights = total / np.maximum(counts, 1)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def train_gru(
        train_pkl: str | pathlib.Path,
        test_pkl: str | pathlib.Path,
        class_freqs_pt: str | pathlib.Path,
        events_pkl: str | pathlib.Path,
        emb_path: str | pathlib.Path,
        model_out: str = "gru_model.pt",
        emb_out: str = "gru_embeddings.parquet",
        seq_len: int = 96,
        epochs: int = 20,
        batch_size: int = 128,
        lr: float = 3e-4,
        patience: int = 3,
        device: str | None = None
):
    """
    Train a GRU to predict microtrend labels with early stopping, learning-rate scheduling,
    and logging to TensorBoard. Embeddings of the hidden state are extracted after training.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GRU] Using device: {device}")

    print(f"[GRU] Loading train dataset from {train_pkl}")
    ds_train = joblib.load(train_pkl)
    print(f"[GRU] Loading validation dataset from {test_pkl}")
    ds_val = joblib.load(test_pkl)

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=(device == "cuda"))
    loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=(device == "cuda"))

    input_dim = ds_train.features.shape[1]
    model = MicroTrendGRU(input_dim=input_dim).to(device)
    print("[GRU] Model initialized.")

    weights = _compute_class_weights_from_labels(ds_train.labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    writer = SummaryWriter(log_dir="cache/runs/GRU")
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None

    print(f"[GRU] Starting training for up to {epochs} epochs")
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for X, y in loader_train:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(X)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        model.eval()
        val_losses, all_preds, all_trues = [], [], []
        with torch.no_grad():
            for X, y in loader_val:
                X, y = X.to(device), y.to(device)
                logits, _ = model(X)
                loss = criterion(logits, y)
                val_losses.append(loss.item())
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.append(preds)
                all_trues.append(y.cpu().numpy())

        mean_train_loss = np.mean(train_losses)
        mean_val_loss = np.mean(val_losses)
        all_preds = np.concatenate(all_preds)
        all_trues = np.concatenate(all_trues)
        val_acc = accuracy_score(all_trues, all_preds)
        val_f1 = f1_score(all_trues, all_preds, average="macro")

        writer.add_scalars("Loss", {"train": mean_train_loss, "val": mean_val_loss}, epoch)
        writer.add_scalars("Metrics", {"val_acc": val_acc, "val_f1": val_f1}, epoch)

        print(
            f"[GRU][Epoch {epoch}] train_loss={mean_train_loss:.4f}, val_loss={mean_val_loss:.4f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}")

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_epoch = epoch
            best_state = model.state_dict()
            print(f"[GRU] New best model at epoch {epoch}, val_loss={mean_val_loss:.4f}")
        elif epoch - best_epoch >= patience:
            print(f"[GRU] Early stopping at epoch {epoch}. Best epoch was {best_epoch}.")
            break

    writer.close()

    if best_state is not None:
        torch.save(best_state, model_out)
        print(f"[GRU] Saved best model state_dict to {model_out}")

    # extract hidden embeddings using the best model
    model.load_state_dict(torch.load(model_out))
    model.eval()
    embs = []
    with torch.no_grad():
        for X, _ in DataLoader(ds_train, batch_size=batch_size, shuffle=False,
                               num_workers=4, pin_memory=(device == "cuda")):
            X = X.to(device)
            _, h = model(X)
            embs.append(h.cpu())
        for X, _ in DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                               num_workers=4, pin_memory=(device == "cuda")):
            X = X.to(device)
            _, h = model(X)
            embs.append(h.cpu())

    gru_emb = torch.cat(embs, dim=0).numpy()

    try:
        df_orig = (pd.read_parquet(emb_path) if emb_path.endswith(".parquet")
                   else pd.read_csv(emb_path, index_col=0, parse_dates=True))
        ts_idx = df_orig.index[seq_len - 1: seq_len - 1 + len(gru_emb)]
        print(f"[GRU] Aligned embeddings to original timestamps from {emb_path}")
    except Exception:
        warnings.warn("Falling back to raw events timestamps")
        print("[GRU] Falling back to raw events timestamps")
        ev = joblib.load(events_pkl).set_index("ts").sort_index()
        ts_idx = ev.index[seq_len - 1: seq_len - 1 + len(gru_emb)]

    df_out = pd.DataFrame(gru_emb, index=ts_idx)
    try:
        df_out.to_parquet(emb_out)
        print(f"[GRU] Saved embeddings to {emb_out}")
    except Exception:
        csv_out = emb_out.replace(".parquet", ".csv")
        df_out.to_csv(csv_out)
        print(f"[GRU] Saved embeddings to {csv_out}")

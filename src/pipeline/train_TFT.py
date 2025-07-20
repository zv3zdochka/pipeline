# src/train/train_tft.py
import os
import warnings

# Suppress TensorFlow / XLA logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning)

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score

from .models.TFT import TemporalFusionTransformer
from .prepare_dataset_TFT import TFTDataset

# Configure logger
t_logger = logging.getLogger("TFT")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def train_tft(
    *,
    train_features_path: str | Path,
    train_targets_path: str | Path,
    test_features_path: str | Path,
    test_targets_path: str | Path,
    feature_groups_pkl: str | Path,
    class_freqs_pt: str | Path,
    seq_len: int,
    model_out: str | Path = "tft_model.pt",
    emb_out: str | Path = "tft_embeddings.parquet",
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 3e-4,
    step_size: int = 5,
    gamma: float = 0.5,
    patience: int = 5,
    num_workers: int = 0,
    device: str | None = None,
):
    """
    Train Temporal Fusion Transformer with class balancing, learning-rate
    scheduling, early stopping, and embedding extraction.

    Args:
        train_features_path: Numpy .npy file with training feature matrix.
        train_targets_path:  Numpy .npy file with training targets (original labels -1/0/1).
        test_features_path:  Numpy .npy file with validation feature matrix.
        test_targets_path:   Numpy .npy file with validation targets.
        feature_groups_pkl:  Joblib file containing dict with keys
                              {"continuous", "categorical", "binary"} listing feature names.
        class_freqs_pt:      Torch-saved dict of class counts for original labels {-1,0,1}.
        seq_len:             Sliding window length.
        model_out:           Path to save best model state dict.
        emb_out:             Path to save validation embeddings parquet.
        epochs:              Max training epochs.
        batch_size:          DataLoader batch size.
        lr:                  Initial learning rate.
        step_size:           Step size for StepLR.
        gamma:               Multiplicative factor for LR schedulers.
        patience:            Early stopping patience (epochs w/o F1 improvement).
        num_workers:         DataLoader workers.
        device:              Override device ("cuda" / "cpu").
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    t_logger.info("TRAINING STARTED on %s", device)

    # Load data
    X_train = np.load(train_features_path)
    y_train = np.load(train_targets_path)
    X_test = np.load(test_features_path)
    y_test = np.load(test_targets_path)

    train_ds = TFTDataset(X_train, y_train, seq_len)
    test_ds = TFTDataset(X_test, y_test, seq_len)

    # Feature groups
    groups = joblib.load(feature_groups_pkl)
    cont_dim = len(groups["continuous"]) + len(groups["binary"])
    n_cat = len(groups["categorical"])
    cat_cards = [256] * n_cat  # Placeholder; adjust if real cardinalities available.

    # Model
    model = TemporalFusionTransformer(
        seq_len=seq_len,
        n_cont_dim=cont_dim,
        n_cat=n_cat,
        cat_cardinalities=cat_cards,
        d_model=128,
        num_classes=3,
    ).to(device)

    # Class weights (inverse frequency)
    freqs = torch.load(class_freqs_pt)
    weights = torch.tensor(
        [sum(freqs.values()) / max(freqs.get(c, 1), 1) for c in (-1, 0, 1)],
        dtype=torch.float32,
    )
    weights /= weights.mean()
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    # Sampler for balanced training windows
    window_labels = y_train[seq_len:].astype(int)
    sample_weights = weights[window_labels + 1]
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Optimizer & schedulers
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    step_scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    plateau_scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=gamma, patience=2
    )

    best_val_f1 = -1.0
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, preds, targets = 0.0, [], []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).long()
            cont = xb[..., :cont_dim]
            cat = xb[..., cont_dim:].long() if n_cat else None

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(cont, cat)
            # Shift labels from {-1,0,1} to {0,1,2}
            loss = criterion(logits, yb + 1)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * yb.size(0)
            preds.extend((logits.argmax(1).cpu() - 1).tolist())
            targets.extend(yb.cpu().tolist())

        avg_loss = total_loss / len(train_ds)
        train_f1 = f1_score(targets, preds, average="macro")
        lr_current = optimizer.param_groups[0]["lr"]
        t_logger.info(
            "Epoch %2d/%d  loss=%.4f  f1_macro=%.3f  lr=%.2e",
            epoch,
            epochs,
            avg_loss,
            train_f1,
            lr_current,
        )

        step_scheduler.step()

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device).long()
                cont = xb[..., :cont_dim]
                cat = xb[..., cont_dim:].long() if n_cat else None
                logits, _ = model(cont, cat)
                val_preds.extend((logits.argmax(1).cpu() - 1).tolist())
                val_targets.extend(yb.cpu().tolist())
        val_f1 = f1_score(val_targets, val_preds, average="macro")
        t_logger.info("Validation f1_macro=%.3f", val_f1)
        plateau_scheduler.step(val_f1)

        # Early stopping / checkpoint
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_out)
            t_logger.info(
                "New best model saved (val f1_macro=%.3f) → %s",
                best_val_f1,
                model_out,
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                t_logger.info(
                    "Early stopping: no improvement for %d epochs (stopped at epoch %d)",
                    patience,
                    epoch,
                )
                break

    # Embedding extraction from best model
    model.load_state_dict(torch.load(model_out, map_location=device))
    model.eval()
    embs = []
    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(device)
            cont = xb[..., :cont_dim]
            cat = xb[..., cont_dim:].long() if n_cat else None
            _, emb = model(cont, cat)
            embs.append(emb.cpu())

    emb_mat = torch.cat(embs).numpy()
    try:
        existing = pd.read_parquet(emb_out).index
        if len(existing) != len(emb_mat):
            raise ValueError
        idx = existing
    except Exception:
        idx = pd.RangeIndex(len(emb_mat))
    pd.DataFrame(emb_mat, index=idx).to_parquet(emb_out)
    t_logger.info("Embeddings saved → %s", emb_out)

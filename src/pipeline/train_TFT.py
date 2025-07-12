# src/train/train_tft.py
from __future__ import annotations

import pathlib
import joblib
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import pandas as pd

from src.models.TFT import TemporalFusionTransformer
from src.pipeline.prepare_dataset_TFT import TFTDataset


def train_tft(
        *,
        train_dataset_path: str | pathlib.Path,
        test_dataset_path: str | pathlib.Path,
        feature_groups_pkl: str | pathlib.Path,
        class_freqs_pt: str | pathlib.Path,
        model_out: str = "tft_model.pt",
        emb_out: str = "tft_embeddings.parquet",
        epochs: int = 10,
        batch_size: int = 128,
        lr: float = 3e-4,
        step_size: int = 5,
        gamma: float = 0.5,
        device: str | None = None,
):
    print("[TFT] TRAINING STARTED")

    # Load train and test datasets
    train_raw = torch.load(train_dataset_path)
    test_raw = torch.load(test_dataset_path)
    X_train, y_train = train_raw["X"], train_raw["y"]
    X_test, y_test = test_raw["X"], test_raw["y"]

    train_ds = TFTDataset(X_train, y_train)
    test_ds = TFTDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # Feature groups
    groups = joblib.load(feature_groups_pkl)
    cont_dim = len(groups["continuous"]) + len(groups["binary"])
    n_cat = len(groups["categorical"])
    cat_cards = [256] * n_cat

    # Model
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalFusionTransformer(
        seq_len=X_train.shape[1],
        n_cont_dim=cont_dim,
        n_cat=n_cat,
        cat_cardinalities=cat_cards,
        d_model=128,
        num_classes=3,
    ).to(device)

    # Loss with class weights
    freqs = torch.load(class_freqs_pt)
    weights = torch.tensor(
        [sum(freqs.values()) / max(freqs.get(c, 1), 1) for c in (-1, 0, 1)],
        dtype=torch.float32,
    )
    weights /= weights.mean()
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    # Optimizer and schedulers
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    step_scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    plateau_scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        train_preds, train_targets = [], []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            cont = xb[..., :cont_dim]
            cat = xb[..., cont_dim:].long() if n_cat else None

            optimizer.zero_grad()
            logits, _ = model(cont, cat)
            loss = criterion(logits, yb + 1)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * yb.size(0)
            train_preds.extend((logits.argmax(dim=1).cpu() - 1).tolist())
            train_targets.extend(yb.cpu().tolist())

        train_f1 = f1_score(train_targets, train_preds, average="macro")
        avg_loss = total_loss / len(train_ds)
        print(f"[TFT] Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  f1_macro={train_f1:.3f}")

        # step scheduler
        step_scheduler.step()

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                cont = xb[..., :cont_dim]
                cat = xb[..., cont_dim:].long() if n_cat else None
                logits, _ = model(cont, cat)
                val_preds.extend((logits.argmax(dim=1).cpu() - 1).tolist())
                val_targets.extend(yb.cpu().tolist())

        val_f1 = f1_score(val_targets, val_preds, average="macro")
        print(f"[TFT] Validation f1_macro={val_f1:.3f}")

        # plateau scheduler on validation F1
        plateau_scheduler.step(val_f1)

    # Save model
    torch.save(model.state_dict(), model_out)
    print(f"[TFT] Model saved to {model_out}")

    # Extract embeddings on test set
    model.eval()
    embs = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            cont = xb[..., :cont_dim]
            cat = xb[..., cont_dim:].long() if n_cat else None
            _, emb = model(cont, cat)
            embs.append(emb.cpu())
    emb_mat = torch.cat(embs).numpy()

    try:
        idx = pd.read_parquet(emb_out).index
        if len(idx) != len(emb_mat):
            raise ValueError
    except Exception:
        idx = pd.RangeIndex(len(emb_mat))

    pd.DataFrame(emb_mat, index=idx).to_parquet(emb_out)
    print(f"[TFT] Embeddings saved to {emb_out}")

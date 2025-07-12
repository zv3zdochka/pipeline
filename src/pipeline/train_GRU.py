# src/pipeline/train_GRU.py

from __future__ import annotations

import functools
import pathlib
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score

from .prepare_dataset_GRU import GRUSequenceDataset
from .models.GRU import MicroTrendGRU


def focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    gamma: float = 2.0,
    alpha: list[float] | None = None,
) -> torch.Tensor:
    """
    Focal loss с фокус-фактором gamma и весами alpha (inverse freq).
    Если alpha=None, используются единичные весы.
    """
    if alpha is None:
        alpha = [1.0] * logits.size(1)
    alpha_t = torch.tensor(alpha, device=logits.device)[target]    # (B,)
    ce = cross_entropy(logits, target, reduction="none")          # (B,)
    pt = torch.exp(-ce)
    return (alpha_t * (1.0 - pt).pow(gamma) * ce).mean()


def train_gru(
    *,
    train_pkl: str | pathlib.Path,
    test_pkl:  str | pathlib.Path,
    class_freqs_pt: str | pathlib.Path,  # остаётся для совместимости, но не нужен
    events_pkl:    str | pathlib.Path,   # тоже больше не используем
    emb_path:      str | pathlib.Path,   # idem
    model_out:    str | pathlib.Path = "gru_model.pt",
    emb_out:      str | pathlib.Path = "gru_embeddings.parquet",
    seq_len:      int = 96,
    epochs:       int = 20,
    batch_size:   int = 128,
    lr:          float = 1e-3,
    patience:     int = 5,
    device:      str | None = None,
):
    """
    Обучение GRU с focal_loss, AdamW + ReduceLROnPlateau по val_f1 и ранней остановкой по F1.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GRU] device → {device}")

    # --- Загружаем уже готовые GRUSequenceDataset из pickle ---
    train_ds: GRUSequenceDataset = joblib.load(train_pkl)
    val_ds:   GRUSequenceDataset = joblib.load(test_pkl)

    # --- Считаем alpha для focal_loss по распределению оконных меток ---
    #    Метки в датасете — уже {0,1,2}, то есть (microtrend_label+1)
    labels = train_ds.labels  # numpy array shape=(N,)
    freq  = np.bincount(labels, minlength=3)
    alpha = (freq.max() / np.clip(freq, 1, None)).tolist()
    print(f"[GRU] focal alpha → {alpha}")

    # --- Сэмплер, чтобы дополнительно балансировать выборку (можно убрать, если не нужно) ---
    #    Берём метки для окон: label для каждого окна = labels[seq_len-1:]
    win_labels = labels[seq_len - 1 :]
    sample_weights = torch.tensor([alpha[int(l)] for l in win_labels], dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device == "cuda"),
    )

    # --- Модель, оптимизатор, scheduler, tensorboard ---
    model = MicroTrendGRU(input_dim=train_ds.features.shape[1]).to(device)
    print("[GRU] Model initialized.")

    loss_fn = functools.partial(focal_loss, gamma=2.0, alpha=alpha)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, verbose=True)

    writer = SummaryWriter(log_dir="cache/runs/GRU")

    best_f1 = 0.0
    no_improve = 0

    # --- Цикл обучения ---
    for epoch in range(1, epochs + 1):
        # train
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * X.size(0)
        train_loss = total_loss / len(train_loader.dataset)

        # validation
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits, _ = model(X)
                p = logits.argmax(1).cpu().numpy()
                preds.extend(p)
                trues.extend(y.cpu().numpy())

        val_f1  = f1_score(trues, preds, average="macro", zero_division=0)
        val_acc = accuracy_score(trues, preds)

        # шаг scheduler по метрике
        scheduler.step(val_f1)

        # логгируем
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("acc/val",   val_acc,    epoch)
        writer.add_scalar("f1/val",    val_f1,     epoch)
        print(f"[GRU][Epoch {epoch:02d}/{epochs}] "
              f"train_loss={train_loss:.4f}  "
              f"val_acc={val_acc:.3f}  val_f1={val_f1:.3f}")

        # ранняя остановка по F1
        if val_f1 > best_f1:
            best_f1 = val_f1
            no_improve = 0
            torch.save(model.state_dict(), model_out)
            print(f"  ↳ new best F1={best_f1:.3f} (model saved → {model_out})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  ↳ early stopping (no improve for {patience} epochs)")
                break

    writer.close()
    print(f"[GRU] best F1 = {best_f1:.3f}")

    # --- Сохраняем эмбеддинги последней эпохи на валидации ---
    model.load_state_dict(torch.load(model_out, map_location=device))
    model.eval()
    embs = []
    with torch.no_grad():
        for X, _ in val_loader:
            X = X.to(device)
            _, h = model(X)
            embs.append(h.cpu())
    emb_mat = torch.cat(embs, dim=0).numpy()
    pd.DataFrame(emb_mat).to_parquet(emb_out)
    print(f"[GRU] embeddings → {emb_out}")

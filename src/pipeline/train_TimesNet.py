from __future__ import annotations

import math
import pathlib
import time
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score

from .models.TimesNet import TimesNetModel


class TimesNetDataset(torch.utils.data.Dataset):
    """Wrapper around dict{'X','y'} loaded from .pt files producing sliding windows."""

    def __init__(self, data: dict[str, np.ndarray], seq_len: int):
        self.X = data["X"].astype("float32")
        self.y = data["y"].astype("int8")
        self.seq_len = seq_len

    def __len__(self):
        return len(self.y) - self.seq_len

    def __getitem__(self, idx: int):
        j = idx + self.seq_len
        x_win = self.X[idx:j]
        y_lab = int(self.y[j]) + 1  # shift to 0/1/2
        return torch.from_numpy(x_win), y_lab, j


def _calc_class_weights(labels: np.ndarray) -> torch.Tensor:
    idx = labels + 1
    cnts = Counter(idx)
    tot = sum(cnts.values())
    w = {c: tot / (len(cnts) * n) for c, n in cnts.items()}
    return torch.tensor([w[0], w[1], w[2]], dtype=torch.float32)


def _epoch(
    model,
    loader,
    optimizer=None,
    device="cpu",
    weights: torch.Tensor | None = None,
):
    training = optimizer is not None
    total_loss, ys, ys_pred = 0.0, [], []
    for X, y, _ in loader:
        X = X.to(device)
        y = y.to(device)
        logits, _ = model(X)
        loss = F.cross_entropy(logits, y, weight=weights)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        total_loss += loss.item() * len(y)
        ys.append(y.detach().cpu())
        ys_pred.append(logits.argmax(dim=1).cpu())
    ys = torch.cat(ys).numpy()
    ys_pred = torch.cat(ys_pred).numpy()
    f1 = f1_score(ys, ys_pred, average="macro")
    acc = accuracy_score(ys, ys_pred)
    return total_loss / len(loader.dataset), acc, f1


def train_timesnet(
    *,
    train_pt: str | pathlib.Path,
    test_pt: str | pathlib.Path,
    events_pkl: str | pathlib.Path,
    model_out: str | pathlib.Path,
    embed_out: str | pathlib.Path,
    forecast_out: str | pathlib.Path,
    seq_len: int,
    epochs: int = 25,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: str | None = None,
    patience: int = 6,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TIMESNET] device → {device}")

    train_data = torch.load(train_pt, weights_only=False)
    test_data = torch.load(test_pt, weights_only=False)

    n_features = train_data["X"].shape[1]

    ds_train = TimesNetDataset(train_data, seq_len)
    ds_test = TimesNetDataset(test_data, seq_len)

    dl_train = torch.utils.data.DataLoader(
        ds_train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    dl_test = torch.utils.data.DataLoader(
        ds_test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    model = TimesNetModel(
        seq_len=seq_len,
        n_features=n_features,
        d_model=128,
        n_blocks=4,
        num_classes=3,
    ).to(device)

    class_weights = _calc_class_weights(train_data["y"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_f1, epochs_no_improve = -math.inf, 0
    best_state = None

    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = _epoch(model, dl_train, optimizer, device, class_weights)
        val_loss, val_acc, val_f1 = _epoch(model, dl_test, None, device, class_weights)
        print(
            f"[ep {ep:02d}/{epochs}] "
            f"tr_loss={tr_loss:.4f}  val_acc={val_acc:.3f}  val_f1={val_f1:.3f} "
            f"({time.time()-t0:.1f}s)"
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_no_improve = 0
            best_state = model.state_dict()
            torch.save(best_state, model_out)
            print(f"  ↳ new best F1={best_f1:.3f} (model saved → {model_out})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("  ↳ early stopping")
                break

    print(f"[TIMESNET] best F1 = {best_f1:.3f}")
    model.load_state_dict(best_state)
    model.eval()

    def _extract(loader):
        prob, emb, lab, idxs = [], [], [], []
        with torch.no_grad():
            for X, y, j in loader:
                X = X.to(device)
                logits, e = model(X)
                prob.append(torch.softmax(logits, dim=1).cpu())
                emb.append(e.cpu())
                lab.append((y - 1))
                idxs.append(j)
        return (
            torch.cat(prob).numpy(),
            torch.cat(emb).numpy(),
            torch.cat(lab).numpy(),
            torch.cat(idxs).numpy(),
        )

    prob_tr, emb_tr, y_tr, idx_tr = _extract(dl_train)
    prob_te, emb_te, y_te, idx_te = _extract(dl_test)

    prob_all = np.concatenate([prob_tr, prob_te])
    emb_all = np.concatenate([emb_tr, emb_te])
    y_all = np.concatenate([y_tr, y_te])
    idx_all = np.concatenate([idx_tr, idx_te])

    events = joblib.load(events_pkl)
    ts = events.loc[idx_all, "ts"].reset_index(drop=True)

    df_emb = pd.DataFrame(
        emb_all,
        columns=[f"emb_{i}" for i in range(emb_all.shape[1])],
    )
    df_emb["ts"] = ts
    df_emb.to_parquet(embed_out, index=False)
    print(f"[TIMESNET] embeddings → {embed_out}")

    df_fc = pd.DataFrame(prob_all, columns=["p_-1", "p_0", "p_1"])
    df_fc["pred"] = prob_all.argmax(axis=1) - 1
    df_fc["true"] = y_all
    df_fc["ts"] = ts
    df_fc.to_parquet(forecast_out, index=False)
    print(f"[TIMESNET] forecast → {forecast_out}")

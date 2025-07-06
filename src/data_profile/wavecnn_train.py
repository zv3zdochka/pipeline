# wavecnn_train.py
import argparse
import pathlib
import joblib
import warnings
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import accuracy_score, f1_score

from .data_cnn import CNNWindowDataset  # updated import path
from .models.wave_cnn import WaveCNN


def _make_loader(df, window, batch, shuffle):
    ds = CNNWindowDataset(df, window_size=window)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0)


def _compute_class_weights(freqs: dict[int, int]) -> torch.Tensor:
    total = sum(freqs.values())
    weights = torch.tensor(
        [total / max(freqs.get(lbl, 1), 1) for lbl in (-1, 0, 1)],
        dtype=torch.float32,
    )
    return weights / weights.mean()  # mean‑normalise for stability


def _safe_save_embeddings(emb_df: pd.DataFrame, path: str | pathlib.Path) -> None:
    """Save *emb_df* to Parquet if engine available, otherwise CSV."""
    path = pathlib.Path(path)
    try:
        emb_df.to_parquet(path)
        print(f"[INFO] Saved embeddings {emb_df.shape} → {path}")
    except (ImportError, ValueError):
        fallback = path.with_suffix(".csv")
        emb_df.to_csv(fallback, index=True)
        warnings.warn(
            "pyarrow/fastparquet not installed – saved embeddings as CSV instead.",
            RuntimeWarning,
        )
        print(f"[INFO] Saved embeddings {emb_df.shape} → {fallback}")


def train_wavecnn(
    dataset_pkl: str | pathlib.Path,
    class_freqs_pt: str | pathlib.Path,
    model_out: str | pathlib.Path = "wavecnn_model.pt",
    emb_out: str | pathlib.Path = "cnn_embeddings.parquet",
    window: int = 24,
    epochs: int = 10,
    batch: int = 256,
    lr: float = 3e-4,
    device: str | None = None,
):
    # ---------- 0. load ----------
    df: pd.DataFrame = joblib.load(dataset_pkl)
    n_channels = len(df.columns) - 1  # exclude label
    train_loader = _make_loader(df, window, batch, shuffle=True)

    # ---------- 1. model ----------
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = WaveCNN(in_channels=n_channels).to(device)

    class_freqs = torch.load(class_freqs_pt)
    weights = _compute_class_weights(class_freqs).to(device)

    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    # ---------- 2. train ----------
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, preds, gold = 0.0, [], []
        for x, y_raw in train_loader:
            y_enc = (y_raw + 1).to(device)  # {-1,0,1} → {0,1,2}
            x = x.to(device)

            optim.zero_grad()
            logits, _ = model(x)
            loss = loss_fn(logits, y_enc)
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            epoch_loss += loss.item() * y_enc.size(0)
            pred_enc = logits.argmax(1).cpu()
            preds.extend((pred_enc - 1).tolist())  # back to {-1,0,1}
            gold.extend(y_raw.tolist())

        acc = accuracy_score(gold, preds)
        f1 = f1_score(gold, preds, average="macro")
        print(
            f"[{epoch:02d}/{epochs}] "
            f"loss={epoch_loss/len(df):.4f} | acc={acc:.3f} | f1={f1:.3f}"
        )

    torch.save(model.state_dict(), model_out)
    print(f"[INFO] Saved model → {model_out}")

    # ---------- 3. embeddings ----------
    model.eval()
    emb_list, ts_list = [], df.index[window - 1 :]
    with torch.no_grad():
        for x, _ in _make_loader(df, window, batch, shuffle=False):
            _, emb = model(x)  # emb размер (B, emb_dim)
            emb_list.append(emb.cpu())

    emb_mat = torch.cat(emb_list).numpy()  # (N, emb_dim)
    ts_list = df.index[window - 1:]
    emb_df = pd.DataFrame(emb_mat, index=ts_list)
    _safe_save_embeddings(emb_df, emb_out)


# -------------------------------------------------------------------------------------
# CLI helper
# -------------------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train WaveNet+1D-CNN and export embeddings")
    ap.add_argument("--dataset", default="wavecnn_dataset.pkl", help="path to serialized dataset")
    ap.add_argument("--freqs", default="class_freqs.pt", help="path to class freqs")
    ap.add_argument("--model", default="wavecnn_model.pt", help="where to save model weights")
    ap.add_argument("--emb", default="cnn_embeddings.parquet", help="where to save embeddings (Parquet/CSV)")
    ap.add_argument("--epochs", type=int, default=10, help="number of epochs")
    ap.add_argument("--window", type=int, default=24, help="CNN window size")
    ap.add_argument("--batch", type=int, default=256, help="batch size")
    ap.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    args = ap.parse_args()

    train_wavecnn(
        dataset_pkl=args.dataset,
        class_freqs_pt=args.freqs,
        model_out=args.model,
        emb_out=args.emb,
        epochs=args.epochs,
        window=args.window,
        batch=args.batch,
        lr=args.lr,
    )

from __future__ import annotations
import pathlib, joblib, warnings
import torch, pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from .models.TFT import TemporalFusionTransformer


# ---------------------------------------------------------
# helpers
# ---------------------------------------------------------

def _loader(ds, batch, shuffle):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0)


def _class_weights(freqs: dict[int, int]) -> torch.Tensor:
    total = sum(freqs.values())
    w = torch.tensor([total / max(freqs.get(i, 1), 1) for i in (-1, 0, 1)], dtype=torch.float32)
    return w / w.mean()


# ---------------------------------------------------------
# main entry
# ---------------------------------------------------------

def train_tft(
        *,
        dataset_pt: str | pathlib.Path,
        feature_groups_pkl: str | pathlib.Path,
        class_freqs_pt: str | pathlib.Path,
        model_out: str = "tft_model.pt",
        emb_out: str = "tft_embeddings.parquet",
        epochs: int = 10,
        batch_size: int = 128,
        lr: float = 3e-4,
        d_model: int = 128,
        device: str | None = None,
):
    """End‑to‑end обучение TFT + экспорт эмбеддингов."""

    # --- Dataset ----------------------------------------------------------------
    raw = torch.load(dataset_pt, weights_only=False)
    X, y = raw["X"], raw["y"]  # X:(N,L,D)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    train_loader = _loader(ds, batch_size, True)
    eval_loader = _loader(ds, batch_size, False)

    # --- Feature groups ---------------------------------------------------------
    groups: dict = joblib.load(feature_groups_pkl)
    cont_dim = len(groups["continuous"]) + len(groups["binary"])
    n_cat = len(groups["categorical"])

    # заглушка: считаем карт. каждой категории <=256
    cat_cards = [256] * n_cat

    # --- Model ------------------------------------------------------------------
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalFusionTransformer(
        seq_len=X.shape[1],
        n_cont_dim=cont_dim,
        n_cat=n_cat,
        cat_cardinalities=cat_cards,
        d_model=d_model,
        num_classes=3,
    ).to(device)

    # --- Loss & Optim -----------------------------------------------------------
    freqs = torch.load(class_freqs_pt)
    criterion = nn.CrossEntropyLoss(weight=_class_weights(freqs).to(device))
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    # --- Train loop -------------------------------------------------------------
    for ep in range(1, epochs + 1):
        model.train()
        loss_sum, preds, gold = 0.0, [], []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            cont = xb[..., :cont_dim]
            cat = xb[..., cont_dim:].long() if n_cat else None

            optim.zero_grad()
            logits, _ = model(cont, cat)
            loss = criterion(logits, yb + 1)  # {-1,0,1} → {0,1,2}
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            loss_sum += loss.item() * yb.size(0)
            preds.extend((logits.argmax(1).cpu() - 1).tolist())
            gold.extend(yb.cpu().tolist())

        f1 = f1_score(gold, preds, average="macro")
        print(f"[TFT] epoch {ep}/{epochs}  loss={loss_sum / len(ds):.4f}  f1_macro={f1:.3f}")

    torch.save(model.state_dict(), model_out)
    print(f"[TFT] saved model → {model_out}")

    # --- Embeddings -------------------------------------------------------------
    model.eval()
    emb_list = []
    with torch.no_grad():
        for xb, _ in eval_loader:
            xb = xb.to(device)
            cont = xb[..., :cont_dim]
            cat = xb[..., cont_dim:].long() if n_cat else None
            _, emb = model(cont, cat)
            emb_list.append(emb.cpu())
    emb_mat = torch.cat(emb_list).numpy()

    # индекс попытаться взять из существующего файла (если был)
    try:
        idx = pd.read_parquet(emb_out).index
        if len(idx) != len(emb_mat):
            raise ValueError
    except Exception:
        idx = pd.RangeIndex(len(emb_mat))

    pd.DataFrame(emb_mat, index=idx).to_parquet(emb_out)
    print(f"[TFT] embeddings → {emb_out}")

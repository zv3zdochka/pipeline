# gru_train.py

import warnings
import joblib
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from .models.GRU import MicroTrendGRU


def _compute_class_weights(freqs: dict) -> torch.Tensor:
    total = sum(freqs.values())
    # order совпадает с метками [-1, 0, +1]
    weights = torch.tensor([total / max(freqs.get(lbl, 1), 1) for lbl in (-1, 0, 1)], dtype=torch.float32)
    return weights / weights.mean()


def train_gru(
        dataset_pkl: str,
        class_freqs_pt: str,
        events_pkl: str,
        emb_path: str,
        model_out: str = "gru_model.pt",
        emb_out: str = "gru_embeddings.parquet",
        seq_len: int = 96,
        epochs: int = 10,
        batch_size: int = 128,
        lr: float = 3e-4,
        device: str | None = None
):
    # 1) загрузка готового датасета
    ds = joblib.load(dataset_pkl)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # 2) модель, loss и optimizer
    input_dim = ds.X.shape[1]
    model = MicroTrendGRU(input_dim=input_dim).to(device)
    freqs = torch.load(class_freqs_pt)  # {-1:…,0:…,1:…}
    weights = _compute_class_weights(freqs).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # 3) цикл обучения
    for ep in range(1, epochs + 1):
        model.train()
        running_loss, all_preds, all_trues = 0.0, [], []
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(X)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            all_preds += logits.argmax(dim=1).cpu().tolist()
            all_trues += y.cpu().tolist()

        acc = sum(p == t for p, t in zip(all_preds, all_trues)) / len(all_trues)
        f1 = f1_score(all_trues, all_preds, average="macro")
        print(
            f"[GRU] Epoch {ep}/{epochs} — loss: {running_loss / len(all_trues):.4f}, acc: {acc:.3f}, f1_macro: {f1:.3f}")

    # 4) сохраняем веса модели
    torch.save(model.state_dict(), model_out)
    print(f"[GRU] Веса сохранены в {model_out}")

    # 5) собираем эмбеддинги скрытого состояния
    model.eval()
    embs = []
    with torch.no_grad():
        eval_loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        for X, _ in eval_loader:
            X = X.to(device)
            _, h = model(X)
            embs.append(h.cpu())
    gru_emb = torch.cat(embs, dim=0).numpy()

    # восстанавливаем временные метки
    try:
        if emb_path.endswith(".parquet"):
            emb_df = pd.read_parquet(emb_path)
        else:
            emb_df = pd.read_csv(emb_path, index_col=0, parse_dates=True)
        ts_idx = emb_df.index[seq_len - 1:]
    except Exception as e:
        warnings.warn(f"Не удалось взять индекс из {emb_path}: {e}")
        ev = joblib.load(events_pkl).set_index("ts").sort_index()
        ts_idx = ev.index[seq_len - 1:]

    if len(ts_idx) > gru_emb.shape[0]:
        ts_idx = ts_idx[: gru_emb.shape[0]]
    elif len(ts_idx) < gru_emb.shape[0]:
        gru_emb = gru_emb[: len(ts_idx)]

    df_out = pd.DataFrame(gru_emb, index=ts_idx)

    try:
        df_out.to_parquet(emb_out)
    except:
        df_out.to_csv(emb_out.replace(".parquet", ".csv"))
    print(f"[GRU] Эмбеддинги сохранены в {emb_out}")

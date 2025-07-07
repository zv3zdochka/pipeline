from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------
# 1. ВСПОМОГАТЕЛЬНЫЕ БЛОКИ
# ---------------------------------------------------------------------
class GLU(nn.Module):
    """Gated Linear Unit (без смещения)."""

    def __init__(self, d: int):
        super().__init__()
        self.fc = nn.Linear(d, d * 2)

    def forward(self, x):
        y, g = self.fc(x).chunk(2, dim=-1)
        return y * torch.sigmoid(g)


class GRN(nn.Module):
    """Gated Residual Network (TFT §3.3)."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.gate = GLU(d_out)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x):
        y = self.elu(self.fc1(x))
        y = self.dropout(self.fc2(y))
        y = self.gate(y)
        return self.norm(self.skip(x) + y)


class VariableSelectionNetwork(nn.Module):
    """VSN: динамический выбор переменных для каждого таймстепа."""

    def __init__(self, n_vars: int, d_model: int, d_hidden: int):
        super().__init__()
        self.n_vars = n_vars
        # GRN, производящий alpha‑веса важности (B,L,n_vars)
        self.grn_weights = GRN(d_model, d_hidden, n_vars)
        #   и отдельный GRN для каждого канала (shared weights запрещены)
        self.var_grns = nn.ModuleList(
            [GRN(d_model, d_hidden, d_model) for _ in range(n_vars)]
        )

    def forward(self, x):  # x:(B,L,N,d)
        B, L, N, D = x.shape
        assert N == self.n_vars, (
            f"VSN: входных переменных {N}, ожидалось {self.n_vars}")
        # 1) важность переменных
        alpha = self.grn_weights(x.mean(dim=2))  # (B,L,N)
        alpha = torch.softmax(alpha, dim=-1).unsqueeze(-1)  # (B,L,N,1)
        # 2) трансформация переменных
        var_out = torch.stack(
            [grn(x[..., i, :]) for i, grn in enumerate(self.var_grns)], dim=2
        )  # (B,L,N,d)
        # 3) взвешенная сумма
        out = (alpha * var_out).sum(dim=2)        # (B,L,d)
        return out, alpha.squeeze(-1)


class InterpretableMultiHeadAttention(nn.Module):
    """IMHA = nn.MultiheadAttention + Residual + LayerNorm."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, mask=None):
        y, attn = self.mha(q, k, v, key_padding_mask=mask, need_weights=True)
        y = self.dropout(y)
        return self.norm(q + y), attn


# ---------------------------------------------------------------------
# 2.   FULL  TFT
# ---------------------------------------------------------------------
class TemporalFusionTransformer(nn.Module):
    def __init__(
        self,
        *,
        seq_len: int,
        n_cont_dim: int,               # количество колонок continuous+binary
        n_cat: int = 0,
        cat_cardinalities: list[int] | None = None,
        d_model: int = 128,
        d_hidden: int = 256,
        n_heads: int = 4,
        n_decoder_layers: int = 1,
        dropout: float = 0.1,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.n_cat = n_cat
        self.d_model = d_model

        # --- Continuous‑projection (вся матрица → d_model) ---
        self.cont_proj = nn.Linear(n_cont_dim, d_model)

        # --- Categorical embeddings ---
        if n_cat > 0:
            assert cat_cardinalities is not None and len(cat_cardinalities) == n_cat
            self.cat_embs = nn.ModuleList(
                [nn.Embedding(card, d_model) for card in cat_cardinalities]
            )
        else:
            self.cat_embs = None

        # === Variable Selection ===
        self.vsn = VariableSelectionNetwork(
            n_vars=1 + n_cat,          # 1 конт‑канал + n_cat категорий
            d_model=d_model,
            d_hidden=d_hidden,
        )

        # === Encoder / Decoder ===
        self.encoder = nn.LSTM(d_model, d_model, batch_first=True)
        self.decoder = nn.LSTM(d_model, d_model, num_layers=n_decoder_layers, batch_first=True)

        # === Attention ===
        self.attn = InterpretableMultiHeadAttention(d_model, n_heads, dropout)

        # === Final head ===
        self.head = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, num_classes),
        )

    # -------------------------------------------------------------
    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor | None = None, mask=None):
        """
        Parameters
        ----------
        x_cont : (B, L, n_cont_dim)
        x_cat  : (B, L, n_cat)   (может быть None, если n_cat==0)
        mask   : (B, L)          True = паддинг (optional)
        """
        # 1) Continuous → d_model
        cont_emb = self.cont_proj(x_cont)            # (B,L,D)
        vars_emb = [cont_emb]

        # 2) Categorical
        if self.n_cat > 0 and x_cat is not None:
            cat_embs = [emb(x_cat[..., i].long()) for i, emb in enumerate(self.cat_embs)]
            vars_emb.extend(cat_embs)               # len = 1 + n_cat

        # (B,L,N,D)
        vars_stacked = torch.stack(vars_emb, dim=2)

        # 3) Variable‑Selection
        v_selected, _ = self.vsn(vars_stacked)      # (B,L,D)

        # 4) Encoder / Decoder
        enc_out, _ = self.encoder(v_selected)       # (B,L,D)
        dec_out, _ = self.decoder(enc_out)          # (B,L,D)

        # 5) Temporal Attention
        att_out, _ = self.attn(dec_out, dec_out, dec_out, mask)

        # 6) Head (последний таймстеп)
        last = att_out[:, -1]                       # (B,D)
        logits = self.head(last)                    # (B,C)
        return logits, last                        # logits + embedding

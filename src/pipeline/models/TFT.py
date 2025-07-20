from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class GLU(nn.Module):
    """Gated Linear Unit (no bias)."""

    def __init__(self, d: int):
        super().__init__()
        self.fc = nn.Linear(d, d * 2)

    def forward(self, x):
        y, g = self.fc(x).chunk(2, dim=-1)
        return y * torch.sigmoid(g)


class GRN(nn.Module):
    """Gated Residual Network (Temporal Fusion Transformer §3.3)."""

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
    """Variable Selection Network: dynamic variable weighting per timestep."""

    def __init__(self, n_vars: int, d_model: int, d_hidden: int):
        super().__init__()
        self.n_vars = n_vars
        self.grn_weights = GRN(d_model, d_hidden, n_vars)
        self.var_grns = nn.ModuleList(
            [GRN(d_model, d_hidden, d_model) for _ in range(n_vars)]
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, L, N, D)
        Returns:
            out: (B, L, D) fused representation
            alpha: (B, L, N) variable importance weights
        """
        B, L, N, D = x.shape
        assert N == self.n_vars, f"VSN: got {N} variables, expected {self.n_vars}"
        alpha = self.grn_weights(x.mean(dim=2))
        alpha = torch.softmax(alpha, dim=-1).unsqueeze(-1)
        var_out = torch.stack(
            [grn(x[..., i, :]) for i, grn in enumerate(self.var_grns)], dim=2
        )
        out = (alpha * var_out).sum(dim=2)
        return out, alpha.squeeze(-1)


class InterpretableMultiHeadAttention(nn.Module):
    """MultiheadAttention with residual connection and LayerNorm (returns attention weights)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q, k, v: (B, L, D)
            mask: (B, L) boolean padding mask (True = pad) or None
        Returns:
            y: (B, L, D)
            attn: (B, n_heads, L, L) attention weights
        """
        y, attn = self.mha(q, k, v, key_padding_mask=mask, need_weights=True)
        y = self.dropout(y)
        return self.norm(q + y), attn


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer (simplified):
        - Continuous projection
        - Categorical embeddings
        - Variable Selection Network
        - LSTM encoder/decoder
        - Interpretable multi-head attention
        - Classification head (uses last timestep)
    """

    def __init__(
        self,
        *,
        seq_len: int,
        n_cont_dim: int,
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

        self.cont_proj = nn.Linear(n_cont_dim, d_model)

        if n_cat > 0:
            assert cat_cardinalities is not None and len(cat_cardinalities) == n_cat
            self.cat_embs = nn.ModuleList(
                [nn.Embedding(card, d_model) for card in cat_cardinalities]
            )
        else:
            self.cat_embs = None

        self.vsn = VariableSelectionNetwork(
            n_vars=1 + n_cat,
            d_model=d_model,
            d_hidden=d_hidden,
        )

        self.encoder = nn.LSTM(d_model, d_model, batch_first=True)
        self.decoder = nn.LSTM(d_model, d_model, num_layers=n_decoder_layers, batch_first=True)

        self.attn = InterpretableMultiHeadAttention(d_model, n_heads, dropout)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, num_classes),
        )

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor | None = None, mask=None):
        """
        Args:
            x_cont: (B, L, n_cont_dim)
            x_cat: (B, L, n_cat) or None if n_cat == 0
            mask: (B, L) boolean padding mask (True = pad) or None
        Returns:
            logits: (B, num_classes)
            embedding: (B, d_model) pooled last-timestep representation
        """
        cont_emb = self.cont_proj(x_cont)
        vars_emb = [cont_emb]

        if self.n_cat > 0 and x_cat is not None:
            cat_embs = [emb(x_cat[..., i].long()) for i, emb in enumerate(self.cat_embs)]
            vars_emb.extend(cat_embs)

        vars_stacked = torch.stack(vars_emb, dim=2)
        v_selected, _ = self.vsn(vars_stacked)
        enc_out, _ = self.encoder(v_selected)
        dec_out, _ = self.decoder(enc_out)
        att_out, _ = self.attn(dec_out, dec_out, dec_out, mask)
        last = att_out[:, -1]
        logits = self.head(last)
        return logits, last

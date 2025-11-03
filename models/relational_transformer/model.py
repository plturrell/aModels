"""Model components for the Relational Transformer."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import RELATIONAL_DTYPE_TO_ID


class MaskedMultiheadSelfAttention(nn.Module):
    """Multi-head self-attention with boolean masks."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq, hidden_dim).
            attn_mask: Boolean tensor of shape (batch, seq, seq) indicating
                permitted attention edges.
            padding_mask: Optional boolean tensor of shape (batch, seq) where
                True marks valid tokens.
        """
        batch, seq, _ = x.shape
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0).expand(batch, -1, -1)

        if padding_mask is not None:
            valid = padding_mask.unsqueeze(1) & padding_mask.unsqueeze(2)
            attn_mask = attn_mask & valid

        # Ensure self connections remain valid to avoid NaNs
        eye = torch.eye(seq, dtype=torch.bool, device=x.device).unsqueeze(0)
        attn_mask = attn_mask | eye

        qkv = self.qkv(x)
        qkv = qkv.reshape(batch, seq, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        mask = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = attn_weights.masked_fill(~mask, 0.0)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).reshape(batch, seq, self.hidden_dim)
        attn_output = self.out_proj(attn_output)
        return self.dropout(attn_output)


class RelationalBlock(nn.Module):
    """A single RT transformer block (Column → Feature → Neighbor → Temporal → Full → MLP)."""

    def __init__(self, hidden_dim: int, num_heads: int, mlp_hidden_dim: int, dropout: float):
        super().__init__()
        self.column_attn = MaskedMultiheadSelfAttention(hidden_dim, num_heads, dropout)
        self.feature_attn = MaskedMultiheadSelfAttention(hidden_dim, num_heads, dropout)
        self.neighbor_attn = MaskedMultiheadSelfAttention(hidden_dim, num_heads, dropout)
        self.temporal_attn = MaskedMultiheadSelfAttention(hidden_dim, num_heads, dropout)
        self.full_attn = MaskedMultiheadSelfAttention(hidden_dim, num_heads, dropout)

        self.norm_column = nn.LayerNorm(hidden_dim)
        self.norm_feature = nn.LayerNorm(hidden_dim)
        self.norm_neighbor = nn.LayerNorm(hidden_dim)
        self.norm_temporal = nn.LayerNorm(hidden_dim)
        self.norm_full = nn.LayerNorm(hidden_dim)
        self.norm_mlp = nn.LayerNorm(hidden_dim)

        self.mlp_fc1 = nn.Linear(hidden_dim, mlp_hidden_dim * 2)
        self.mlp_fc2 = nn.Linear(mlp_hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, masks: Dict[str, torch.Tensor], padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        residual = x
        y = self.column_attn(self.norm_column(x), masks["column"], padding_mask)
        x = residual + y

        residual = x
        y = self.feature_attn(self.norm_feature(x), masks["feature"], padding_mask)
        x = residual + y

        residual = x
        y = self.neighbor_attn(self.norm_neighbor(x), masks["neighbor"], padding_mask)
        x = residual + y

        residual = x
        temporal_mask = masks.get("temporal", masks["full"])
        y = self.temporal_attn(self.norm_temporal(x), temporal_mask, padding_mask)
        x = residual + y

        residual = x
        y = self.full_attn(self.norm_full(x), masks["full"], padding_mask)
        x = residual + y

        residual = x
        y = self.mlp_fc1(self.norm_mlp(x))
        gate, act = y.chunk(2, dim=-1)
        y = F.silu(gate) * act
        y = self.mlp_fc2(y)
        y = self.dropout(y)

        x = residual + y
        return x


class RelationalTransformer(nn.Module):
    """End-to-end Relational Transformer network."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 12,
        num_heads: int = 8,
        mlp_hidden_dim: int = 1024,
        value_dim: int = 384,
        schema_dim: int = 384,
        temporal_dim: int = 5,
        role_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.value_dim = value_dim

        self.schema_projector = nn.Linear(schema_dim, hidden_dim, bias=False)
        self.temporal_projector = nn.Linear(temporal_dim, hidden_dim, bias=False)
        self.role_projector = nn.Linear(role_dim, hidden_dim, bias=False)
        self.dtype_embeddings = nn.Embedding(len(RELATIONAL_DTYPE_TO_ID), hidden_dim)

        self.value_projectors = nn.ModuleDict(
            {
                dtype: nn.Linear(value_dim, hidden_dim, bias=False)
                for dtype in RELATIONAL_DTYPE_TO_ID.keys()
            }
        )

        self.variable_gate = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.Sigmoid(),
        )

        self.mask_embeddings = nn.Parameter(
            torch.randn(len(RELATIONAL_DTYPE_TO_ID), hidden_dim)
        )

        self.layers = nn.ModuleList(
            [
                RelationalBlock(hidden_dim, num_heads, mlp_hidden_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)

        self.output_heads = nn.ModuleDict(
            {
                "numeric": nn.Linear(hidden_dim, 1),
                "boolean": nn.Linear(hidden_dim, 1),
                "datetime": nn.Linear(hidden_dim, 1),
            }
        )

        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        value_embeddings: torch.Tensor,
        schema_embeddings: torch.Tensor,
        temporal_embeddings: Optional[torch.Tensor],
        role_embeddings: Optional[torch.Tensor],
        dtype_ids: torch.Tensor,
        masks: Dict[str, torch.Tensor],
        is_masked: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = value_embeddings.device
        dtype_ids = dtype_ids.to(device)
        batch, seq, _ = value_embeddings.shape

        schema_proj = self.schema_projector(schema_embeddings)
        token_embeddings = torch.zeros_like(schema_proj)
        temporal_proj = self.temporal_projector(temporal_embeddings.to(device)) if temporal_embeddings is not None else torch.zeros_like(schema_proj)
        role_proj = self.role_projector(role_embeddings.to(device)) if role_embeddings is not None else torch.zeros_like(schema_proj)
        dtype_embed = self.dtype_embeddings(dtype_ids)

        value_proj = torch.zeros_like(schema_proj)
        for dtype, dtype_id in RELATIONAL_DTYPE_TO_ID.items():
            projector = self.value_projectors[dtype]
            mask = dtype_ids == dtype_id
            if not mask.any():
                continue
            projected = projector(value_embeddings[mask])
            value_proj[mask] = projected

            if is_masked is not None:
                masked_positions = mask & is_masked
                if masked_positions.any():
                    value_proj[masked_positions] = self.mask_embeddings[dtype_id]

        combined = value_proj + schema_proj + temporal_proj + role_proj + dtype_embed
        variable_context = torch.cat([value_proj, schema_proj, temporal_proj, role_proj, dtype_embed], dim=-1)
        gate = self.variable_gate(variable_context)
        token_embeddings = combined * gate

        if padding_mask is not None:
            token_embeddings = token_embeddings * padding_mask.unsqueeze(-1)

        attn_masks = {name: mask.to(device) for name, mask in masks.items()}
        if 'temporal' not in attn_masks or attn_masks['temporal'] is None:
            attn_masks['temporal'] = attn_masks.get('full')

        for layer in self.layers:
            token_embeddings = layer(token_embeddings, attn_masks, padding_mask)

        token_embeddings = self.final_norm(token_embeddings)
        return token_embeddings

    def decode(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits: Dict[str, torch.Tensor] = {}
        for dtype, head in self.output_heads.items():
            logits[dtype] = head(hidden_states).squeeze(-1)
        return logits

    def link_logits(self, row_embeddings: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return torch.empty((0,), device=row_embeddings.device)
        src = row_embeddings[edge_index[:, 0]]
        dst = row_embeddings[edge_index[:, 1]]
        pair = torch.cat([src, dst], dim=-1)
        return self.link_predictor(pair).squeeze(-1)

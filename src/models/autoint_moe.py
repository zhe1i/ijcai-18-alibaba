from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiValueAttentionPooling(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.score_proj = nn.Linear(embed_dim, 1)

    def forward(
        self,
        token_emb: torch.Tensor,
        token_mask: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        # token_emb: [B, L, D], token_mask: [B, L], query: [B, D]
        q = self.query_proj(query).unsqueeze(1)
        h = torch.tanh(self.key_proj(token_emb) + q)
        score = self.score_proj(h).squeeze(-1)
        score = score.masked_fill(token_mask <= 0, -1e9)
        attn = torch.softmax(score, dim=1)
        pooled = torch.bmm(attn.unsqueeze(1), token_emb).squeeze(1)
        return pooled


class AutoIntLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.mha(x, x, x, need_weights=False)
        x = self.norm1(x + self.drop(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.drop(ffn_out))
        return x


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int], dropout: float, out_dim: int = 1) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)])
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AutoIntMoEModel(nn.Module):
    def __init__(
        self,
        cat_cardinalities: dict[str, int],
        multi_cardinalities: dict[str, int],
        num_dense: int,
        num_gate: int,
        embed_dim: int = 16,
        attn_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        shared_hidden: list[int] | None = None,
        num_experts: int = 3,
        expert_hidden: list[int] | None = None,
        use_moe: bool = True,
        use_multivalue: bool = True,
    ) -> None:
        super().__init__()
        if shared_hidden is None:
            shared_hidden = [256, 128]
        if expert_hidden is None:
            expert_hidden = [64]

        self.cat_fields = list(cat_cardinalities.keys())
        self.multi_fields = list(multi_cardinalities.keys())
        self.use_moe = use_moe
        self.use_multivalue = use_multivalue
        self.num_experts = num_experts

        self.single_embeddings = nn.ModuleDict(
            {
                f: nn.Embedding(cardinality, embed_dim, padding_idx=0)
                for f, cardinality in cat_cardinalities.items()
            }
        )

        self.multi_embeddings = nn.ModuleDict(
            {
                f: nn.Embedding(cardinality, embed_dim, padding_idx=0)
                for f, cardinality in multi_cardinalities.items()
            }
        )
        self.multi_pooling = nn.ModuleDict({f: MultiValueAttentionPooling(embed_dim) for f in self.multi_fields})

        self.dense_proj = nn.Linear(max(num_dense, 1), embed_dim)
        self.query_proj = nn.Linear(max(num_gate, 1), embed_dim)

        self.autoint_layers = nn.ModuleList(
            [AutoIntLayer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout) for _ in range(attn_layers)]
        )

        token_count = len(self.cat_fields) + 1
        if self.use_multivalue:
            token_count += len(self.multi_fields)

        self.flatten_dim = token_count * embed_dim
        shared_layers = [self.flatten_dim] + shared_hidden
        shared_blocks: list[nn.Module] = []
        for i in range(len(shared_layers) - 1):
            shared_blocks.extend(
                [
                    nn.Linear(shared_layers[i], shared_layers[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
        self.shared_net = nn.Sequential(*shared_blocks)
        self.shared_out_dim = shared_hidden[-1] if shared_hidden else self.flatten_dim

        if self.use_moe:
            self.experts = nn.ModuleList(
                [MLP(self.shared_out_dim, expert_hidden, dropout=dropout, out_dim=1) for _ in range(num_experts)]
            )
            self.gate = MLP(max(num_gate, 1), [64, 32], dropout=dropout, out_dim=num_experts)
        else:
            self.head = MLP(self.shared_out_dim, expert_hidden, dropout=dropout, out_dim=1)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        single_cat = batch["single_cat"]
        dense = batch["dense"]
        gate = batch["gate"]

        if dense.shape[1] == 0:
            dense_in = torch.zeros((dense.shape[0], 1), device=dense.device)
        else:
            dense_in = dense

        if gate.shape[1] == 0:
            gate_in = torch.zeros((gate.shape[0], 1), device=gate.device)
        else:
            gate_in = gate

        token_list = []
        for i, f in enumerate(self.cat_fields):
            emb = self.single_embeddings[f](single_cat[:, i])
            token_list.append(emb)

        query = self.query_proj(gate_in)

        if self.use_multivalue:
            for f in self.multi_fields:
                seq = batch[f]
                mask = batch[f"{f}__mask"]
                emb_seq = self.multi_embeddings[f](seq)
                pooled = self.multi_pooling[f](emb_seq, mask, query)
                token_list.append(pooled)

        dense_token = self.dense_proj(dense_in)
        token_list.append(dense_token)

        x = torch.stack(token_list, dim=1)
        for layer in self.autoint_layers:
            x = layer(x)

        x = x.reshape(x.shape[0], -1)
        shared = self.shared_net(x) if len(self.shared_net) > 0 else x

        extras: dict[str, torch.Tensor] = {}
        if self.use_moe:
            expert_logits = torch.cat([expert(shared) for expert in self.experts], dim=1)
            gate_logits = self.gate(gate_in)
            gate_weights = torch.softmax(gate_logits, dim=1)
            logit = (expert_logits * gate_weights).sum(dim=1, keepdim=True)
            extras["gate_weights"] = gate_weights
            extras["expert_logits"] = expert_logits
        else:
            logit = self.head(shared)

        return logit.squeeze(1), extras


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal = alpha_t * (1.0 - p_t).pow(self.gamma) * bce
        return focal.mean()



def load_balancing_loss(gate_weights: torch.Tensor) -> torch.Tensor:
    num_experts = gate_weights.shape[1]
    mean_w = gate_weights.mean(dim=0)
    target = torch.full_like(mean_w, 1.0 / num_experts)
    return torch.sum((mean_w - target) ** 2)



def sigmoid_np(x: Any) -> Any:
    return 1.0 / (1.0 + torch.exp(-x))

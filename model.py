import math
from collections import defaultdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        mean_square = torch.mean(x * x, dim=-1, keepdim=True)
        x = x * torch.rsqrt(mean_square + self.eps)
        return self.weight * x


class RotaryEmbedding(nn.Module):
    def __init__(
        self, dim: int, max_seq_len: int = 2048, device: Optional[torch.device] = None
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.device = device

        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, dim, 2, device=device).float() / dim)
        )
        t = torch.arange(max_seq_len, device=device).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos", emb.cos()[None, None, :, :])
        self.register_buffer("sin", emb.sin()[None, None, :, :])

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos[:, :, :seq_len, :]
        sin = self.sin[:, :, :seq_len, :]

        q_rot = (q * cos) + (self.rotate_half(q) * sin)
        k_rot = (k * cos) + (self.rotate_half(k) * sin)
        return q_rot, k_rot


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        num_key_value_heads: int,
        shared_weights: Optional["SharedWeights"] = None,
        lora_rank: int = 0,
        init_lora_weights: Optional[dict] = None,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = dim // n_heads

        self.wq: RecursiveLinear | nn.Linear
        self.wk: RecursiveLinear | nn.Linear
        self.wv: RecursiveLinear | nn.Linear
        self.wo: RecursiveLinear | nn.Linear

        # Get shared weights if provided, else create new ones
        if shared_weights is not None:
            self.wq = RecursiveLinear(
                dim,
                dim,
                bias=False,
                shared_weight=shared_weights.get_weight(
                    "self_attn", "q_proj", layer_idx
                ),
                lora_rank=lora_rank,
                init_lora_weights=init_lora_weights["q_proj"]
                if init_lora_weights
                else None,
            )
            self.wk = RecursiveLinear(
                dim,
                dim // 8,
                bias=False,
                shared_weight=shared_weights.get_weight(
                    "self_attn", "k_proj", layer_idx
                ),
                lora_rank=lora_rank,
                init_lora_weights=init_lora_weights["k_proj"]
                if init_lora_weights
                else None,
            )
            self.wv = RecursiveLinear(
                dim,
                dim // 8,
                bias=False,
                shared_weight=shared_weights.get_weight(
                    "self_attn", "v_proj", layer_idx
                ),
                lora_rank=lora_rank,
                init_lora_weights=init_lora_weights["v_proj"]
                if init_lora_weights
                else None,
            )
            self.wo = RecursiveLinear(
                dim,
                dim,
                bias=False,
                shared_weight=shared_weights.get_weight(
                    "self_attn", "o_proj", layer_idx
                ),
                lora_rank=lora_rank,
                init_lora_weights=init_lora_weights["o_proj"]
                if init_lora_weights
                else None,
            )
        else:
            self.wq = nn.Linear(dim, dim, bias=False)
            self.wk = nn.Linear(dim, dim // 8, bias=False)
            self.wv = nn.Linear(dim, dim // 8, bias=False)
            self.wo = nn.Linear(dim, dim, bias=False)

        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, T, C = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.num_key_value_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.num_key_value_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = self.rope(q, k, T)

        if self.num_key_value_heads != self.n_heads:
            k = k.repeat_interleave(self.n_heads // self.num_key_value_heads, dim=1)
            v = v.repeat_interleave(self.n_heads // self.num_key_value_heads, dim=1)

        if mask is not None:
            mask = mask.to(dtype=q.dtype)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
            )
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
            )

        out = out.transpose(1, 2).contiguous()
        out = out.view(B, T, C)

        return self.wo(out)


ACT2FN = {
    "gelu": F.gelu,
    "relu": F.relu,
    "tanh": torch.tanh,
    "silu": F.silu,
}


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        hidden_act: str,
        shared_weights: Optional["SharedWeights"] = None,
        init_lora_weights: Optional[dict] = None,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.gate_proj = RecursiveLinear(
            dim,
            hidden_dim,
            bias=False,
            shared_weight=shared_weights.get_weight("mlp", "gate_proj", layer_idx),
            init_lora_weights=init_lora_weights["gate_proj"]
            if init_lora_weights
            else None,
        )
        self.down_proj = RecursiveLinear(
            hidden_dim,
            dim,
            bias=False,
            shared_weight=shared_weights.get_weight("mlp", "down_proj", layer_idx),
            init_lora_weights=init_lora_weights["down_proj"]
            if init_lora_weights
            else None,
        )
        self.up_proj = RecursiveLinear(
            dim,
            hidden_dim,
            bias=False,
            shared_weight=shared_weights.get_weight("mlp", "up_proj", layer_idx),
            init_lora_weights=init_lora_weights["up_proj"]
            if init_lora_weights
            else None,
        )
        self.act = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class RecursiveTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        intermediate_size: int,
        hidden_act: str,
        num_key_value_heads: int,
        shared_weights: Optional["SharedWeights"] = None,
        lora_rank: int = 0,
        init_lora_weights: Optional[dict] = None,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.attention = Attention(
            dim,
            n_heads,
            num_key_value_heads=num_key_value_heads,
            shared_weights=shared_weights,
            lora_rank=lora_rank,
            init_lora_weights=init_lora_weights[layer_idx]["self_attn"]
            if init_lora_weights
            else None,
            layer_idx=layer_idx,
        )
        self.feed_forward = FeedForward(
            dim,
            intermediate_size,
            hidden_act,
            shared_weights=shared_weights,
            init_lora_weights=init_lora_weights[layer_idx]["mlp"]
            if init_lora_weights
            else None,
            layer_idx=layer_idx,
        )
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, layer_idx: int = 0
    ):
        x = x + self.attention(self.attention_norm(x), mask)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class LoRALayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int,
        init_lora_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.rank = rank

        if init_lora_weights is not None and rank > 0:
            self.B = nn.Parameter(init_lora_weights["B"])
            self.A = nn.Parameter(init_lora_weights["A"])
        else:
            self.B = nn.Parameter(torch.randn(out_dim, rank) / math.sqrt(rank))
            self.A = nn.Parameter(torch.randn(rank, in_dim) / math.sqrt(rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rank == 0:
            return torch.zeros_like(x)
        # First multiply A with x, then B with the result
        return F.linear(F.linear(x, self.A), self.B)

    def get_weight(self) -> torch.Tensor:
        return self.B @ self.A


class RecursiveLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        shared_weight: Optional[nn.Parameter] = None,
        lora_rank: int = 0,
        init_lora_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # W' - shared weights
        if shared_weight is None:
            self.weight = nn.Parameter(torch.empty((out_features, in_features)))
            # Initialize shared weights
            torch.nn.init.normal_(self.weight, mean=0.0, std=0.02)
        else:
            self.weight = shared_weight

        self.lora = LoRALayer(
            in_features,
            out_features,
            lora_rank,
            init_lora_weights if init_lora_weights else None,
        )

        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight + self.lora.get_weight())


class SharedWeights:
    """Manages shared weights between recursive blocks"""

    def __init__(
        self,
        b_layers: int,
        original_num_layers: int,
        backbone_weights: Optional[dict] = None,
        shapes: Optional[dict] = None,
    ):
        self.b_layers = b_layers
        self.original_num_layers = original_num_layers

        if backbone_weights is not None:
            self.shared_weights = []
            for layer in backbone_weights:
                block_p = defaultdict(dict)
                for block_type, projs in layer.items():
                    for proj_type, proj_weight in projs.items():
                        block_p[block_type][proj_type] = nn.Parameter(proj_weight)
                self.shared_weights.append(block_p)
        else:
            self.shared_weights = []
            for layer in range(b_layers):
                layer_p = []
                for block in shapes:
                    block_p = []
                    for proj_shape in block:
                        block_p.append(nn.Parameter(torch.zeros(proj_shape)))
                    layer_p.append(block_p)
                self.shared_weights.append(layer_p)

    def get_weight(self, block_type: str, name: str, idx_l: int) -> nn.Parameter:
        return self.shared_weights[idx_l % self.b_layers][block_type][name]


class RecursiveTinyLlama(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        n_layers: int,
        n_heads: int,
        intermediate_size: int,
        hidden_act: str,
        num_key_value_heads: int,
        num_blocks: int,  # B in the paper
        lora_rank: int = 4,  # Add LoRA rank parameter
        init_lora_weights: Optional[dict] = None,
        backbone_weights: Optional[dict] = None,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)

        # Initialize shared weights
        shared_weights = SharedWeights(
            b_layers=num_blocks,
            original_num_layers=n_layers,
            backbone_weights=backbone_weights,
            shapes={
                "self_attn": {
                    "q_proj": (dim, dim),
                    "k_proj": (dim // 8, dim),
                    "v_proj": (dim // 8, dim),
                    "o_proj": (dim, dim),
                },
                "gate_proj": (intermediate_size, dim),
                "up_proj": (intermediate_size, dim),
                "down_proj": (dim, intermediate_size),
            },
        )

        # Create B blocks with shared weights
        self.blocks = nn.ModuleList(
            [
                RecursiveTransformerBlock(
                    dim=dim,
                    n_heads=n_heads,
                    intermediate_size=intermediate_size,
                    hidden_act=hidden_act,
                    num_key_value_heads=num_key_value_heads,
                    shared_weights=shared_weights,
                    lora_rank=lora_rank,
                    init_lora_weights=init_lora_weights,
                    layer_idx=layer_idx,
                )
                for layer_idx in range(n_layers)
            ]
        )

        self.n_layers = n_layers
        self.num_blocks = num_blocks
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = self.token_embedding(tokens)

        for idx, block in enumerate(self.blocks):
            x = block(x, mask)

        x = self.norm(x)
        return self.lm_head(x)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.token_embedding

    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None:
        if (
            new_embeddings.embedding_dim != self.token_embedding.embedding_dim
            or new_embeddings.num_embeddings != self.token_embedding.num_embeddings
        ):
            raise ValueError(
                "new_embeddings shape does not match current token embedding"
            )
        self.token_embedding = new_embeddings

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_output: nn.Linear) -> None:
        if (
            new_output.in_features != self.lm_head.in_features
            or new_output.out_features != self.lm_head.out_features
        ):
            raise ValueError("new_output shape does not match current lm_head")
        self.lm_head = new_output

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(input_ids)

    def project_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            tokens_context = tokens[:, -2048:]
            # Create causal mask for the context
            seq_len = tokens_context.size(1)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=tokens.device), diagonal=1
            ).bool()
            mask = torch.zeros(1, 1, seq_len, seq_len, device=tokens.device)
            mask = mask.masked_fill(causal_mask, float("-inf"))

            # Forward pass with mask
            logits = self(tokens_context, mask=mask)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            tokens = torch.cat([tokens, next_token], dim=1)

        return tokens

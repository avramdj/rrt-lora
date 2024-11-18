from typing import Optional
import math
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


class RotaryEmbedding:
    def __init__(self, dim: int, max_seq_len: int = 2048):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos = emb.cos()[None, None, :, :]
        self.sin = emb.sin()[None, None, :, :]

    def rotate_half(self, x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def __call__(self, q: torch.Tensor, k: torch.Tensor, seq_len: int):
        cos = self.cos[:, :, :seq_len, :]
        sin = self.sin[:, :, :seq_len, :]
        q_rot = (q * cos) + (self.rotate_half(q) * sin)
        k_rot = (k * cos) + (self.rotate_half(k) * sin)
        return q_rot, k_rot


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, num_key_value_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = dim // n_heads
        
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, (dim // 8), bias=False)
        self.wv = nn.Linear(dim, (dim // 8), bias=False)
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

        if mask is None:
            causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
            mask = torch.zeros(B, 1, T, T, dtype=x.dtype, device=x.device)
            mask.masked_fill_(causal_mask, float('-inf'))

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) * scale
        
        scores = scores + mask
        
        attention = F.softmax(scores, dim=-1)
        out = (attention @ v).transpose(1, 2).contiguous()
        out = out.view(B, T, C)
        
        return self.wo(out)

ACT2FN = {
    "gelu": F.gelu,
    "relu": F.relu,
    "tanh": torch.tanh,
    "silu": F.silu,
}

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, hidden_act: str):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.act = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        intermediate_size: int,
        hidden_act: str,
        num_key_value_heads: int,
    ):
        super().__init__()
        self.attention = Attention(dim, n_heads, num_key_value_heads)
        self.feed_forward = FeedForward(dim, intermediate_size, hidden_act)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = x + self.attention(self.attention_norm(x), mask)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class TinyLlama(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        n_layers: int,
        n_heads: int,
        intermediate_size: int,
        hidden_act: str,
        num_key_value_heads: int,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(dim, n_heads, intermediate_size, hidden_act, num_key_value_heads)
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = self.token_embedding(tokens)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return self.lm_head(x)

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
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=tokens.device), diagonal=1).bool()
            mask = torch.zeros(1, 1, seq_len, seq_len, device=tokens.device)
            mask = mask.masked_fill(causal_mask, float('-inf'))
            
            # Forward pass with mask
            logits = self(tokens_context, mask=mask)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            tokens = torch.cat([tokens, next_token], dim=1)

        return tokens

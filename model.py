from typing import Optional, Tuple
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
    def __init__(self, dim: int, max_seq_len: int = 2048, device: Optional[torch.device] = None):
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.device = device
        
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Initialize on CPU, will move to correct device in __call__
        self.register_buffer = {
            'cos': emb.cos()[None, None, :, :],
            'sin': emb.sin()[None, None, :, :]
        }

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def __call__(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Move embeddings to the same device as input tensors
        device = q.device
        if device != self.device:
            self.device = device
            self.register_buffer = {
                key: tensor.to(device) 
                for key, tensor in self.register_buffer.items()
            }
        
        cos = self.register_buffer['cos'][:, :, :seq_len, :]
        sin = self.register_buffer['sin'][:, :, :seq_len, :]
        
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
        init_weights: Optional[dict] = None
    ):
        super().__init__()
        self.n_heads = n_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = dim // n_heads
        
        # Get shared weights if provided, else create new ones
        if shared_weights is not None:
            self.wq = RecursiveLinear(dim, dim, False, 
                shared_weights.get_weight('attn_q'),
                lora_rank, init_weights.get('q') if init_weights else None)
            kv_dim = self.num_key_value_heads * self.head_dim
            self.wk = RecursiveLinear(dim, kv_dim, False,
                shared_weights.get_weight('attn_k'),
                lora_rank, init_weights.get('k') if init_weights else None)
            self.wv = RecursiveLinear(dim, kv_dim, False,
                shared_weights.get_weight('attn_v'),
                lora_rank, init_weights.get('v') if init_weights else None)
            self.wo = RecursiveLinear(dim, dim, False,
                shared_weights.get_weight('attn_o'),
                lora_rank, init_weights.get('o') if init_weights else None)
        else:
            self.wq = nn.Linear(dim, dim, bias=False)
            kv_dim = self.num_key_value_heads * self.head_dim
            self.wk = nn.Linear(dim, kv_dim, bias=False)
            self.wv = nn.Linear(dim, kv_dim, bias=False)
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
    ):
        super().__init__()
        self.attention = Attention(dim, n_heads, num_key_value_heads, shared_weights, lora_rank)
        self.feed_forward = FeedForward(dim, intermediate_size, hidden_act)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = x + self.attention(self.attention_norm(x), mask)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class LoRALayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int, init_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.rank = rank
        
        if init_weights is not None and rank > 0:
            # Initialize using truncated SVD of the residual as per paper
            U, S, Vh = torch.linalg.svd(init_weights)
            # Take only top-r singular values/vectors
            self.B = nn.Parameter(U[:, :rank] @ torch.diag(S[:rank]))  # out_dim x rank
            self.A = nn.Parameter(Vh[:rank, :])  # rank x in_dim
        else:
            # Random initialization if no init weights provided
            self.B = nn.Parameter(torch.randn(out_dim, rank) / math.sqrt(rank))
            self.A = nn.Parameter(torch.randn(rank, in_dim) / math.sqrt(rank))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rank == 0:
            return torch.zeros_like(x)
        # First multiply A with x, then B with the result
        return F.linear(F.linear(x, self.A), self.B)

class RecursiveLinear(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool,
        shared_weight: Optional[nn.Parameter] = None,
        lora_rank: int = 0,
        init_weights: Optional[torch.Tensor] = None
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
            
        # Position-specific LoRA components
        self.lora = LoRALayer(in_features, out_features, lora_rank, init_weights)
        
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # W'x + BAx
        return F.linear(x, self.weight) + self.lora(x)

class SharedWeights:
    """Manages shared weights between recursive blocks"""
    def __init__(self, shapes: dict):
        self.shared_weights = {
            name: nn.Parameter(torch.empty(shape))
            for name, shape in shapes.items()
        }
        # Initialize all shared weights
        for weight in self.shared_weights.values():
            torch.nn.init.normal_(weight, mean=0.0, std=0.02)
    
    def get_weight(self, name: str) -> nn.Parameter:
        return self.shared_weights[name]

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
        num_blocks: int,
        lora_rank: int = 4,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        
        # Calculate key-value dimension
        head_dim = dim // n_heads
        kv_dim = num_key_value_heads * head_dim
        
        # Initialize shared weights with correct dimensions
        shared_weights = SharedWeights({
            'attn_q': (dim, dim),
            'attn_k': (kv_dim, dim),
            'attn_v': (kv_dim, dim),
            'attn_o': (dim, dim),
            'ffn_gate': (intermediate_size, dim),
            'ffn_up': (intermediate_size, dim),
            'ffn_down': (dim, intermediate_size),
        })
        
        # Create B blocks with shared weights
        self.blocks = nn.ModuleList([
            RecursiveTransformerBlock(
                dim=dim,
                n_heads=n_heads,
                intermediate_size=intermediate_size,
                hidden_act=hidden_act,
                num_key_value_heads=num_key_value_heads,
                shared_weights=shared_weights,
                lora_rank=lora_rank
            ) for _ in range(num_blocks)
        ])
        
        self.n_layers = n_layers
        self.num_blocks = num_blocks
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = self.token_embedding(tokens)
        
        # Apply layers recursively using blocks
        for l in range(self.n_layers):
            # Fix the block index calculation
            block_idx = (l % self.num_blocks)  # Simplified formula since we're 0-based
            x = self.blocks[block_idx](x, mask)
            
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

import torch
import torch.nn as nn

import math
from einops import rearrange, einsum
import einx

from .embedding import Linear, Embedding

class RMS(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.g = nn.Parameter(
            torch.ones(
                d_model,
                device=device,
                dtype=dtype
            ),
        )
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, sequence_length, d_model)
        in_dtype = x.dtype
        x = x.to(torch.float32)

        RMSNorm = torch.sqrt(
            torch.mean(x ** 2, dim=-1, keepdim=True)
            + self.eps
        )
        RMSNorm = torch.mul(x / RMSNorm, self.g)

        return RMSNorm.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, sequence_length, d_model)

        x1 = self.w1(x)
        x3 = self.w3(x)
        silu = x1 * torch.sigmoid(x1)

        return self.w2(silu * x3)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k

        positions = torch.arange(
            max_seq_len,
            device=device,
            dtype=torch.float32,
        )

        k = torch.arange(
            d_k // 2,
            device=device,
            dtype=torch.float32,
        )
        freq = theta ** (-2*k/d_k)

        angles = einx.dot(
            "max_seq_len, d -> max_seq_len d",
            positions, freq
        )

        self.register_buffer(
            "cos_cache",
            torch.cos(angles),
            persistent=False,
        )
        self.register_buffer(
            "sin_cache",
            torch.sin(angles),
            persistent=False,
        )


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x (Float[Tensor, "... sequence_length d_k"])  token_positions (Int[Tensor, "... sequence_length"])
        token_positions = token_positions.to(
            device=self.cos_cache.device
        )

        cos = self.cos_cache[token_positions].to(dtype=x.dtype)
        sin = self.sin_cache[token_positions].to(dtype=x.dtype)
        
        x_odd = x[..., 1::2]
        x_even = x[..., 0::2]

        x_out = torch.empty_like(x)

        x_out[..., 0::2] = x_even * cos - x_odd * sin
        x_out[..., 1::2] = x_even * sin + x_odd * cos
        return x_out


class Softmax(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x - torch.max(x, dim=self.dim, keepdim=True).values
        exp_x = torch.exp(x)
        return exp_x / torch.sum(exp_x, dim=self.dim, keepdim=True) 


class Attention(nn.Module):
    def __init__(self, mask=None):
        super().__init__()
        self.mask = mask

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        d_k = Q.size(-1)
        S = einx.dot(
            "... queries d_k, ... keys d_k -> ... queries keys",
            Q, K
        ) / math.sqrt(d_k)

        M = torch.zeros_like(S)
        if self.mask is not None:
            M = M.masked_fill(self.mask == False, float("-inf"))

        S += M
        A = Softmax(dim=-1)(S)

        return einx.dot(
            "... queries keys, ... keys d_v -> ... queries d_v",
            A, V
        )


class Multihead_Attention(nn.Module):
    def __init__(self, num_heads: int, d_model: int, device=None, dtype=None, rope: RotaryPositionalEmbedding | None = None,):
        super().__init__()

        self.num_heads = num_heads
        self.d_model = d_model
        self.d_head = d_model // num_heads

        self.proj = Linear(d_model, 3 * d_model, device, dtype) # Q K V proj
        self.o_proj = Linear(d_model, d_model, device, dtype)
        self.rope = rope

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None,) -> torch.Tensor:
        L = x.size(-2)
        QKV = self.proj(x)
        QKV = rearrange(QKV, 
            "... L (three h d_head) -> ... (three h) L d_head",
            h=self.num_heads, three=3
        )

        Q, K, V = torch.chunk(QKV, chunks=3, dim=-3)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(L, device=x.device)

            rope_positions = token_positions.unsqueeze(-2)

            Q = self.rope(Q, rope_positions)
            K = self.rope(K, rope_positions)
        
        mask = torch.tril(
            torch.ones(L, L, dtype=torch.bool, device=x.device)
        )

        result = rearrange(Attention(mask)(Q, K, V),
            "... h L d_head -> ... L (h d_head)"
        )

        return self.o_proj(result)


class transformer_block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()

        self.norm1 = RMS(d_model)
        self.norm2 = RMS(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
        self.rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len)
        self.attn = Multihead_Attention(num_heads, d_model, rope=self.rope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.attn(self.norm1(x))
        x = x + x1

        x2 = self.ffn(self.norm2(x))
        x = x + x2

        return x


class Transformer(nn.Module):
    def __init__(self, 
        vocab_size: int, context_length: int, d_model: int, num_layers: int,
        num_heads: int, d_ff: int, rope_theta: float,):
        super().__init__()

        self.token_embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.blocks = nn.ModuleList([
            transformer_block(d_model, num_heads, d_ff, context_length, rope_theta)
            for _ in range(num_layers)
        ])
        # self.block = transformer_block(d_model, num_heads, d_ff, context_length, rope_theta)
        self.norm = RMS(d_model)
        self.out_embedding = Linear(d_model, vocab_size)
        # self.softmax = Softmax(dim)(score)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(token_ids)
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = self.out_embedding(x)
        return x
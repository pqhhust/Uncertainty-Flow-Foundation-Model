"""DiT (Diffusion Transformer) backbone for flow matching.

Adapted from TransDiff/CIFAR/models/DiT.py. Uses adaLN-Zero conditioning
on timestep (and optionally interval width h for MeanFlow) for velocity field
prediction in the flow matching framework.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x, shift, scale):
    """Apply adaptive layer norm modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SwiGLU(nn.Module):
    """SwiGLU activation: element-wise product of SiLU-gated and linear projections.

    Replaces the (Linear + activation) pair in standard MLPs with two parallel
    linear projections — one passed through SiLU (gate) and one kept linear (up) —
    whose element-wise product forms the output.

    Reference: Shazeer, "GLU Variants Improve Transformer", 2020.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.w_gate = nn.Linear(in_features, out_features, bias=bias)
        self.w_up = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return F.silu(self.w_gate(x)) * self.w_up(x)


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations via sinusoidal encoding + MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            SwiGLU(frequency_embedding_size, hidden_size, bias=True),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class DiTBlock(nn.Module):
    """A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(2 * hidden_size * mlp_ratio / 3)
        self.mlp = nn.Sequential(
            SwiGLU(hidden_size, mlp_hidden_dim),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            SwiGLU(hidden_size, hidden_size),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        # Self-attention with adaLN
        normed = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + gate_msa.unsqueeze(1) * attn_out
        # MLP with adaLN
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class DiT(nn.Module):
    """DiT backbone: timestep-conditioned transformer blocks.

    Takes hidden states (B, S, D) and timestep (B,), returns transformed hidden states.
    Optionally accepts interval width h (B,) for MeanFlow-style mean velocity prediction.
    """

    def __init__(
        self,
        hidden_size: int = 896,
        depth: int = 1,
        num_heads: int = 14,
        mlp_ratio: float = 4.0,
        use_h_embedding: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.use_h_embedding = use_h_embedding
        self.t_embedder = TimestepEmbedder(hidden_size)
        if use_h_embedding:
            self.h_embedder = TimestepEmbedder(hidden_size)
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        # Initialize timestep embedding MLP (mlp[0] = SwiGLU, mlp[1] = Linear)
        for name in ('w_gate', 'w_up'):
            nn.init.normal_(getattr(self.t_embedder.mlp[0], name).weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[1].weight, std=0.02)
        # Initialize h embedding MLP (if present)
        if self.use_h_embedding:
            for name in ('w_gate', 'w_up'):
                nn.init.normal_(getattr(self.h_embedder.mlp[0], name).weight, std=0.02)
            nn.init.normal_(self.h_embedder.mlp[1].weight, std=0.02)
        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def forward(self, x, t, h=None):
        """
        Args:
            x: (B, S, D) hidden states
            t: (B,) timestep indices or continuous timesteps
            h: (B,) interval width for MeanFlow (optional, defaults to 0)
        Returns:
            (B, S, D) transformed hidden states
        """
        c = self.t_embedder(t)  # (B, D)
        if self.use_h_embedding:
            if h is None:
                h = torch.zeros_like(t)
            c = c + self.h_embedder(h)  # (B, D)
        for block in self.blocks:
            x = block(x, c)
        return x

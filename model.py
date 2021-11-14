import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce, repeat
from typing import Sequence, Tuple, Callable
from torch.nn import Module, ModuleList, Linear, LayerNorm, GroupNorm, Conv2d


class ConditionalNHWC(Module):
    """
    Conditional NHWC layer.
    """

    def __init__(self, out_features):
        super().__init__()
        inv_freq = 1. / torch.logspace(-5, 5, out_features//2)
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x, cond):
        freqs = torch.outer(cond, self.inv_freq) # num_channels = num_freqs/2
        posemb = repeat(freqs, "b c-> b (2 c)")
        odds, evens = rearrange(x, "... (j c) -> ... j c", j=2).unbind(dim=-2)
        rotated = torch.cat([evens, odds], dim=-1)
        return torch.einsum("b ... d, b ... d -> b ... d", x, posemb.cos()) + \
            torch.einsum("b ... d, b ... d -> b ... d", rotated, posemb.sin())


class SelfAttention(Module):
    """
    Self-attention layer.
    """

    def __init__(self, head_dim: int, heads: int):
        super().__init__()
        self.head_dim = head_dim
        self.heads = heads
        self.hidden_dim = self.head_dim * self.heads
        self.in_proj = Linear(hidden_dim, hidden_dim*3)
        self.out_proj = Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        b, h, w, d = x.shape
        x = rearrange(x, "b h w d -> b (h w) d")
        p = self.in_proj(x)
        q, k, v = torch.split(p, [
            self.head_dim, self.head_dim, self.head_dim],
            -1)
        (q, k, v) = map(lambda x: rearrange(x, "b i (h d) -> b i h d", h=self.heads), (q, k, v))
        a = torch.einsum("b i h d, b j h d -> b h i j", q, k) * (self.head_dim**-0.5)
        a = F.softmax(a, dim=-1)
        o = torch.einsum("b h i j, b j h d-> b i h d", a, v)
        o = rearrange(o, "b i h d -> b i (h d)")
        x = self.out_proj(o)
        x = rearrange(x, "b (h w) d -> b h w d", h=h, w=w)
        return x


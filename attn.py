import torch
from torch import nn, Tensor
import torch.nn.functional as F
from xformers.ops import memory_efficient_attention
from typing import Callable, Optional
from collections import OrderedDict
from torch.nn.init import xavier_uniform_, constant_
from einops import rearrange
from timm.layers import DropPath

class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float=1e-5, inplace:bool=False):
        super().__init__()
        self.inplace = inplace
        self.dim = dim
        self.init_values = init_values
        self.gamma = nn.Parameter(self.init_values * torch.ones(self.dim))

    def forward(self, x: Tensor):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class AttentionPooling(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_probe: int = 1,
        mlp_ratio: int = 4,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        assert self.embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.probe = nn.Parameter(torch.randn(1, num_probe, self.embed_dim))
        self.attn = nn.MultiheadAttention(self.embed_dim, self.num_heads, batch_first=True)

        self.layernorm = norm_layer(embed_dim)
        self.mlp_width = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(self.embed_dim, self.mlp_width)),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Linear(self.mlp_width, self.embed_dim)),
                ]
            )
        )

    def forward(self, x: Tensor):
        batch, _, _ = x.shape

        q = self.probe.repeat((batch, 1, 1)).to(x.dtype)
        x = self.attn(q, x, x, need_weights=False)[0]
        x = x + self.mlp(self.layernorm(x))

        return x


class SelfAttention(nn.Module):
    r"""
    Implements sequence packed attention and RoPe
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        rope: Optional[nn.Module] = None,
        qk_norm: bool = False,
    ):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        # To make this compatibile with nn.MultiHeadAttention
        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()

        self.rope = rope
        self.scale = self.head_dim ** (-0.5)

    def init_tensors(self):
        xavier_uniform_(self.in_proj_weight)
        constant_(self.in_proj_bias, 0.0)
        constant_(self.out_proj.bias, 0.0)

    def forward(self, x, attn_mask=None):
        batch, seq, embed_dim = x.shape
        proj = F.linear(x, self.in_proj_weight, self.in_proj_bias)

        # reshape to 3, E and not E, 3 is deliberate for better memory coalescing and keeping same order as chunk()
        proj = (
            proj.unflatten(-1, (3, embed_dim))
            .unsqueeze(0)
            .transpose(0, -2)
            .squeeze(-2)
            .contiguous()
        )
        q, k, v = proj[0], proj[1], proj[2]

        # Use "q_" so that we don't accidentally quit in pdb :)
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.rope:
            q, k = self.rope(q, k)

        attn = memory_efficient_attention(
            q, k, v, attn_bias=None, p=0.0, scale=self.scale,
        )
        attn = rearrange(attn, "b h s d -> b s (h d)")

        return F.linear(attn, self.out_proj.weight, self.out_proj.bias)


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        ls_init_value: Optional[float] = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        drop_path: float = 0.0,
        qk_norm: bool = True,
        rope: Optional[nn.Module] = None,
    ):
        super().__init__()

        if rope:
            self.attn = SelfAttention(embed_dim, n_head, rope=rope, qk_norm=qk_norm)
        else:
            self.attn = nn.MultiheadAttention(embed_dim, n_head, batch_first=True)

        self.ls_1 = LayerScale(embed_dim, ls_init_value) if ls_init_value else nn.Identity()
        self.ls_2 = LayerScale(embed_dim, ls_init_value) if ls_init_value else nn.Identity()

        self.ln_1 = norm_layer(embed_dim)
        self.ln_2 = norm_layer(embed_dim)

        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        mlp_width = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(embed_dim, mlp_width)),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Linear(mlp_width, embed_dim)),
                ]
            )
        )

    def _call_attn(
        self,
        q_x: Tensor,
        attn_mask: Optional[Tensor] = None,
    ):

        if attn_mask is not None:
            # Leave boolean masks as is
            if not attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.to(q_x.dtype)

        if isinstance(self.attn, SelfAttention):
            return self.attn(q_x, attn_mask=attn_mask)
        else:
            return self.attn(q_x, q_x, q_x, attn_mask=attn_mask, need_weights=False)[0]

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
    ):
        x = x + self.drop_path1(self.ls_1(self._call_attn(self.ln_1(x), attn_mask=attn_mask)))
        x = x + self.drop_path2(self.ls_2(self.mlp(self.ln_2(x))))
        return x

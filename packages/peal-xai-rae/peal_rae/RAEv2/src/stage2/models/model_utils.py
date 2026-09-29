import math
from math import pi
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all(len(set(t[1])) <= 2 for t in expandable_dims), "invalid dimensions for broadcastable concatenation"
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len=16,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for="lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
    ):
        super().__init__()
        if custom_freqs is not None:
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f"unknown modality {freqs_for}")

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len
        freqs = torch.einsum("..., f -> ... f", t, freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)
        self.register_buffer("freqs_cos", freqs.cos().view(-1, freqs.shape[-1]))
        self.register_buffer("freqs_sin", freqs.sin().view(-1, freqs.shape[-1]))

    def forward(self, t):
        _, _, seq_len, _ = t.shape
        base_len, _ = self.freqs_cos.shape
        repeat_factor = seq_len // base_len
        freqs_cos = self.freqs_cos
        freqs_sin = self.freqs_sin
        if repeat_factor != 1:
            freqs_cos = freqs_cos.repeat_interleave(repeat_factor, dim=0)
            freqs_sin = freqs_sin.repeat_interleave(repeat_factor, dim=0)
        return t * freqs_cos + rotate_half(t) * freqs_sin


class RoPE(nn.Module):
    def __init__(self, dim, vis_len, cond_len=0, theta=10000.0):
        super().__init__()
        d, T = dim // 2, int(vis_len**0.5)
        vis_freqs = 1.0 / (theta ** (torch.arange(0, d, 2).float() / d))
        vis_base_angles = torch.outer(torch.arange(T).float(), vis_freqs)
        vis_angles = torch.cat(
            [
                vis_base_angles[:, None].expand(-1, T, -1),
                vis_base_angles[None, :].expand(T, -1, -1),
            ],
            dim=-1,
        ).reshape(vis_len, d)
        cond_angles = torch.zeros(cond_len, dim // 2)
        angles = torch.cat([vis_angles, cond_angles], dim=0).repeat_interleave(2, dim=-1)
        self.register_buffer("freqs_cos", angles.cos())
        self.register_buffer("freqs_sin", angles.sin())

    def forward(self, t):
        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight


class NormAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        fused_attn: bool = True,
        use_rmsnorm: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        if use_rmsnorm:
            norm_layer = RMSNorm

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, rope=None, attn_mask=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if rope is not None:
            q = rope(q)
            k = rope(k)

        if self.fused_attn:
            q = q.to(v.dtype)
            k = k.to(v.dtype)
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    """Multi-head cross-attention backed by PyTorch SDPA.

    The query and context streams have the same width. Keeping this module
    separate from ``NormAttention`` avoids concatenating conditioning tokens
    with the spatial stream and makes the conditioning route explicit.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        use_rmsnorm: bool = True,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = int(num_heads)
        self.head_dim = dim // self.num_heads
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        norm_layer = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.out_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or context.ndim != 3:
            raise ValueError(
                f"cross-attention expects [B, N, D] tensors, got x={tuple(x.shape)}, "
                f"context={tuple(context.shape)}"
            )
        if x.shape[0] != context.shape[0] or x.shape[-1] != context.shape[-1]:
            raise ValueError(
                f"cross-attention shape mismatch: x={tuple(x.shape)}, context={tuple(context.shape)}"
            )

        batch_size, query_length, dim = x.shape
        context_length = context.shape[1]
        q = self.q_proj(x).reshape(
            batch_size, query_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        kv = self.kv_proj(context).reshape(
            batch_size, context_length, 2, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        attended = F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v)
        attended = attended.transpose(1, 2).reshape(batch_size, query_length, dim)
        return self.out_proj(attended)


class GaussianFourierEmbedding(nn.Module):
    def __init__(self, hidden_size, n_tokens: int = 1, embedding_size: int = 256, scale: float = 1.0):
        super().__init__()
        self.n_tokens = int(n_tokens)
        self.W = nn.Parameter(torch.normal(0, scale, (embedding_size,)), requires_grad=False)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size * 2, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.learnable_tokens = None
        if self.n_tokens > 1:
            self.learnable_tokens = nn.Parameter(
                torch.normal(0, 1 / hidden_size**0.5, (self.n_tokens, hidden_size))
            )

    def forward(self, t, return_base_embed: bool = False):
        t = t[:, None] * self.W[None, :] * 2 * torch.pi
        t_embed = torch.cat([torch.sin(t), torch.cos(t)], dim=-1)
        t_embed = self.mlp(t_embed)
        if self.n_tokens <= 1:
            return t_embed
        if return_base_embed:
            return t_embed.unsqueeze(1), self.learnable_tokens + t_embed.unsqueeze(1)
        return self.learnable_tokens + t_embed.unsqueeze(1)


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob, num_tokens: int = 1):
        super().__init__()
        self.num_classes = int(num_classes)
        self.dropout_prob = float(dropout_prob)
        self.num_tokens = int(num_tokens)
        self.token_dropout_prob = 0.5 * self.dropout_prob
        self.embedding_table = nn.Embedding(self.num_classes + 1, hidden_size * self.num_tokens)
        self.embedding_tables = [self.embedding_table]

    def forward(self, labels, train: bool, force_drop_ids=None):
        if labels.dtype != torch.long:
            labels = labels.long()

        batch_size = labels.shape[0]
        device = labels.device
        null_label = torch.full_like(labels, self.num_classes)

        drop_all = torch.zeros(batch_size, device=device, dtype=torch.bool)
        if train and self.dropout_prob > 0:
            drop_all |= torch.rand(batch_size, device=device) < self.dropout_prob
        if force_drop_ids is not None:
            drop_all |= force_drop_ids.to(device=device).bool()

        labels_all = torch.where(drop_all, null_label, labels)
        emb = self.embedding_table(labels_all)
        hidden_size = emb.shape[-1] // self.num_tokens
        emb = emb.view(batch_size, self.num_tokens, hidden_size)

        if train and self.num_tokens > 1 and self.token_dropout_prob > 0:
            token_drop = torch.rand(batch_size, self.num_tokens, device=device) < self.token_dropout_prob
            null_emb = self.embedding_table(null_label).view(batch_size, self.num_tokens, hidden_size)
            emb = torch.where(token_drop.unsqueeze(-1), null_emb, emb)

        return emb.squeeze(1) if self.num_tokens == 1 else emb


class ConditionEmbedder(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_classes=1000,
        context_dim=768,
        condition_type="label",
        n_tokens=8,
        latent_in_channels=768,
        latent_patch_size=1,
        n_action_tokens=4,
    ):
        super().__init__()
        self.condition_type = condition_type
        self.hidden_size = hidden_size

        if condition_type in {"label", "label_meta"}:
            self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
            self.learnable_tokens = nn.Parameter(torch.normal(0, 1 / hidden_size**0.5, (n_tokens, hidden_size)))
        elif condition_type == "text":
            self.norm = RMSNorm(context_dim)
            self.proj = nn.Linear(context_dim, hidden_size)
        elif condition_type == "nwm":
            self.context_patch_embed = nn.Conv2d(
                latent_in_channels,
                hidden_size,
                kernel_size=latent_patch_size,
                stride=latent_patch_size,
            )
            self.n_action_tokens = n_action_tokens
            self.action_proj = nn.Linear(3, hidden_size)
            self.action_tokens = nn.Parameter(
                torch.normal(0, 1 / hidden_size**0.5, (n_action_tokens, hidden_size))
            )
            self.time_emb = GaussianFourierEmbedding(hidden_size, n_tokens=1)
        else:
            raise ValueError(f"Unknown condition_type: {condition_type}")

    def forward(self, y) -> torch.Tensor:
        if self.condition_type == "nwm":
            ctx = y["context_latents"]
            B, K = ctx.shape[:2]
            patches = self.context_patch_embed(ctx.flatten(0, 1))
            patches = patches.flatten(2).transpose(1, 2)
            patches = patches.reshape(B, K * patches.shape[1], -1)
            act_tokens = self.action_tokens + self.action_proj(y["action"]).unsqueeze(1)
            time_token = self.time_emb(y["rel_time"].squeeze(-1)).unsqueeze(1)
            return torch.cat([patches, act_tokens, time_token], dim=1)
        if self.condition_type in {"label", "label_meta"}:
            return self.learnable_tokens + self.embedding_table(y).unsqueeze(1)
        return self.proj(self.norm(y))

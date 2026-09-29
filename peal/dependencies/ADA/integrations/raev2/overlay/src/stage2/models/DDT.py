# overlay for RAEv2 src/stage2/models/DDT.py
"""
DDT head model for stage-2.

- Provides DiTwDDTHead (the symbol imported by stage2/__init__.py).
- Includes optional MeDi-style meta conditioning:
    * meta_num_classes: list of vocab sizes per meta field
    * meta_dropout_prob: dropout applied to summed meta embedding
    * meta_fields: optional list of field names (enables dict meta input)

Meta behavior:
- If meta_num_classes is None/empty: meta is accepted but ignored (no crash).
- If meta is provided and meta_num_classes is set: meta embedding is added into the encoder conditioning.
- forward/forward_with_cfg/forward_with_autoguidance accept meta and **kwargs for trainer compatibility.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Mlp

from .model_utils import (
    CrossAttention,
    VisionRotaryEmbeddingFast,
    RMSNorm,
    SwiGLUFFN,
    GaussianFourierEmbedding,
    LabelEmbedder,
    NormAttention,
    get_2d_sincos_pos_embed,
)


def DDTModulate(x: torch.Tensor, shift: Optional[torch.Tensor], scale: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Per-segment modulation:
      x:     (B, Lx, D)
      shift: (B, L,  D) or None
      scale: (B, L,  D) or None

    Returns:
      x * (1 + scale) + shift  (with shift/scale repeated along length if needed).
    """
    if shift is None or scale is None:
        return x

    B, Lx, D = x.shape
    _, L, _ = shift.shape
    if Lx % L != 0:
        raise ValueError(f"Lx ({Lx}) must be divisible by L ({L})")
    rep = Lx // L
    if rep != 1:
        shift = shift.repeat_interleave(rep, dim=1)
        scale = scale.repeat_interleave(rep, dim=1)
    return x * (1 + scale) + shift


def DDTGate(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """
    Per-segment gating:
      x:    (B, Lx, D)
      gate: (B, L,  D)

    Returns:
      x * gate  (with gate repeated along length if needed).
    """
    B, Lx, D = x.shape
    _, L, _ = gate.shape
    if Lx % L != 0:
        raise ValueError(f"Lx ({Lx}) must be divisible by L ({L})")
    rep = Lx // L
    if rep != 1:
        gate = gate.repeat_interleave(rep, dim=1)
    return x * gate


def _normalize_modulation_bound(name: str, value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number when provided")
    return value


def _smooth_bound(value: torch.Tensor, max_abs: Optional[float]) -> torch.Tensor:
    """Smoothly constrain an AdaLN component without changing zero initialization."""
    if max_abs is None:
        return value
    return max_abs * torch.tanh(value / max_abs)


class LightningDDTBlock(nn.Module):
    """
    DDT block:
    - Attention + MLP with AdaLN modulation.
    - Uses DDTModulate/DDTGate to allow per-segment modulation.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_qknorm: bool = False,
        use_swiglu: bool = True,
        use_rmsnorm: bool = True,
        wo_shift: bool = False,
        use_cross_attention: bool = False,
        cross_attention_qk_norm: bool = True,
        adaln_shift_bound: Optional[float] = None,
        adaln_scale_bound: Optional[float] = None,
        adaln_gate_bound: Optional[float] = None,
        **block_kwargs,
    ):
        super().__init__()

        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)
        norm_layer = RMSNorm if use_rmsnorm else lambda dim: nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        self.attn = NormAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            **block_kwargs,
        )

        self.cross_attn = (
            CrossAttention(
                hidden_size,
                num_heads=num_heads,
                qk_norm=cross_attention_qk_norm,
                use_rmsnorm=use_rmsnorm,
            )
            if use_cross_attention
            else None
        )
        self.norm_cross = norm_layer(hidden_size) if self.cross_attn is not None else None
        self.cross_attn_gate = nn.Parameter(torch.zeros(hidden_size)) if self.cross_attn is not None else None

        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        def approx_gelu():
            return nn.GELU(approximate="tanh")

        if use_swiglu:
            self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0.0,
            )

        # AdaLN modulation
        if wo_shift:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 4 * hidden_size, bias=True))
        else:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.wo_shift = wo_shift
        self.adaln_shift_bound = _normalize_modulation_bound(
            "adaln_shift_bound", adaln_shift_bound
        )
        self.adaln_scale_bound = _normalize_modulation_bound(
            "adaln_scale_bound", adaln_scale_bound
        )
        self.adaln_gate_bound = _normalize_modulation_bound(
            "adaln_gate_bound", adaln_gate_bound
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        feat_rope=None,
        cross_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Ensure c broadcastable: (B, D) -> (B, 1, D)
        if c.ndim < x.ndim:
            c = c.unsqueeze(1)

        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=-1)
            shift_msa = None
            shift_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)

        shift_msa = _smooth_bound(shift_msa, self.adaln_shift_bound) if shift_msa is not None else None
        shift_mlp = _smooth_bound(shift_mlp, self.adaln_shift_bound) if shift_mlp is not None else None
        scale_msa = _smooth_bound(scale_msa, self.adaln_scale_bound)
        scale_mlp = _smooth_bound(scale_mlp, self.adaln_scale_bound)
        gate_msa = _smooth_bound(gate_msa, self.adaln_gate_bound)
        gate_mlp = _smooth_bound(gate_mlp, self.adaln_gate_bound)

        x = x + DDTGate(self.attn(DDTModulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope), gate_msa)
        if self.cross_attn is not None and cross_context is not None:
            cross_update = self.cross_attn(self.norm_cross(x), cross_context)
            cross_gate = torch.tanh(self.cross_attn_gate).view(1, 1, -1)
            x = x + cross_update * cross_gate
        x = x + DDTGate(self.mlp(DDTModulate(self.norm2(x), shift_mlp, scale_mlp)), gate_mlp)
        return x


class DDTFinalLayer(nn.Module):
    """Final projection layer for DDT."""

    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        use_rmsnorm: bool = False,
        adaln_shift_bound: Optional[float] = None,
        adaln_scale_bound: Optional[float] = None,
    ):
        super().__init__()
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm_final = RMSNorm(hidden_size)

        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))
        self.adaln_shift_bound = _normalize_modulation_bound(
            "adaln_shift_bound", adaln_shift_bound
        )
        self.adaln_scale_bound = _normalize_modulation_bound(
            "adaln_scale_bound", adaln_scale_bound
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if c.ndim < x.ndim:
            c = c.unsqueeze(1)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        shift = _smooth_bound(shift, self.adaln_shift_bound)
        scale = _smooth_bound(scale, self.adaln_scale_bound)
        x = DDTModulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DDTAuxFinalLayer(nn.Module):
    """Final projection layer for pooled auxiliary state."""

    def __init__(
        self,
        hidden_size: int,
        aux_dim: int,
        use_rmsnorm: bool = False,
        adaln_shift_bound: Optional[float] = None,
        adaln_scale_bound: Optional[float] = None,
    ):
        super().__init__()
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm_final = RMSNorm(hidden_size)

        self.linear = nn.Linear(hidden_size, aux_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))
        self.adaln_shift_bound = _normalize_modulation_bound(
            "adaln_shift_bound", adaln_shift_bound
        )
        self.adaln_scale_bound = _normalize_modulation_bound(
            "adaln_scale_bound", adaln_scale_bound
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if c.ndim < x.ndim:
            c = c.unsqueeze(1)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        shift = _smooth_bound(shift, self.adaln_shift_bound)
        scale = _smooth_bound(scale, self.adaln_scale_bound)
        x = DDTModulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x.squeeze(1)


class DDTAuxTokensFinalLayer(nn.Module):
    """Final projection layer for token auxiliary state."""

    def __init__(
        self,
        hidden_size: int,
        aux_token_dim: int,
        use_rmsnorm: bool = False,
        adaln_shift_bound: Optional[float] = None,
        adaln_scale_bound: Optional[float] = None,
    ):
        super().__init__()
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm_final = RMSNorm(hidden_size)

        self.linear = nn.Linear(hidden_size, aux_token_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))
        self.adaln_shift_bound = _normalize_modulation_bound(
            "adaln_shift_bound", adaln_shift_bound
        )
        self.adaln_scale_bound = _normalize_modulation_bound(
            "adaln_scale_bound", adaln_scale_bound
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if c.ndim < x.ndim:
            c = c.unsqueeze(1)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        shift = _smooth_bound(shift, self.adaln_shift_bound)
        scale = _smooth_bound(scale, self.adaln_scale_bound)
        x = DDTModulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiTwDDTHead(nn.Module):
    """
    Two-stage DiT-with-DDT head.

    Notes on inputs:
    - This model expects "x" to have channel dimension equal to x_channel_per_token
      (= in_channels * x_patch_size^2).
      That’s the original design: you feed a token-grid where each token carries an in_channels patch.

    Meta conditioning (MeDi-style):
    - Provide meta_num_classes=[...] to enable; otherwise meta is ignored but accepted.
    - meta can be:
        * Tensor(B, M) long
        * dict[field->Tensor(B,)] long if meta_fields provided
    """

    def __init__(
        self,
        input_size: int = 1,
        patch_size: Union[List[int], int] = 1,
        in_channels: int = 768,
        hidden_size: Sequence[int] = (1152, 2048),
        depth: Sequence[int] = (28, 2),
        num_heads: Union[Sequence[int], int] = (16, 16),
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        num_classes: int = 1000,
        use_qknorm: bool = False,
        use_swiglu: bool = True,
        use_rope: bool = True,
        use_rmsnorm: bool = True,
        wo_shift: bool = False,
        use_pos_embed: bool = True,
        condition_type: str = "label",
        context_dim: Optional[int] = None,
        cond_arch: Optional[Any] = None,
        enable_repa: bool = False,
        repa_layer_depth: int = 8,
        z_dim: Optional[int] = None,
        base_model_depth: Optional[int] = None,

        # --- MeDi / meta conditioning (optional) ---
        meta_num_classes: Optional[Sequence[int]] = None,  # e.g. [3, 3] for CelebA (0=null,1/2 bin)
        meta_dropout_prob: float = 0.0,
        meta_fields: Optional[Sequence[str]] = None,
        predict_aux: bool = False,
        aux_dim: Optional[int] = None,
        predict_aux_tokens: bool = False,
        aux_token_dim: Optional[int] = None,
        num_aux_tokens: Optional[int] = None,
        num_register_tokens: int = 0,
        cls_conditioning: bool = False,
        cls_dim: int = 1024,
        cls_projector_hidden_mult: int = 4,
        cls_condition_source: str = "encoder_x_norm_clstoken",
        cls_projector_max_norm: Optional[float] = None,
        cls_conditioning_mode: str = "adaln",
        cls_cross_attention_tokens: int = 8,
        cls_self_attention_tokens: int = 1,
        cls_cross_attention_every: int = 4,
        cls_cross_attention_qk_norm: bool = True,
        adaln_shift_bound: Optional[float] = None,
        adaln_scale_bound: Optional[float] = None,
        adaln_gate_bound: Optional[float] = None,
    ):
        super().__init__()

        self.in_channels = int(in_channels)
        self.out_channels = int(in_channels)
        self.predict_aux = bool(predict_aux)
        self.aux_dim = int(aux_dim) if aux_dim is not None else int(in_channels)
        self.predict_aux_tokens = bool(predict_aux_tokens)
        self.aux_token_dim = int(aux_token_dim) if aux_token_dim is not None else int(in_channels)
        self.num_aux_tokens = int(num_aux_tokens) if num_aux_tokens is not None else 0
        self.num_register_tokens = int(num_register_tokens)
        self.cls_conditioning = bool(cls_conditioning)
        self.cls_dim = int(cls_dim)
        self.cls_projector_hidden_mult = int(cls_projector_hidden_mult)
        self.cls_condition_source = str(cls_condition_source)
        self.cls_projector_max_norm = (
            float(cls_projector_max_norm)
            if cls_projector_max_norm is not None
            else None
        )
        self.cls_conditioning_mode = str(cls_conditioning_mode)
        self.cls_cross_attention_tokens = int(cls_cross_attention_tokens)
        self.cls_self_attention_tokens = int(cls_self_attention_tokens)
        self.cls_cross_attention_every = int(cls_cross_attention_every)
        self.cls_cross_attention_qk_norm = bool(cls_cross_attention_qk_norm)
        self.adaln_shift_bound = _normalize_modulation_bound(
            "adaln_shift_bound", adaln_shift_bound
        )
        self.adaln_scale_bound = _normalize_modulation_bound(
            "adaln_scale_bound", adaln_scale_bound
        )
        self.adaln_gate_bound = _normalize_modulation_bound(
            "adaln_gate_bound", adaln_gate_bound
        )
        self.condition_type = str(condition_type)
        self.context_dim = context_dim
        self.cond_arch = cond_arch
        self.enable_repa = bool(enable_repa)
        self.repa_layer_depth = int(repa_layer_depth)
        self.z_dim = int(z_dim) if z_dim is not None else None
        self.base_model_depth = base_model_depth

        if self.predict_aux and self.predict_aux_tokens:
            raise ValueError("predict_aux and predict_aux_tokens are mutually exclusive.")
        if self.predict_aux and self.aux_dim <= 0:
            raise ValueError("aux_dim must be positive when predict_aux=True")
        if self.predict_aux_tokens and self.num_aux_tokens <= 0:
            raise ValueError("num_aux_tokens must be positive when predict_aux_tokens=True")
        if self.predict_aux_tokens and self.aux_token_dim <= 0:
            raise ValueError("aux_token_dim must be positive when predict_aux_tokens=True")
        if self.num_register_tokens < 0:
            raise ValueError("num_register_tokens must be non-negative")
        if self.cls_conditioning and self.cls_dim <= 0:
            raise ValueError("cls_dim must be positive when cls_conditioning=True")
        if self.cls_projector_max_norm is not None and self.cls_projector_max_norm <= 0:
            raise ValueError("cls_projector_max_norm must be positive when provided")
        if self.cls_conditioning_mode not in {"adaln", "cross_attention", "self_attention"}:
            raise ValueError(
                "cls_conditioning_mode must be 'adaln', 'cross_attention', or "
                "'self_attention', "
                f"got {self.cls_conditioning_mode!r}"
            )
        if self.cls_cross_attention_tokens <= 0 and self.cls_conditioning_mode == "cross_attention":
            raise ValueError("CLS cross-attention requires at least one context token")
        if self.cls_self_attention_tokens <= 0 and self.cls_conditioning_mode == "self_attention":
            raise ValueError("CLS self-attention requires at least one prefix token")
        if self.cls_cross_attention_every <= 0:
            raise ValueError("cls_cross_attention_every must be positive")
        if self.condition_type not in {"label", "label_meta", "cls"}:
            raise ValueError(
                f"DiTwDDTHead currently supports condition_type in {{'label', 'label_meta', 'cls'}}, got {self.condition_type!r}"
            )
        if self.condition_type == "cls" and not self.cls_conditioning:
            raise ValueError("condition_type='cls' requires cls_conditioning=True")
        if self.enable_repa and self.z_dim is None:
            raise ValueError("z_dim must be provided when enable_repa=True")

        hidden_size = list(hidden_size)
        depth = list(depth)
        if len(hidden_size) != 2 or len(depth) != 2:
            raise ValueError("hidden_size and depth must be length-2 sequences: [encoder, decoder]")

        self.encoder_hidden_size = int(hidden_size[0])
        self.decoder_hidden_size = int(hidden_size[1])
        if self.cls_conditioning_mode == "cross_attention":
            self.cls_conditioning_tokens = self.cls_cross_attention_tokens
        elif self.cls_conditioning_mode == "self_attention":
            self.cls_conditioning_tokens = self.cls_self_attention_tokens
        else:
            self.cls_conditioning_tokens = 0

        if isinstance(num_heads, int):
            self.num_heads = [int(num_heads), int(num_heads)]
        else:
            nh = list(num_heads)
            if len(nh) != 2:
                raise ValueError("num_heads must be int or length-2 sequence: [enc, dec]")
            self.num_heads = [int(nh[0]), int(nh[1])]

        self.num_encoder_blocks = int(depth[0])
        self.num_decoder_blocks = int(depth[1])
        self.num_blocks = self.num_encoder_blocks + self.num_decoder_blocks

        # patch sizes: [s_patch_size, x_patch_size]
        if isinstance(patch_size, (int, float)):
            patch_size = [int(patch_size), int(patch_size)]
        patch_size = list(patch_size)
        if len(patch_size) != 2:
            raise ValueError(f"patch_size must be int or [s_patch_size, x_patch_size], got {patch_size}")
        self.s_patch_size = int(patch_size[0])
        self.x_patch_size = int(patch_size[1])

        self.s_channel_per_token = self.in_channels * self.s_patch_size * self.s_patch_size
        self.x_channel_per_token = self.in_channels * self.x_patch_size * self.x_patch_size

        # Embedders
        self.s_embedder = PatchEmbed(
            img_size=input_size,
            patch_size=self.s_patch_size,
            in_chans=self.s_channel_per_token,
            embed_dim=self.encoder_hidden_size,
            bias=True,
        )
        self.x_embedder = PatchEmbed(
            img_size=input_size,
            patch_size=self.x_patch_size,
            in_chans=self.x_channel_per_token,
            embed_dim=self.decoder_hidden_size,
            bias=True,
        )

        self.s_projector = (
            nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size)
            if self.encoder_hidden_size != self.decoder_hidden_size
            else nn.Identity()
        )

        self.t_embedder = GaussianFourierEmbedding(self.encoder_hidden_size)
        self.y_embedder = (
            None
            if self.condition_type == "cls"
            else LabelEmbedder(num_classes, self.encoder_hidden_size, class_dropout_prob)
        )
        self.repa_projector = nn.Linear(self.encoder_hidden_size, self.z_dim) if self.enable_repa else None
        if self.cls_conditioning:
            self.null_cls = nn.Parameter(torch.zeros(self.cls_dim))
            if self.cls_conditioning_mode == "adaln":
                cls_hidden = self.cls_projector_hidden_mult * self.encoder_hidden_size
                self.cls_embedder = nn.Sequential(
                    nn.LayerNorm(self.cls_dim, elementwise_affine=False, eps=1e-6),
                    nn.Linear(self.cls_dim, cls_hidden, bias=True),
                    nn.SiLU(),
                    nn.Linear(cls_hidden, self.encoder_hidden_size, bias=True),
                )
                self.cls_tokenizer = None
                self.cls_token_pos_embed = None
            else:
                self.cls_embedder = None
                self.cls_tokenizer = nn.Sequential(
                    nn.LayerNorm(self.cls_dim, elementwise_affine=False, eps=1e-6),
                    nn.Linear(
                        self.cls_dim,
                        self.cls_conditioning_tokens * self.encoder_hidden_size,
                        bias=True,
                    ),
                )
                self.cls_token_pos_embed = nn.Parameter(
                    torch.zeros(1, self.cls_conditioning_tokens, self.encoder_hidden_size)
                )
        else:
            self.null_cls = None
            self.cls_embedder = None
            self.cls_tokenizer = None
            self.cls_token_pos_embed = None

        # --- optional meta embedding ---
        self.meta_fields = list(meta_fields) if meta_fields is not None else None
        self.meta_num_classes = list(meta_num_classes) if meta_num_classes is not None else None
        if self.meta_num_classes is not None and len(self.meta_num_classes) > 0:
            self.meta_embedders = nn.ModuleList(
                [nn.Embedding(int(nc), self.encoder_hidden_size) for nc in self.meta_num_classes]
            )
            self.meta_dropout = nn.Dropout(float(meta_dropout_prob)) if meta_dropout_prob and meta_dropout_prob > 0 else nn.Identity()
        else:
            self.meta_embedders = None
            self.meta_dropout = None

        # final layer predicts per-token patch (patch_size=1 because embedder already tokenizes)
        self.final_layer = DDTFinalLayer(
            hidden_size=self.decoder_hidden_size,
            patch_size=1,
            out_channels=self.x_channel_per_token,
            use_rmsnorm=use_rmsnorm,
            adaln_shift_bound=self.adaln_shift_bound,
            adaln_scale_bound=self.adaln_scale_bound,
        )
        self.base_final_layer = None
        if self.predict_aux:
            self.aux_embedder = nn.Linear(self.aux_dim, self.encoder_hidden_size, bias=True)
            self.aux_pos_embed = nn.Parameter(torch.zeros(1, 1, self.encoder_hidden_size))
            self.aux_final_layer = DDTAuxFinalLayer(
                hidden_size=self.encoder_hidden_size,
                aux_dim=self.aux_dim,
                use_rmsnorm=use_rmsnorm,
                adaln_shift_bound=self.adaln_shift_bound,
                adaln_scale_bound=self.adaln_scale_bound,
            )
        else:
            self.aux_embedder = None
            self.aux_pos_embed = None
            self.aux_final_layer = None

        if self.predict_aux_tokens:
            self.aux_tokens_embedder = nn.Linear(self.aux_token_dim, self.encoder_hidden_size, bias=True)
            self.aux_tokens_pos_embed = nn.Parameter(torch.zeros(1, self.num_aux_tokens, self.encoder_hidden_size))
            self.aux_tokens_final_layer = DDTAuxTokensFinalLayer(
                hidden_size=self.encoder_hidden_size,
                aux_token_dim=self.aux_token_dim,
                use_rmsnorm=use_rmsnorm,
                adaln_shift_bound=self.adaln_shift_bound,
                adaln_scale_bound=self.adaln_scale_bound,
            )
        else:
            self.aux_tokens_embedder = None
            self.aux_tokens_pos_embed = None
            self.aux_tokens_final_layer = None

        if self.num_register_tokens > 0:
            # ViT-style learned compute tokens: participate in attention but are discarded before output.
            self.enc_register_tokens = nn.Parameter(torch.zeros(1, self.num_register_tokens, self.encoder_hidden_size))
            self.dec_register_tokens = nn.Parameter(torch.zeros(1, self.num_register_tokens, self.decoder_hidden_size))
        else:
            self.enc_register_tokens = None
            self.dec_register_tokens = None

        # Positional embeddings
        self.use_pos_embed = bool(use_pos_embed)
        if self.use_pos_embed:
            num_patches = self.s_embedder.num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.encoder_hidden_size), requires_grad=False)
            self.x_pos_embed = None  # optional; keep for backward-compat
        else:
            self.pos_embed = None
            self.x_pos_embed = None

        # RoPE
        self.use_rope = bool(use_rope)
        if self.use_rope:
            enc_half_head_dim = self.encoder_hidden_size // self.num_heads[0] // 2
            enc_hw = int(math.sqrt(self.s_embedder.num_patches))
            self.enc_feat_rope = VisionRotaryEmbeddingFast(dim=enc_half_head_dim, pt_seq_len=enc_hw)

            dec_half_head_dim = self.decoder_hidden_size // self.num_heads[1] // 2
            dec_hw = int(math.sqrt(self.x_embedder.num_patches))
            self.dec_feat_rope = VisionRotaryEmbeddingFast(dim=dec_half_head_dim, pt_seq_len=dec_hw)
        else:
            self.enc_feat_rope = None
            self.dec_feat_rope = None

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                LightningDDTBlock(
                    hidden_size=(self.encoder_hidden_size if i < self.num_encoder_blocks else self.decoder_hidden_size),
                    num_heads=(self.num_heads[0] if i < self.num_encoder_blocks else self.num_heads[1]),
                    mlp_ratio=mlp_ratio,
                    use_qknorm=use_qknorm,
                    use_rmsnorm=use_rmsnorm,
                    use_swiglu=use_swiglu,
                    wo_shift=wo_shift,
                    use_cross_attention=(
                        self.cls_conditioning
                        and self.cls_conditioning_mode == "cross_attention"
                        and i < self.num_encoder_blocks
                        and (i + 1) % self.cls_cross_attention_every == 0
                    ),
                    cross_attention_qk_norm=self.cls_cross_attention_qk_norm,
                    adaln_shift_bound=self.adaln_shift_bound,
                    adaln_scale_bound=self.adaln_scale_bound,
                    adaln_gate_bound=self.adaln_gate_bound,
                )
                for i in range(self.num_blocks)
            ]
        )

        self.initialize_weights()

    def _meta_embed(self, meta: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        meta -> (B, D_enc)

        Supported:
          - Tensor(B,M) long
          - dict[field -> Tensor(B,)] long  (requires meta_fields)
        """
        # meta accepted but ignored
        if self.meta_embedders is None:
            if isinstance(meta, dict):
                v0 = next(iter(meta.values()))
                return torch.zeros(v0.shape[0], self.encoder_hidden_size, device=v0.device, dtype=torch.float32)
            return torch.zeros(meta.shape[0], self.encoder_hidden_size, device=meta.device, dtype=torch.float32)

        if isinstance(meta, dict):
            if self.meta_fields is None:
                raise ValueError("meta is a dict but meta_fields was not provided in DiTwDDTHead.__init__")
            cols = []
            for k in self.meta_fields:
                if k not in meta:
                    raise KeyError(f"meta dict missing key '{k}' (meta_fields={self.meta_fields})")
                cols.append(meta[k].long())
            meta_t = torch.stack(cols, dim=1)  # (B,M)
        else:
            meta_t = meta.long() if meta.dtype != torch.long else meta

        if meta_t.ndim != 2:
            raise ValueError(f"meta must be (B,M), got {tuple(meta_t.shape)}")
        if meta_t.shape[1] != len(self.meta_embedders):
            raise ValueError(f"meta has M={meta_t.shape[1]} but meta_num_classes has {len(self.meta_embedders)}")

        embs = [emb(meta_t[:, i]) for i, emb in enumerate(self.meta_embedders)]  # list (B,D)
        out = torch.stack(embs, dim=0).sum(dim=0)  # (B,D)
        out = self.meta_dropout(out) if self.meta_dropout is not None else out
        return out

    def initialize_weights(self, xavier_uniform_init: bool = False):
        if xavier_uniform_init:
            def _basic_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            self.apply(_basic_init)

        # PatchEmbed (Conv2d) like Linear init
        for pe in [self.x_embedder, self.s_embedder]:
            w = pe.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            if pe.proj.bias is not None:
                nn.init.constant_(pe.proj.bias, 0)

        # label embedding init (support both styles); absent for CLS-only runs.
        if self.y_embedder is not None:
            if hasattr(self.y_embedder, "embedding_tables"):
                for emb in self.y_embedder.embedding_tables:
                    nn.init.normal_(emb.weight, std=0.02)
            else:
                nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        if self.predict_aux:
            nn.init.normal_(self.aux_embedder.weight, std=0.02)
            nn.init.constant_(self.aux_embedder.bias, 0)
            nn.init.normal_(self.aux_pos_embed, std=0.02)

        if self.predict_aux_tokens:
            nn.init.normal_(self.aux_tokens_embedder.weight, std=0.02)
            nn.init.constant_(self.aux_tokens_embedder.bias, 0)
            nn.init.normal_(self.aux_tokens_pos_embed, std=0.02)

        if self.num_register_tokens > 0:
            nn.init.normal_(self.enc_register_tokens, std=0.02)
            nn.init.normal_(self.dec_register_tokens, std=0.02)

        # meta embedding init (optional)
        if self.meta_embedders is not None:
            for emb in self.meta_embedders:
                nn.init.normal_(emb.weight, std=0.02)

        if self.cls_embedder is not None:
            nn.init.normal_(self.cls_embedder[1].weight, std=0.02)
            nn.init.constant_(self.cls_embedder[1].bias, 0)
            nn.init.constant_(self.cls_embedder[3].weight, 0)
            nn.init.constant_(self.cls_embedder[3].bias, 0)
        if self.cls_tokenizer is not None:
            nn.init.normal_(self.cls_tokenizer[1].weight, std=0.02)
            nn.init.constant_(self.cls_tokenizer[1].bias, 0)
            nn.init.normal_(self.cls_token_pos_embed, std=0.02)

        # fixed sin-cos pos embed
        if self.use_pos_embed and self.pos_embed is not None:
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.s_embedder.num_patches ** 0.5))
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # zero AdaLN mods
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # zero output
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        if self.predict_aux:
            nn.init.constant_(self.aux_final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.aux_final_layer.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.aux_final_layer.linear.weight, 0)
            nn.init.constant_(self.aux_final_layer.linear.bias, 0)

        if self.predict_aux_tokens:
            nn.init.constant_(self.aux_tokens_final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.aux_tokens_final_layer.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.aux_tokens_final_layer.linear.weight, 0)
            nn.init.constant_(self.aux_tokens_final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:    (B, T, patch_size**2 * C_token) where patch_size is 1 here
        imgs: (B, C_token, H, W)  (C_token = x_channel_per_token)
        """
        c = self.x_channel_per_token
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        if h * w != x.shape[1]:
            raise ValueError("Token count is not a square; cannot unpatchify.")
        x = x.reshape((x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape((x.shape[0], c, h * p, h * p))
        return imgs

    @staticmethod
    def _is_interval_list(cfg_interval) -> bool:
        return isinstance(cfg_interval, (list, tuple)) and len(cfg_interval) > 0 and isinstance(cfg_interval[0], (list, tuple))

    @staticmethod
    def _interval_mask(t: torch.Tensor, cfg_interval, default_all: bool = False) -> torch.Tensor:
        if cfg_interval is None:
            return torch.ones_like(t, dtype=torch.bool) if default_all else torch.zeros_like(t, dtype=torch.bool)

        if DiTwDDTHead._is_interval_list(cfg_interval):
            mask = torch.zeros_like(t, dtype=torch.bool)
            for (a, b) in cfg_interval:
                mask |= (t >= float(a)) & (t <= float(b))
            return mask

        if isinstance(cfg_interval, (list, tuple)) and len(cfg_interval) == 2:
            a, b = float(cfg_interval[0]), float(cfg_interval[1])
            return (t >= a) & (t <= b)

        return torch.ones_like(t, dtype=torch.bool) if default_all else torch.zeros_like(t, dtype=torch.bool)

    def _wrap_rope_with_prefix(self, rope_module, num_prefix_tokens: int):
        if rope_module is None:
            return None
        if num_prefix_tokens == 0:
            return rope_module

        def _rope(x):
            prefix = x[:, :, :num_prefix_tokens, :]
            tokens = x[:, :, num_prefix_tokens:, :]
            if tokens.shape[2] == 0:
                return x
            tokens = rope_module(tokens)
            return torch.cat([prefix, tokens], dim=2)

        return _rope

    def _split_outputs(self, model_out):
        if self.predict_aux or self.predict_aux_tokens:
            if not isinstance(model_out, (tuple, list)) or len(model_out) != 2:
                raise ValueError("Expected model output to be (x_out, aux_out) when auxiliary prediction is enabled")
            return model_out[0], model_out[1]
        return model_out, None

    @staticmethod
    def _mask_like(mask: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return mask.view(-1, *([1] * (ref.ndim - 1)))

    def _extract_inputs(self, x, aux: Optional[torch.Tensor]):
        if isinstance(x, (tuple, list)):
            if len(x) != 2:
                raise ValueError(f"Expected tuple/list input of length 2, got {len(x)}")
            x, state_aux = x
            if aux is None:
                aux = state_aux
        return x, aux

    def _coerce_cls(
        self, cls: Optional[torch.Tensor], batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if cls is None:
            cls = self.null_cls.expand(batch_size, -1)
        elif cls.ndim == 3:
            if cls.shape[1] < 1:
                raise ValueError(f"cls has no tokens: shape={tuple(cls.shape)}")
            cls = cls[:, 0]
        elif cls.ndim != 2:
            raise ValueError(f"cls must be [B, C] or [B, T, C], got {tuple(cls.shape)}")
        if cls.shape[0] != batch_size:
            raise ValueError(f"cls batch size {cls.shape[0]} does not match x batch size {batch_size}")
        if cls.shape[-1] != self.cls_dim:
            raise ValueError(f"cls dim {cls.shape[-1]} does not match configured cls_dim={self.cls_dim}")
        return cls.to(device=device, dtype=dtype)

    def _cls_embed(self, cls: Optional[torch.Tensor], batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.cls_embedder is None:
            return torch.zeros(batch_size, self.encoder_hidden_size, device=device, dtype=dtype)
        cls = self._coerce_cls(cls, batch_size, device, dtype)
        embedded = self.cls_embedder(cls)
        if self.cls_projector_max_norm is not None:
            output_norm = embedded.float().norm(dim=-1, keepdim=True)
            scale = (self.cls_projector_max_norm / output_norm.clamp_min(1.0e-6)).clamp(max=1.0)
            embedded = embedded * scale.to(dtype=embedded.dtype)
        return embedded

    def _cls_tokens(
        self, cls: Optional[torch.Tensor], batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        if self.cls_tokenizer is None:
            return None
        cls = self._coerce_cls(cls, batch_size, device, dtype)
        tokens = self.cls_tokenizer(cls).reshape(
            batch_size, self.cls_conditioning_tokens, self.encoder_hidden_size
        )
        tokens = tokens + self.cls_token_pos_embed.to(device=device, dtype=dtype)
        return F.layer_norm(tokens.float(), (self.encoder_hidden_size,), eps=1.0e-6).to(dtype=dtype)

    def cls_geometry_representation(
        self, cls: torch.Tensor, *, apply_bound: bool = True
    ) -> torch.Tensor:
        dtype = next(self.parameters()).dtype
        if self.cls_embedder is not None:
            if apply_bound:
                return self._cls_embed(cls, cls.shape[0], cls.device, dtype)
            cls = self._coerce_cls(cls, cls.shape[0], cls.device, dtype)
            return self.cls_embedder(cls)
        tokens = self._cls_tokens(cls, cls.shape[0], cls.device, dtype)
        if tokens is None:
            raise RuntimeError("CLS geometry requested for a model without CLS conditioning")
        return tokens.flatten(1) / math.sqrt(float(self.cls_conditioning_tokens))

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        s: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        meta: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        cls: Optional[torch.Tensor] = None,
        aux: Optional[torch.Tensor] = None,
        return_intermediate: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        x:    (B, x_channel_per_token, H, W) or tuple((B, x_channel_per_token, H, W), aux_state)
        t:    (B,) float or tensor
        y:    (B,) long
        meta: optional (B,M) long or dict
        """
        if y is None:
            y = kwargs.pop("context", None)
        if y is None:
            raise ValueError("DiTwDDTHead.forward requires y or context.")
        if s is None:
            s = kwargs.pop("s", kwargs.pop("encoder_state", None))
        x, aux = self._extract_inputs(x, aux)
        if cls is None:
            cls = kwargs.pop("cls_condition", None)

        # Timestep always uses AdaLN. CLS uses additive AdaLN, separate
        # cross-attention, or a prefix token in encoder self-attention.
        t_emb = self.t_embedder(t)                   # (B, D_enc)
        cross_context = None
        self_attention_prefix = None
        if self.condition_type == "cls":
            if self.cls_conditioning_mode == "cross_attention":
                c = t_emb
                cross_context = self._cls_tokens(y, x.shape[0], x.device, t_emb.dtype)
            elif self.cls_conditioning_mode == "self_attention":
                c = t_emb
                self_attention_prefix = self._cls_tokens(
                    y, x.shape[0], x.device, t_emb.dtype
                )
            else:
                c = t_emb + self._cls_embed(y, x.shape[0], x.device, t_emb.dtype)
        else:
            y_emb = self.y_embedder(y, self.training)    # (B, D_enc)
            c = t_emb + y_emb
            if self.cls_conditioning_mode == "cross_attention":
                cross_context = self._cls_tokens(cls, x.shape[0], x.device, c.dtype)
            elif self.cls_conditioning_mode == "self_attention":
                self_attention_prefix = self._cls_tokens(
                    cls, x.shape[0], x.device, c.dtype
                )
            else:
                c = c + self._cls_embed(cls, x.shape[0], x.device, c.dtype)


        if self.meta_embedders is not None:
            if meta is None:
                B = x.shape[0]
                M = len(self.meta_embedders)
                meta = torch.zeros((B, M), device=x.device, dtype=torch.long)
            c = c + self._meta_embed(meta).to(dtype=c.dtype)

        c = F.silu(c)
        aux_out = None
        zt_intermediate = None
        base_state = None

        if s is None:
            # Encode s from x (original behavior)
            s = self.s_embedder(x)  # (B, Ls, D_enc)
            if self.use_pos_embed and self.pos_embed is not None:
                s = s + self.pos_embed

            aux_prefix_len = 0
            cls_prefix_len = 0

            if self.predict_aux:
                if aux is None:
                    aux = torch.zeros((x.shape[0], self.aux_dim), device=x.device, dtype=x.dtype)
                if aux.ndim != 2:
                    raise ValueError(f"aux must be (B, aux_dim), got {tuple(aux.shape)}")
                if aux.shape[1] != self.aux_dim:
                    raise ValueError(f"aux has dim={aux.shape[1]} but expected aux_dim={self.aux_dim}")

                aux_prefix = self.aux_embedder(aux.to(dtype=s.dtype)).unsqueeze(1) + self.aux_pos_embed.to(dtype=s.dtype)
                s = torch.cat([aux_prefix, s], dim=1)
                aux_prefix_len = 1
            elif self.predict_aux_tokens:
                if aux is None:
                    aux = torch.zeros(
                        (x.shape[0], self.num_aux_tokens, self.aux_token_dim),
                        device=x.device,
                        dtype=x.dtype,
                    )
                if aux.ndim != 3:
                    raise ValueError(f"aux tokens must be (B, K, aux_token_dim), got {tuple(aux.shape)}")
                if aux.shape[1] != self.num_aux_tokens:
                    raise ValueError(f"aux has K={aux.shape[1]} but expected num_aux_tokens={self.num_aux_tokens}")
                if aux.shape[2] != self.aux_token_dim:
                    raise ValueError(f"aux has dim={aux.shape[2]} but expected aux_token_dim={self.aux_token_dim}")

                aux_prefix = self.aux_tokens_embedder(aux.to(dtype=s.dtype)) + self.aux_tokens_pos_embed.to(dtype=s.dtype)
                s = torch.cat([aux_prefix, s], dim=1)
                aux_prefix_len = self.num_aux_tokens

            if self_attention_prefix is not None:
                cls_prefix = self_attention_prefix.to(dtype=s.dtype)
                s = torch.cat(
                    [s[:, :aux_prefix_len, :], cls_prefix, s[:, aux_prefix_len:, :]],
                    dim=1,
                )
                cls_prefix_len = cls_prefix.shape[1]

            encoder_prefix_len = aux_prefix_len + cls_prefix_len
            if self.num_register_tokens > 0:
                enc_registers = self.enc_register_tokens.expand(x.shape[0], -1, -1).to(dtype=s.dtype)
                s = torch.cat(
                    [s[:, :encoder_prefix_len, :], enc_registers, s[:, encoder_prefix_len:, :]],
                    dim=1,
                )
            enc_feat_rope = self._wrap_rope_with_prefix(
                self.enc_feat_rope,
                encoder_prefix_len + self.num_register_tokens,
            )

            for i in range(self.num_encoder_blocks):
                s = self.blocks[i](s, c, feat_rope=enc_feat_rope, cross_context=cross_context)
                if self.base_final_layer is not None and self.base_model_depth is not None and (i + 1) == int(self.base_model_depth):
                    base_state = s
                if return_intermediate and self.repa_projector is not None and (i + 1) == self.repa_layer_depth:
                    patch_start = encoder_prefix_len + self.num_register_tokens
                    zt_intermediate = self.repa_projector(s[:, patch_start:, :])

            # Broadcast timestep embedding to tokens and gate
            t_tok = t_emb.unsqueeze(1).repeat(1, s.shape[1], 1)
            s = F.silu(t_tok + s)
            if base_state is not None:
                base_t_tok = t_emb.unsqueeze(1).repeat(1, base_state.shape[1], 1)
                base_state = F.silu(base_t_tok + base_state)

            if self.predict_aux:
                s_aux = s[:, :1, :]
                s = s[:, 1:, :]
                if base_state is not None:
                    base_state = base_state[:, 1:, :]
                aux_out = self.aux_final_layer(s_aux, c)
            elif self.predict_aux_tokens:
                s_aux = s[:, : self.num_aux_tokens, :]
                s = s[:, self.num_aux_tokens :, :]
                if base_state is not None:
                    base_state = base_state[:, self.num_aux_tokens :, :]
                aux_out = self.aux_tokens_final_layer(s_aux, c)

            if cls_prefix_len > 0:
                s = s[:, cls_prefix_len:, :]
                if base_state is not None:
                    base_state = base_state[:, cls_prefix_len:, :]

            if return_intermediate and self.repa_projector is not None and zt_intermediate is None:
                zt_intermediate = self.repa_projector(s)

        elif self_attention_prefix is not None:
            raise ValueError(
                "External encoder state s=... is not supported with CLS "
                "self-attention prefix conditioning."
            )
        elif self.predict_aux or self.predict_aux_tokens:
            raise ValueError("External encoder state s=... is not supported when auxiliary prediction is enabled.")

        # project encoder->decoder hidden
        s = self.s_projector(s)  # (B, Ls, D_dec)
        base_s = self.s_projector(base_state) if base_state is not None else None

        # Decode on token grid from x
        x_tok = self.x_embedder(x)  # (B, Lx, D_dec)
        if self.use_pos_embed and self.x_pos_embed is not None:
            x_tok = x_tok + self.x_pos_embed

        dec_feat_rope = self.dec_feat_rope
        if self.num_register_tokens > 0:
            dec_registers = self.dec_register_tokens.expand(x.shape[0], -1, -1).to(dtype=x_tok.dtype)
            x_tok = torch.cat([dec_registers, x_tok], dim=1)
            dec_feat_rope = self._wrap_rope_with_prefix(self.dec_feat_rope, self.num_register_tokens)

        for i in range(self.num_encoder_blocks, self.num_blocks):
            x_tok = self.blocks[i](x_tok, s, feat_rope=dec_feat_rope)

        if self.num_register_tokens > 0:
            x_tok = x_tok[:, self.num_register_tokens :, :]
            s = s[:, self.num_register_tokens :, :]
            if base_s is not None:
                base_s = base_s[:, self.num_register_tokens :, :]

        x_tok = self.final_layer(x_tok, s)
        x_img = self.unpatchify(x_tok)
        if base_s is not None and self.base_final_layer is not None:
            base_tok = self.base_final_layer(base_s, base_s)
            base_img = self.unpatchify(base_tok)
            if aux_out is not None:
                if return_intermediate:
                    return (x_img, aux_out), zt_intermediate
                return x_img, aux_out
            if return_intermediate:
                return (x_img, base_img), zt_intermediate
            return x_img, base_img
        if self.base_final_layer is not None and self.base_model_depth is None:
            # Keep a resumed IG checkpoint DDP/optimizer-compatible while the
            # early-loss route is disabled. Every dormant head parameter stays
            # in the graph with an exactly zero contribution and zero gradient.
            zero_dependency = sum(
                parameter.reshape(-1)[0] * 0.0
                for parameter in self.base_final_layer.parameters()
            )
            x_img = x_img + zero_dependency.to(dtype=x_img.dtype)
        if aux_out is not None:
            if return_intermediate:
                return (x_img, aux_out), zt_intermediate
            return x_img, aux_out
        if return_intermediate:
            return x_img, zt_intermediate
        return x_img

    def forward_with_cfg(
        self,
        x,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        cfg_scale_aux: float = 1.0,
        meta: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        cls: Optional[torch.Tensor] = None,
        aux: Optional[torch.Tensor] = None,
        cfg_interval: Union[Tuple[float, float], List[Tuple[float, float]]] = (0.0, 1.0),
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Classic CFG wrapper.
        Convention: caller passes y (and meta, if used) as length 2n:
          y = cat([y_cond, y_null])
          meta = cat([meta_cond, meta_null])  (optional)
        and x has length 2n.

        We duplicate x[:n] into combined=(2n) and run forward(combined, t, y, meta).
        """
        if y is None:
            y = kwargs.pop("context", None)
        if y is None:
            raise ValueError("DiTwDDTHead.forward_with_cfg requires y or context.")
        x, aux = self._extract_inputs(x, aux)
        half_x = x[: len(x) // 2]
        combined = torch.cat([half_x, half_x], dim=0)

        combined_aux = None
        if self.predict_aux or self.predict_aux_tokens:
            if aux is None:
                raise ValueError("forward_with_cfg requires aux when auxiliary prediction is enabled")
            half_aux = aux[: len(aux) // 2]
            combined_aux = torch.cat([half_aux, half_aux], dim=0)

        model_out = self.forward(combined, t, y, meta=meta, cls=cls, aux=combined_aux, **kwargs)
        x_out, aux_out = self._split_outputs(model_out)

        eps, rest = x_out[:, : self.in_channels], x_out[:, self.in_channels :]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

        if aux_out is not None:
            cond_aux, uncond_aux = torch.split(aux_out, len(aux_out) // 2, dim=0)

        t_half = t[: len(t) // 2]
        in_mask = self._interval_mask(t_half, cfg_interval, default_all=not self._is_interval_list(cfg_interval))
        half_eps = torch.where(
            self._mask_like(in_mask, cond_eps),
            uncond_eps + float(cfg_scale) * (cond_eps - uncond_eps),
            cond_eps,
        )

        eps = torch.cat([half_eps, half_eps], dim=0)
        x_return = torch.cat([eps, rest], dim=1)

        if aux_out is not None:
            half_aux = torch.where(
                self._mask_like(in_mask, cond_aux),
                uncond_aux + float(cfg_scale_aux) * (cond_aux - uncond_aux),
                cond_aux,
            )
            aux_return = torch.cat([half_aux, half_aux], dim=0)
            return x_return, aux_return

        return x_return

    def forward_with_autoguidance(
        self,
        x,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        additional_model_forward=None,
        cfg_scale_aux: float = 1.0,
        meta: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        cls: Optional[torch.Tensor] = None,
        aux: Optional[torch.Tensor] = None,
        cfg_interval: Union[Tuple[float, float], List[Tuple[float, float]]] = (0.0, 1.0),
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Autoguidance wrapper: blend eps with auxiliary model's eps inside cfg_interval.
        """
        if y is None:
            y = kwargs.pop("context", None)
        if y is None:
            raise ValueError("DiTwDDTHead.forward_with_autoguidance requires y or context.")
        if additional_model_forward is None:
            raise ValueError("forward_with_autoguidance requires additional_model_forward.")
        x, aux = self._extract_inputs(x, aux)
        half = x[: len(x) // 2]
        t_half = t[: len(t) // 2]
        y_half = y[: len(y) // 2]
        meta_half = meta[: len(meta) // 2] if meta is not None else None
        cls_half = cls[: len(cls) // 2] if cls is not None else None
        aux_half = aux[: len(aux) // 2] if aux is not None else None

        model_out = self.forward(half, t_half, y_half, meta=meta_half, cls=cls_half, aux=aux_half, **kwargs)
        # be defensive: aux model may not accept meta/kwargs
        try:
            ag_model_out = additional_model_forward(half, t_half, y_half, meta=meta_half, cls=cls_half, aux=aux_half, **kwargs)
        except TypeError:
            ag_model_out = additional_model_forward(half, t_half, y_half)

        x_out, aux_out = self._split_outputs(model_out)
        ag_x_out, ag_aux_out = self._split_outputs(ag_model_out)

        eps, rest = x_out[:, : self.in_channels], x_out[:, self.in_channels :]
        ag_eps = ag_x_out[:, : self.in_channels]

        in_mask = self._interval_mask(t_half, cfg_interval, default_all=not self._is_interval_list(cfg_interval))
        out = torch.where(
            self._mask_like(in_mask, eps),
            ag_eps + float(cfg_scale) * (eps - ag_eps),
            eps,
        )

        x_ret_half = torch.cat([out, rest], dim=1)
        x_ret = torch.cat([x_ret_half, x_ret_half], dim=0)

        if aux_out is not None:
            if ag_aux_out is None:
                raise ValueError("additional_model_forward must also return aux output when auxiliary prediction is enabled")
            aux_ret_half = torch.where(
                self._mask_like(in_mask, aux_out),
                ag_aux_out + float(cfg_scale_aux) * (aux_out - ag_aux_out),
                aux_out,
            )
            aux_ret = torch.cat([aux_ret_half, aux_ret_half], dim=0)
            return x_ret, aux_ret

        return x_ret


class DiTwDDTHeadIG(DiTwDDTHead):
    """DDT head with an early-exit base prediction for internal guidance."""

    def __init__(self, base_model_depth: int = 8, use_rmsnorm: bool = True, **kwargs):
        super().__init__(base_model_depth=base_model_depth, use_rmsnorm=use_rmsnorm, **kwargs)
        self.base_final_layer = DDTFinalLayer(
            hidden_size=self.decoder_hidden_size,
            patch_size=1,
            out_channels=self.x_channel_per_token,
            use_rmsnorm=use_rmsnorm,
            adaln_shift_bound=self.adaln_shift_bound,
            adaln_scale_bound=self.adaln_scale_bound,
        )
        nn.init.constant_(self.base_final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.base_final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.base_final_layer.linear.weight, 0)
        nn.init.constant_(self.base_final_layer.linear.bias, 0)

__all__ = ["DiTwDDTHead", "DiTwDDTHeadIG"]

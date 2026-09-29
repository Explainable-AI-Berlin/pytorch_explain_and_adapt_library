# coding=utf-8
# Copyright 2022 Facebook AI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch ViT MAE decoder, extended for RAE aux-token conditioning."""

import collections.abc
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from .utils import ViTMAEConfig, ACT2FN, ModelOutput
from transformers.modeling_outputs import BaseModelOutput


@dataclass
class ViTMAEModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    mask: torch.LongTensor = None
    ids_restore: torch.LongTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class ViTMAEDecoderOutput(ModelOutput):
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    aux_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class ViTMAEForPreTrainingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    mask: torch.LongTensor = None
    ids_restore: torch.LongTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


def get_2d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if add_cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def modulate(hidden_states: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    AdaLN / FiLM-style modulation.

    hidden_states: [B, N, C]
    shift:         [B, C]
    scale:         [B, C]
    """
    return hidden_states * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class ViTMAEEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.
    """

    def __init__(self, config):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.patch_embeddings = ViTMAEPatchEmbeddings(config)
        self.num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, config.hidden_size), requires_grad=False
        )
        self.config = config
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.position_embeddings.shape[-1],
            int(self.patch_embeddings.num_patches**0.5),
            add_cls_token=True,
        )
        self.position_embeddings.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.patch_embeddings.projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=self.config.initializer_range)

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1

        if num_patches == num_positions and height == width:
            return self.position_embeddings

        class_pos_embed = self.position_embeddings[:, 0, :]
        patch_pos_embed = self.position_embeddings[:, 1:, :]
        dim = embeddings.shape[-1]
        h0 = height // self.config.patch_size
        w0 = width // self.config.patch_size
        h0, w0 = h0 + 0.1, w0 + 0.1
        patch_pos_embed = patch_pos_embed.reshape(
            1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim
        )
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
            mode="bicubic",
            align_corners=False,
        )
        if int(h0) != patch_pos_embed.shape[-2] or int(w0) != patch_pos_embed.shape[-1]:
            raise ValueError("Width or height does not match with the interpolated position embeddings")
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def random_masking(self, sequence, noise=None):
        batch_size, seq_length, dim = sequence.shape
        len_keep = int(seq_length * (1 - self.config.mask_ratio))

        if noise is None:
            noise = torch.rand(batch_size, seq_length, device=sequence.device)

        ids_shuffle = torch.argsort(noise, dim=1).to(sequence.device)
        ids_restore = torch.argsort(ids_shuffle, dim=1).to(sequence.device)

        ids_keep = ids_shuffle[:, :len_keep]
        sequence_unmasked = torch.gather(
            sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim)
        )

        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return sequence_unmasked, mask, ids_restore

    def forward(self, pixel_values, noise=None, interpolate_pos_encoding: bool = False):
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        if interpolate_pos_encoding:
            position_embeddings = self.interpolate_pos_encoding(embeddings, height, width)
        else:
            position_embeddings = self.position_embeddings

        embeddings = embeddings + position_embeddings[:, 1:, :]
        embeddings, mask, ids_restore = self.random_masking(embeddings, noise)

        cls_token = self.cls_token + position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        return embeddings, mask, ids_restore


class ViTMAEPatchEmbeddings(nn.Module):
    """
    Turns pixel_values of shape (B, C, H, W) into hidden_states of shape (B, N, D).
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.projection = nn.Conv2d(
            num_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, pixel_values, interpolate_pos_encoding: bool = False):
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values matches the configuration."
            )

        if not interpolate_pos_encoding and (height != self.image_size[0] or width != self.image_size[1]):
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model "
                f"({self.image_size[0]}*{self.image_size[1]})."
            )
        x = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return x


class ViTMAESelfAttention(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class ViTMAECrossAttention(nn.Module):
    """
    Standard multi-head cross-attention:
      queries from hidden_states
      keys/values from context_states
    """

    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        context_states: torch.Tensor,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(context_states))
        value_layer = self.transpose_for_scores(self.value(context_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        context_layer = self.out(context_layer)
        context_layer = self.out_dropout(context_layer)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class ViTMAESdpaSelfAttention(ViTMAESelfAttention):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__(config)
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        context_layer = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            head_mask,
            self.attention_probs_dropout_prob if self.training else 0.0,
            is_causal=False,
            scale=None,
        )

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer, None


class ViTMAESelfOutput(nn.Module):
    """
    The residual connection is defined in ViTMAELayer instead of here.
    """

    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class ViTMAEAttention(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.attention = ViTMAESelfAttention(config)
        self.output = ViTMAESelfOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class ViTMAEIntermediate(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class ViTMAEOutput(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class ViTMAELayer(nn.Module):
    """Corresponds to the Block class in the timm implementation, extended for aux conditioning."""

    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

        self.attention = ViTMAEAttention(config)
        self.intermediate = ViTMAEIntermediate(config)
        self.output = ViTMAEOutput(config)

        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # legacy / pooled AdaLN path
        self.use_global_cond = bool(getattr(config, "use_global_cond", False))
        self.global_cond_dim = int(getattr(config, "global_cond_dim", config.hidden_size))
        self.adaln_zero_init = bool(getattr(config, "adaln_zero_init", True))

        if self.use_global_cond:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.global_cond_dim, 4 * config.hidden_size, bias=True),
            )
            if self.adaln_zero_init:
                nn.init.zeros_(self.adaLN_modulation[-1].weight)
                nn.init.zeros_(self.adaLN_modulation[-1].bias)
        else:
            self.adaLN_modulation = None

        # new explicit aux mode
        self.decoder_aux_mode = str(getattr(config, "decoder_aux_mode", "discard"))
        self.use_cross_attn = (self.decoder_aux_mode == "cross_attn")
        self.cross_attn_bidirectional = bool(getattr(config, "cross_attn_bidirectional", False))

        if self.use_cross_attn:
            self.cross_attn_q_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.cross_attn_kv_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.cross_attn = ViTMAECrossAttention(config)
            if self.cross_attn_bidirectional:
                self.aux_cross_attn_q_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
                self.aux_cross_attn_kv_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
                self.aux_cross_attn = ViTMAECrossAttention(config)
            else:
                self.aux_cross_attn_q_norm = None
                self.aux_cross_attn_kv_norm = None
                self.aux_cross_attn = None
        else:
            self.cross_attn_q_norm = None
            self.cross_attn_kv_norm = None
            self.cross_attn = None
            self.aux_cross_attn_q_norm = None
            self.aux_cross_attn_kv_norm = None
            self.aux_cross_attn = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        cond: Optional[torch.Tensor] = None,
        aux_tokens: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        normed_before = self.layernorm_before(hidden_states)

        if self.use_global_cond and cond is not None:
            shift_attn, scale_attn, shift_mlp, scale_mlp = self.adaLN_modulation(cond).chunk(4, dim=-1)
            normed_before = modulate(normed_before, shift_attn, scale_attn)
        else:
            shift_mlp = scale_mlp = None

        self_attention_outputs = self.attention(
            normed_before,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        hidden_states = attention_output + hidden_states

        if self.use_cross_attn and aux_tokens is not None:
            cross_q = self.cross_attn_q_norm(hidden_states)
            cross_kv = self.cross_attn_kv_norm(aux_tokens)
            cross_outputs = self.cross_attn(
                cross_q,
                cross_kv,
                output_attentions=False,
            )
            hidden_states = hidden_states + cross_outputs[0]
            if self.cross_attn_bidirectional:
                aux_q = self.aux_cross_attn_q_norm(aux_tokens)
                aux_kv = self.aux_cross_attn_kv_norm(hidden_states)
                aux_outputs = self.aux_cross_attn(
                    aux_q,
                    aux_kv,
                    output_attentions=False,
                )
                aux_tokens = aux_tokens + aux_outputs[0]

        layer_output = self.layernorm_after(hidden_states)
        if self.use_global_cond and cond is not None:
            layer_output = modulate(layer_output, shift_mlp, scale_mlp)

        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output, hidden_states)

        if self.use_cross_attn and self.cross_attn_bidirectional and aux_tokens is not None:
            outputs = (layer_output, aux_tokens) + outputs
        else:
            outputs = (layer_output,) + outputs
        return outputs


class GeneralDecoder(nn.Module):
    def __init__(self, config, num_patches):
        super().__init__()
        self.decoder_embed = nn.Linear(config.hidden_size, config.decoder_hidden_size, bias=True)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.decoder_hidden_size), requires_grad=False
        )

        # legacy pooled-cond path
        legacy_use_global_cond = bool(getattr(config, "use_global_cond", False))
        self.global_cond_dim = int(getattr(config, "global_cond_dim", config.hidden_size))
        self.adaln_zero_init = bool(getattr(config, "adaln_zero_init", True))

        # explicit aux mode
        self.decoder_aux_mode = getattr(config, "decoder_aux_mode", None)
        if self.decoder_aux_mode is None:
            self.decoder_aux_mode = "adaln_pool" if legacy_use_global_cond else "discard"
        self.decoder_aux_mode = str(self.decoder_aux_mode)

        if self.decoder_aux_mode not in {"discard", "adaln_pool", "prepend", "cross_attn"}:
            raise ValueError(
                "decoder_aux_mode must be one of "
                "['discard', 'adaln_pool', 'prepend', 'cross_attn'], "
                f"got {self.decoder_aux_mode}"
            )

        self.use_global_cond = legacy_use_global_cond or (self.decoder_aux_mode == "adaln_pool")
        self.num_aux_tokens = int(getattr(config, "num_aux_tokens", 0))

        self.use_aux_tokens = self.decoder_aux_mode in {"prepend", "cross_attn"}
        if self.use_aux_tokens and self.num_aux_tokens <= 0:
            raise ValueError(
                f"decoder_aux_mode='{self.decoder_aux_mode}' requires num_aux_tokens > 0, "
                f"got {self.num_aux_tokens}"
            )

        if self.use_aux_tokens:
            self.decoder_aux_embed = nn.Linear(config.hidden_size, config.decoder_hidden_size, bias=True)
            self.aux_pos_embed = nn.Parameter(
                torch.zeros(1, self.num_aux_tokens, config.decoder_hidden_size)
            )
        else:
            self.decoder_aux_embed = None
            self.aux_pos_embed = None

        decoder_config = deepcopy(config)
        decoder_config.hidden_size = config.decoder_hidden_size
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        decoder_config.use_global_cond = self.use_global_cond
        decoder_config.global_cond_dim = self.global_cond_dim
        decoder_config.adaln_zero_init = self.adaln_zero_init
        decoder_config.decoder_aux_mode = self.decoder_aux_mode
        decoder_config.cross_attn_bidirectional = bool(getattr(config, "cross_attn_bidirectional", False))

        self.decoder_layers = nn.ModuleList(
            [ViTMAELayer(decoder_config) for _ in range(config.decoder_num_hidden_layers)]
        )

        self.decoder_norm = nn.LayerNorm(config.decoder_hidden_size, eps=config.layer_norm_eps)
        self.decoder_pred = nn.Linear(
            config.decoder_hidden_size, config.patch_size**2 * config.num_channels, bias=True
        )

        self.gradient_checkpointing = False
        self.config = config
        self.num_patches = num_patches
        self.decoder_config = decoder_config
        self.cross_attn_bidirectional = bool(getattr(decoder_config, "cross_attn_bidirectional", False))

        self.initialize_weights(num_patches)
        self.set_trainable_cls_token()

    def set_trainable_cls_token(self, tensor: Optional[torch.Tensor] = None):
        tensor = torch.zeros(1, 1, self.decoder_config.hidden_size) if tensor is None else tensor
        self.trainable_cls_token = nn.Parameter(tensor)

    def interpolate_pos_encoding(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Interpolation function for decoder patch positions only.
        """
        embeddings_positions = embeddings.shape[1] - 1
        num_positions = self.decoder_pos_embed.shape[1] - 1

        class_pos_embed = self.decoder_pos_embed[:, 0, :]
        patch_pos_embed = self.decoder_pos_embed[:, 1:, :]
        dim = self.decoder_pos_embed.shape[-1]

        patch_pos_embed = patch_pos_embed.reshape(1, 1, -1, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(1, embeddings_positions / num_positions),
            mode="bicubic",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def interpolate_latent(self, x: torch.Tensor) -> torch.Tensor:
        b, l, c = x.shape
        if l == self.num_patches:
            return x
        h, w = int(l**0.5), int(l**0.5)
        if h * w != l:
            raise ValueError(f"Latent token count {l} is not a perfect square and cannot be interpolated as a grid.")
        x = x.reshape(b, h, w, c)
        x = x.permute(0, 3, 1, 2)
        target_size = (int(self.num_patches**0.5), int(self.num_patches**0.5))
        x = nn.functional.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        x = x.permute(0, 2, 3, 1).contiguous().view(b, self.num_patches, c)
        return x

    def initialize_weights(self, num_patches):
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(num_patches**0.5),
            add_cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        if self.use_aux_tokens:
            nn.init.normal_(self.aux_pos_embed, std=self.config.initializer_range)
            nn.init.xavier_uniform_(self.decoder_aux_embed.weight)
            if self.decoder_aux_embed.bias is not None:
                nn.init.zeros_(self.decoder_aux_embed.bias)

    def unpatchify(self, patchified_pixel_values, original_image_size: Optional[Tuple[int, int]] = None):
        patch_size, num_channels = self.config.patch_size, self.config.num_channels
        original_image_size = (
            original_image_size
            if original_image_size is not None
            else (self.config.image_size, self.config.image_size)
        )
        original_height, original_width = original_image_size
        num_patches_h = original_height // patch_size
        num_patches_w = original_width // patch_size

        if num_patches_h * num_patches_w != patchified_pixel_values.shape[1]:
            raise ValueError(
                f"The number of patches in the patchified pixel values {patchified_pixel_values.shape[1]} "
                f"does not match the number of patches on original image {num_patches_h}*{num_patches_w}"
            )

        batch_size = patchified_pixel_values.shape[0]
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_patches_h,
            num_patches_w,
            patch_size,
            patch_size,
            num_channels,
        )
        patchified_pixel_values = torch.einsum("nhwpqc->nchpwq", patchified_pixel_values)
        pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_channels,
            num_patches_h * patch_size,
            num_patches_w * patch_size,
        )
        return pixel_values

    def _prepare_patch_tokens(
        self,
        hidden_states: torch.Tensor,
        drop_cls_token: bool,
    ) -> torch.Tensor:
        x = self.decoder_embed(hidden_states)

        if drop_cls_token:
            x_ = x[:, 1:, :]
            x_ = self.interpolate_latent(x_)
        else:
            x_ = self.interpolate_latent(x)

        return x_

    def _prepare_aux_tokens(self, aux_tokens: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if not self.use_aux_tokens:
            return None
        if aux_tokens is None:
            return None
        if aux_tokens.ndim != 3:
            raise ValueError(f"aux_tokens must have shape [B, K, C], got {tuple(aux_tokens.shape)}")
        if aux_tokens.shape[1] != self.num_aux_tokens:
            raise ValueError(
                f"Expected aux_tokens.shape[1] == num_aux_tokens == {self.num_aux_tokens}, "
                f"got {aux_tokens.shape[1]}"
            )
        aux_tokens = self.decoder_aux_embed(aux_tokens)
        aux_tokens = aux_tokens + self.aux_pos_embed
        return aux_tokens

    def forward(
        self,
        hidden_states,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        interpolate_pos_encoding: bool = False,
        drop_cls_token: bool = False,
        cond: Optional[torch.Tensor] = None,
        aux_tokens: Optional[torch.Tensor] = None,
    ):
        patch_tokens = self._prepare_patch_tokens(hidden_states, drop_cls_token=drop_cls_token)

        cls_token = self.trainable_cls_token.expand(patch_tokens.shape[0], -1, -1)
        aux_tokens_proj = self._prepare_aux_tokens(aux_tokens)

        if interpolate_pos_encoding:
            assert drop_cls_token, "interpolate_pos_encoding only works with drop_cls_token=True"

        if self.decoder_aux_mode == "prepend":
            if aux_tokens_proj is None:
                raise ValueError("decoder_aux_mode='prepend' requires aux_tokens.")
            x = torch.cat([cls_token, aux_tokens_proj, patch_tokens], dim=1)

            if interpolate_pos_encoding:
                patch_pos = self.interpolate_pos_encoding(torch.cat([cls_token, patch_tokens], dim=1))[:, 1:, :]
                cls_pos = self.interpolate_pos_encoding(torch.cat([cls_token, patch_tokens], dim=1))[:, :1, :]
            else:
                cls_pos = self.decoder_pos_embed[:, :1, :]
                patch_pos = self.decoder_pos_embed[:, 1:, :]
            pos_embed = torch.cat([cls_pos, self.aux_pos_embed, patch_pos], dim=1)
            hidden_states = x + pos_embed

        else:
            x = torch.cat([cls_token, patch_tokens], dim=1)
            if interpolate_pos_encoding:
                decoder_pos_embed = self.interpolate_pos_encoding(x)
            else:
                decoder_pos_embed = self.decoder_pos_embed
            hidden_states = x + decoder_pos_embed

        all_hidden_states = () if output_hidden_states else None
        all_aux_hidden_states = (
            ()
            if output_hidden_states and self.decoder_aux_mode == "cross_attn" and self.cross_attn_bidirectional
            else None
        )
        all_self_attentions = () if output_attentions else None

        layer_aux_tokens = aux_tokens_proj if self.decoder_aux_mode == "cross_attn" else None

        for i, layer_module in enumerate(self.decoder_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                if all_aux_hidden_states is not None:
                    all_aux_hidden_states = all_aux_hidden_states + (layer_aux_tokens,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    None,
                    output_attentions,
                    cond,
                    layer_aux_tokens,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    head_mask=None,
                    output_attentions=output_attentions,
                    cond=cond,
                    aux_tokens=layer_aux_tokens,
                )

            hidden_states = layer_outputs[0]
            if self.decoder_aux_mode == "cross_attn" and self.cross_attn_bidirectional and layer_aux_tokens is not None:
                layer_aux_tokens = layer_outputs[1]
                attn_idx = 2
            else:
                attn_idx = 1

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[attn_idx],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            if all_aux_hidden_states is not None:
                all_aux_hidden_states = all_aux_hidden_states + (layer_aux_tokens,)

        hidden_states = self.decoder_norm(hidden_states)
        logits = self.decoder_pred(hidden_states)

        if self.decoder_aux_mode == "prepend":
            logits = logits[:, 1 + self.num_aux_tokens :, :]
        else:
            logits = logits[:, 1:, :]

        if not return_dict:
            return tuple(v for v in [logits, all_hidden_states, all_aux_hidden_states, all_self_attentions] if v is not None)

        return ViTMAEDecoderOutput(
            logits=logits,
            hidden_states=all_hidden_states,
            aux_hidden_states=all_aux_hidden_states,
            attentions=all_self_attentions,
        )

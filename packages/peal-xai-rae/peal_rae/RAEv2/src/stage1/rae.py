from math import sqrt
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, PretrainedConfig

from encoders.vision_encoder import create_encoder
from .decoders import GeneralDecoder


def _load_decoder(
    config_path,
    hidden_size,
    patch_size,
    num_patches,
    pretrained_path=None,
    decoder_aux_mode: str = "discard",
    aux_token_source: Optional[str] = None,
    num_aux_tokens: int = 0,
    aux_pool: str = "mean",
    cross_attn_bidirectional: bool = False,
):
    # The shipped decoder configs hold the placeholder "SHOULD BE RELOADED" for
    # patch_size; transformers>=5 validates fields on construction and rejects
    # it, so substitute the real value before the config object is built.
    config_dict, _ = PretrainedConfig.get_config_dict(config_path)
    config_dict["patch_size"] = patch_size
    config = AutoConfig.for_model(**config_dict)
    config.hidden_size = hidden_size
    config.patch_size = patch_size
    config.image_size = int(patch_size * sqrt(num_patches))
    config.decoder_aux_mode = str(decoder_aux_mode)
    config.aux_token_source = aux_token_source
    config.aux_pool = aux_pool
    config.num_aux_tokens = int(num_aux_tokens)
    config.cross_attn_bidirectional = bool(cross_attn_bidirectional)
    decoder = GeneralDecoder(config, num_patches=num_patches)
    if pretrained_path is not None:
        print(f"Loading pretrained decoder from {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        keys = decoder.load_state_dict(state_dict, strict=False)
        if keys.missing_keys:
            print(f"Missing keys: {keys.missing_keys}")
    return decoder


def _load_normalization_stats(path):
    if path is None:
        return None, None, None, None, False, False
    stats = torch.load(path, map_location='cpu', weights_only=False)
    print(f"Loaded normalization stats from {path}")
    latent_mean = stats.get('mean', None)
    latent_var = stats.get('var', None)
    aux_mean = stats.get('aux_mean', None)
    aux_var = stats.get('aux_var', None)
    do_latent = latent_mean is not None or latent_var is not None
    do_aux = aux_mean is not None or aux_var is not None
    return latent_mean, latent_var, aux_mean, aux_var, do_latent, do_aux


class RAE(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        resolution: int = 256,
        decoder_config_path: str = 'vit_mae-base',
        decoder_patch_size: int = 16,
        pretrained_decoder_path: Optional[str] = None,
        noise_tau: float = 0.8,
        normalization_stat_path: Optional[str] = None,
        eps: float = 1e-5,
        aux_token_source: Optional[str] = None,
        num_aux_tokens: int = 0,
        decoder_aux_mode: str = "discard",
        aux_pool: str = "mean",
        cross_attn_bidirectional: bool = False,
        aux_dropout_prob: float = 0.0,
        aux_dropout_mode: str = "zero",
        aux_slot_prediction_head: bool = False,
        aux_slot_prediction_source: str = "cls+register",
        aux_slot_prediction_num_tokens: Optional[int] = None,
        aux_slot_prediction_layer_index: int = -1,
        patch_dropout_prob: float = 0.0,
        patch_dropout_mode: str = "zero",
        patch_dropout_keep_at_least_one: bool = True,
    ):
        super().__init__()

        self.encoder = create_encoder(
            encoder_name,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            resolution=resolution,
        )
        self.resolution = resolution
        self.encoder_patch_size = self.encoder.patch_size
        self.latent_dim = self.encoder.hidden_size
        self.base_patches = (resolution // 16) ** 2
        self.eps = float(eps)
        self.noise_tau = float(noise_tau)

        self.decoder_aux_mode = str(decoder_aux_mode)
        self.aux_token_source = None if aux_token_source is None else str(aux_token_source)
        self.num_aux_tokens = int(num_aux_tokens)
        self.aux_pool = str(aux_pool)
        self.cross_attn_bidirectional = bool(cross_attn_bidirectional)
        self.aux_dropout_prob = float(aux_dropout_prob)
        self.aux_dropout_mode = str(aux_dropout_mode)
        self.aux_slot_prediction_head_enabled = bool(aux_slot_prediction_head)
        self.aux_slot_prediction_source = str(aux_slot_prediction_source)
        self.aux_slot_prediction_num_tokens = aux_slot_prediction_num_tokens
        self.aux_slot_prediction_layer_index = int(aux_slot_prediction_layer_index)
        self.patch_dropout_prob = float(patch_dropout_prob)
        self.patch_dropout_mode = str(patch_dropout_mode)
        self.patch_dropout_keep_at_least_one = bool(patch_dropout_keep_at_least_one)

        if self.decoder_aux_mode not in {"discard", "adaln_pool", "prepend", "cross_attn"}:
            raise ValueError(
                "decoder_aux_mode must be one of {'discard', 'adaln_pool', 'prepend', 'cross_attn'}, "
                f"got {self.decoder_aux_mode!r}"
            )
        if self.decoder_aux_mode == "discard":
            self.aux_token_source = None
            self.num_aux_tokens = 0
        elif self.aux_token_source not in {"cls+register", "learned"}:
            raise ValueError(
                "This generic RAEv2 aux path currently supports only "
                "aux_token_source in {'cls+register', 'learned'}."
            )
        elif self.num_aux_tokens <= 0:
            raise ValueError("num_aux_tokens must be positive when decoder_aux_mode is not 'discard'.")
        if self.aux_token_source == "learned" and self.decoder_aux_mode not in {"prepend", "cross_attn"}:
            raise ValueError(
                "aux_token_source='learned' currently requires decoder_aux_mode in "
                "{'prepend', 'cross_attn'}."
            )
        if self.aux_dropout_mode not in {"zero", "learned_null"}:
            raise ValueError(
                f"aux_dropout_mode must be one of ['zero', 'learned_null'], got {self.aux_dropout_mode!r}."
            )
        if not (0.0 <= self.aux_dropout_prob < 1.0):
            raise ValueError(
                f"aux_dropout_prob must be in [0, 1), got {self.aux_dropout_prob}."
            )
        if self.aux_slot_prediction_source not in {"cls", "register", "cls+register"}:
            raise ValueError(
                "aux_slot_prediction_source must be one of ['cls', 'register', 'cls+register'], "
                f"got {self.aux_slot_prediction_source!r}."
            )
        if self.aux_slot_prediction_layer_index < -1:
            raise ValueError(
                "aux_slot_prediction_layer_index must be -1 or a non-negative decoder hidden-state index, "
                f"got {self.aux_slot_prediction_layer_index}."
            )
        if pretrained_decoder_path is not None and (
            self.aux_token_source == "learned" or self.aux_dropout_mode == "learned_null"
        ):
            raise ValueError(
                "Learned-slot / learned-null RAEv2 models must be restored via stage_1.ckpt; "
                "decoder-only loading through pretrained_decoder_path is unsupported."
            )
        if self.patch_dropout_mode not in {"zero"}:
            raise ValueError(
                f"patch_dropout_mode must be one of ['zero'], got {self.patch_dropout_mode!r}."
            )
        if not (0.0 <= self.patch_dropout_prob < 1.0):
            raise ValueError(
                f"patch_dropout_prob must be in [0, 1), got {self.patch_dropout_prob}."
            )

        self.decoder = _load_decoder(
            decoder_config_path,
            self.latent_dim,
            decoder_patch_size,
            self.base_patches,
            pretrained_decoder_path,
            decoder_aux_mode=self.decoder_aux_mode,
            aux_token_source=self.aux_token_source,
            num_aux_tokens=self.num_aux_tokens,
            aux_pool=self.aux_pool,
            cross_attn_bidirectional=self.cross_attn_bidirectional,
        )
        self.learned_input_aux_tokens = None
        self.learned_null_aux_tokens = None
        if self.aux_token_source == "learned":
            self.learned_input_aux_tokens = nn.Parameter(
                torch.zeros(1, int(self.num_aux_tokens), self.latent_dim)
            )
            nn.init.trunc_normal_(self.learned_input_aux_tokens, std=0.02)
        if self.aux_dropout_mode == "learned_null" and self.decoder_aux_mode in {"prepend", "cross_attn"}:
            self.learned_null_aux_tokens = nn.Parameter(
                torch.zeros(1, int(self.num_aux_tokens), self.latent_dim)
            )
        (
            self.latent_mean,
            self.latent_var,
            self.aux_mean,
            self.aux_var,
            self.do_normalization,
            self.do_aux_normalization,
        ) = _load_normalization_stats(normalization_stat_path)
        if self.aux_slot_prediction_head_enabled:
            if self.aux_token_source != "learned" or self.decoder_aux_mode not in {"prepend", "cross_attn"}:
                raise ValueError(
                    "aux_slot_prediction_head=True requires aux_token_source='learned' and "
                    "decoder_aux_mode in {'prepend', 'cross_attn'}."
                )
            if self.decoder_aux_mode == "cross_attn" and not self.cross_attn_bidirectional:
                raise ValueError(
                    "aux_slot_prediction_head=True with decoder_aux_mode='cross_attn' "
                    "requires cross_attn_bidirectional=True."
                )
            pred_tokens = (
                aux_slot_prediction_num_tokens
                if aux_slot_prediction_num_tokens is not None
                else self._infer_num_tokens_for_source(self.aux_slot_prediction_source)
            )
            if pred_tokens is None or int(pred_tokens) <= 0:
                raise ValueError(
                    "aux_slot_prediction_head=True requires aux_slot_prediction_num_tokens or an inferable "
                    f"aux_slot_prediction_source; got source={self.aux_slot_prediction_source!r}."
                )
            self.aux_slot_prediction_num_tokens = int(pred_tokens)
            if self.num_aux_tokens != self.aux_slot_prediction_num_tokens:
                raise ValueError(
                    "num_aux_tokens must match aux_slot_prediction_num_tokens for supervised learned slots: "
                    f"num_aux_tokens={self.num_aux_tokens}, target={self.aux_slot_prediction_num_tokens}."
                )
            decoder_width = int(self.decoder.decoder_config.hidden_size)
            self.aux_slot_prediction_head = nn.Sequential(
                nn.LayerNorm(decoder_width),
                nn.Linear(decoder_width, self.latent_dim),
            )
        else:
            self.aux_slot_prediction_head = None
        print(
            f"RAE: encoder={encoder_name}, resolution={resolution}, "
            f"patch_size={self.encoder_patch_size}, hidden_size={self.latent_dim}, "
            f"decoder_aux_mode={self.decoder_aux_mode}, num_aux_tokens={self.num_aux_tokens}"
        )
        if self.aux_slot_prediction_head_enabled:
            layer_desc = (
                "post_norm_final"
                if self.aux_slot_prediction_layer_index == -1
                else f"hidden_state[{self.aux_slot_prediction_layer_index}]"
            )
            print(
                "Using aux-slot supervision: "
                f"source={self.aux_slot_prediction_source}, "
                f"target_tokens={self.aux_slot_prediction_num_tokens}, "
                f"layer={layer_desc}"
            )
        if self.decoder_aux_mode == "cross_attn" and self.cross_attn_bidirectional:
            print("Using bidirectional cross-attention aux stream evolution.")
        if self.patch_dropout_prob > 0.0:
            print(
                "Using patch-token dropout: "
                f"prob={self.patch_dropout_prob}, "
                f"mode={self.patch_dropout_mode}, "
                f"keep_at_least_one={self.patch_dropout_keep_at_least_one}"
            )

    def _infer_num_tokens_for_source(self, source: Optional[str]) -> Optional[int]:
        if source is None:
            return None
        num_reg = getattr(self.encoder, "num_register_tokens", None)
        if num_reg is None:
            num_reg = getattr(self.encoder, "num_reg_tokens", None)
        num_prefix = getattr(self.encoder, "num_prefix_tokens", None)

        if source == "cls":
            return 1
        if source == "register":
            return int(num_reg) if num_reg is not None else None
        if source == "cls+register":
            if num_reg is not None:
                return 1 + int(num_reg)
            if num_prefix is not None:
                return int(num_prefix)
            return None
        return None

    def _prepare_raw_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.max() <= 1.0:
            x = x * 255.0
        _, _, h, w = x.shape
        if h != self.resolution or w != self.resolution:
            x = nn.functional.interpolate(
                x,
                size=(self.resolution, self.resolution),
                mode='bicubic',
                align_corners=False,
            )
        return x

    def noising(self, x: torch.Tensor) -> torch.Tensor:
        noise_sigma = self.noise_tau * torch.rand((x.size(0),) + (1,) * (len(x.shape) - 1), device=x.device)
        return x + noise_sigma * torch.randn_like(x)

    def _patch_tokens_to_latent(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        if self.training and self.noise_tau > 0:
            patch_tokens = self.noising(patch_tokens)

        b, n, c = patch_tokens.shape
        h = w = int(sqrt(n))
        z = patch_tokens.transpose(1, 2).view(b, c, h, w)
        if self.do_normalization:
            latent_mean = self.latent_mean.to(z.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(z.device) if self.latent_var is not None else 1
            z = (z - latent_mean) / torch.sqrt(latent_var + self.eps)
        return z

    def _denormalize_latent(self, z: torch.Tensor) -> torch.Tensor:
        if not self.do_normalization:
            return z
        latent_mean = self.latent_mean.to(z.device) if self.latent_mean is not None else 0
        latent_var = self.latent_var.to(z.device) if self.latent_var is not None else 1
        return z * torch.sqrt(latent_var + self.eps) + latent_mean

    def _maybe_drop_patch_latent(self, z: torch.Tensor) -> torch.Tensor:
        if not self.training or self.patch_dropout_prob <= 0.0:
            return z

        if z.ndim != 4:
            raise ValueError(f"Unsupported latent shape for patch dropout: {tuple(z.shape)}")

        b, c, h, w = z.shape
        z_tokens = z.view(b, c, h * w).transpose(1, 2)
        keep_prob = 1.0 - self.patch_dropout_prob
        keep_mask = (torch.rand(b, z_tokens.size(1), 1, device=z_tokens.device) < keep_prob).to(z_tokens.dtype)

        if self.patch_dropout_keep_at_least_one and z_tokens.size(1) > 0:
            all_dropped = keep_mask.sum(dim=1, keepdim=True) == 0
            if all_dropped.any():
                rescue_idx = torch.randint(z_tokens.size(1), (b,), device=z_tokens.device)
                rescue_mask = torch.zeros_like(keep_mask)
                rescue_mask[torch.arange(b, device=z_tokens.device), rescue_idx, 0] = 1.0
                keep_mask = torch.where(all_dropped, rescue_mask, keep_mask)

        if self.patch_dropout_mode == "zero":
            z_tokens = z_tokens * keep_mask
        else:
            raise ValueError(f"Unsupported patch_dropout_mode: {self.patch_dropout_mode}")

        return z_tokens.transpose(1, 2).view(b, c, h, w)

    def _select_aux_tokens(self, global_tokens: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if self.decoder_aux_mode == "discard":
            return None
        if global_tokens is None:
            raise ValueError(
                f"decoder_aux_mode={self.decoder_aux_mode!r} requires encoder global tokens, but none were returned."
            )
        return self._select_aux_tokens_from_source(global_tokens, self.aux_token_source)

    def _select_aux_tokens_from_source(self, global_tokens: torch.Tensor, source: Optional[str]) -> torch.Tensor:
        if global_tokens.ndim != 3:
            raise ValueError(
                f"Expected global_tokens to have shape [B, P, C], got {tuple(global_tokens.shape)}."
            )
        if global_tokens.shape[1] == 0:
            raise ValueError("Encoder returned no prefix/global tokens.")
        if source == "cls":
            return global_tokens[:, :1]
        if source == "register":
            if global_tokens.shape[1] <= 1:
                raise ValueError("Requested register tokens, but encoder returned no register bank.")
            return global_tokens[:, 1:]
        if source == "cls+register":
            if global_tokens.shape[1] < self.num_aux_tokens:
                raise ValueError(
                    f"Expected at least {self.num_aux_tokens} global tokens, got {global_tokens.shape[1]}."
                )
            return global_tokens[:, : self.num_aux_tokens]
        raise ValueError(f"Unsupported aux token source: {source!r}")

    def _get_learned_aux_tokens(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.learned_input_aux_tokens is None:
            raise ValueError(
                "aux_token_source='learned' requested, but learned_input_aux_tokens is not initialized."
            )
        return self.learned_input_aux_tokens.to(device=device, dtype=dtype).expand(batch_size, -1, -1)

    def _maybe_drop_aux(self, aux_tokens: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if aux_tokens is None or not self.training or self.aux_dropout_prob <= 0.0:
            return aux_tokens
        keep = (
            torch.rand(aux_tokens.size(0), 1, 1, device=aux_tokens.device) >= self.aux_dropout_prob
        ).to(aux_tokens.dtype)
        if self.aux_dropout_mode == "learned_null":
            if self.learned_null_aux_tokens is None:
                raise ValueError(
                    "learned null token conditioning requested, but learned_null_aux_tokens is not initialized."
                )
            if self.learned_null_aux_tokens.size(1) != aux_tokens.size(1):
                raise ValueError(
                    "learned_null_aux_tokens shape mismatch: "
                    f"expected K={self.learned_null_aux_tokens.size(1)}, got K={aux_tokens.size(1)}"
                )
            null_aux = self.learned_null_aux_tokens.to(device=aux_tokens.device, dtype=aux_tokens.dtype)
            return aux_tokens * keep + null_aux.expand_as(aux_tokens) * (1 - keep)
        return aux_tokens * keep

    def _select_decoder_hidden_for_aux_slot_prediction(
        self,
        hidden_states,
    ) -> torch.Tensor:
        if self.aux_slot_prediction_layer_index == -1:
            return self.decoder.decoder_norm(hidden_states[-1])
        layer_idx = int(self.aux_slot_prediction_layer_index)
        if layer_idx >= len(hidden_states):
            raise ValueError(
                "aux_slot_prediction_layer_index is out of range for decoder hidden states: "
                f"index={layer_idx}, available=0..{len(hidden_states) - 1}"
            )
        return hidden_states[layer_idx]

    def _select_aux_stream_hidden_for_aux_slot_prediction(
        self,
        aux_hidden_states,
    ) -> torch.Tensor:
        if aux_hidden_states is None:
            raise ValueError(
                "Aux-stream supervision requested, but decoder did not return aux_hidden_states. "
                "Enable cross_attn_bidirectional=True for cross-attention learned-slot supervision."
            )
        if self.aux_slot_prediction_layer_index == -1:
            return aux_hidden_states[-1]
        layer_idx = int(self.aux_slot_prediction_layer_index)
        if layer_idx >= len(aux_hidden_states):
            raise ValueError(
                "aux_slot_prediction_layer_index is out of range for decoder aux hidden states: "
                f"index={layer_idx}, available=0..{len(aux_hidden_states) - 1}"
            )
        return aux_hidden_states[layer_idx]

    def normalize_aux_tokens(self, aux_tokens: torch.Tensor) -> torch.Tensor:
        if not self.do_aux_normalization:
            return aux_tokens
        aux_mean = self.aux_mean.to(aux_tokens.device) if self.aux_mean is not None else 0
        aux_var = self.aux_var.to(aux_tokens.device) if self.aux_var is not None else 1
        return (aux_tokens - aux_mean) / torch.sqrt(aux_var + self.eps)

    def denormalize_aux_tokens(self, aux_tokens: torch.Tensor) -> torch.Tensor:
        if not self.do_aux_normalization:
            return aux_tokens
        aux_mean = self.aux_mean.to(aux_tokens.device) if self.aux_mean is not None else 0
        aux_var = self.aux_var.to(aux_tokens.device) if self.aux_var is not None else 1
        return aux_tokens * torch.sqrt(aux_var + self.eps) + aux_mean

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_raw_input(x)
        patch_tokens = self.encoder(x)
        z = self._patch_tokens_to_latent(patch_tokens)
        return self._maybe_drop_patch_latent(z)

    def encode_with_cls(self, x: torch.Tensor):
        """Encode patch latents and return the raw encoder CLS token."""
        x = self._prepare_raw_input(x)
        patch_tokens, global_tokens = self.encoder.forward_with_global(x)
        z = self._patch_tokens_to_latent(patch_tokens)
        z = self._maybe_drop_patch_latent(z)
        if global_tokens.ndim != 3 or global_tokens.shape[1] < 1:
            raise ValueError(
                "encode_with_cls expected global tokens with shape [B, P, C] "
                f"and P >= 1, got {tuple(global_tokens.shape)}."
            )
        return z, global_tokens[:, 0]

    def encode_for_stage2(self, x: torch.Tensor, normalize_aux_tokens: bool = True):
        x = self._prepare_raw_input(x)
        if self.decoder_aux_mode == "discard":
            z = self.encode(x)
            return z
        if self.aux_token_source == "learned":
            z = self.encode(x)
            return z

        patch_tokens, global_tokens = self.encoder.forward_with_global(x)
        z = self._patch_tokens_to_latent(patch_tokens)
        z = self._maybe_drop_patch_latent(z)
        aux_tokens = self._select_aux_tokens(global_tokens)
        if normalize_aux_tokens:
            aux_tokens = self.normalize_aux_tokens(aux_tokens)
        return z, aux_tokens

    def decode(
        self,
        z: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        aux_tokens: Optional[torch.Tensor] = None,
        cond_is_normalized: bool = False,
        aux_tokens_are_normalized: bool = False,
    ) -> torch.Tensor:
        del cond, cond_is_normalized

        z = self._denormalize_latent(z)
        if aux_tokens_are_normalized and aux_tokens is not None:
            aux_tokens = self.denormalize_aux_tokens(aux_tokens)

        b, c, h, w = z.shape
        z_tokens = z.view(b, c, h * w).transpose(1, 2)
        if aux_tokens is None and self.decoder_aux_mode in {"prepend", "cross_attn"} and self.aux_token_source == "learned":
            aux_tokens = self._get_learned_aux_tokens(
                batch_size=z_tokens.size(0),
                device=z_tokens.device,
                dtype=z_tokens.dtype,
            )
            aux_tokens = self._maybe_drop_aux(aux_tokens)
        output = self.decoder(
            z_tokens,
            drop_cls_token=False,
            cond=None,
            aux_tokens=None if aux_tokens is None else aux_tokens.to(z_tokens.dtype),
        ).logits
        return self.decoder.unpatchify(output)

    def forward_with_aux_slot_prediction(
        self,
        x: torch.Tensor,
        normalize_aux_target: bool = True,
    ):
        if not self.aux_slot_prediction_head_enabled or self.aux_slot_prediction_head is None:
            raise ValueError("forward_with_aux_slot_prediction requires aux_slot_prediction_head=True.")
        if self.decoder_aux_mode not in {"prepend", "cross_attn"} or self.aux_token_source != "learned":
            raise ValueError(
                "Aux-slot prediction requires learned aux tokens consumed by the decoder. "
                "Use decoder_aux_mode in {'prepend', 'cross_attn'} and aux_token_source='learned'."
            )

        x = self._prepare_raw_input(x)
        with torch.no_grad():
            patch_tokens, global_tokens = self.encoder.forward_with_global(x)
            aux_target = self._select_aux_tokens_from_source(
                global_tokens,
                self.aux_slot_prediction_source,
            )
            if normalize_aux_target:
                aux_target = self.normalize_aux_tokens(aux_target)

        z = self._patch_tokens_to_latent(patch_tokens)
        z = self._maybe_drop_patch_latent(z)
        b, c, h, w = z.shape
        z_tokens = z.view(b, c, h * w).transpose(1, 2)
        aux_tokens = self._get_learned_aux_tokens(
            batch_size=z_tokens.size(0),
            device=z_tokens.device,
            dtype=z_tokens.dtype,
        )
        aux_tokens = self._maybe_drop_aux(aux_tokens)
        decoder_output = self.decoder(
            z_tokens,
            drop_cls_token=False,
            cond=None,
            aux_tokens=aux_tokens,
            output_hidden_states=True,
        )
        x_rec = self.decoder.unpatchify(decoder_output.logits)
        if self.decoder_aux_mode == "prepend":
            aux_hidden = self._select_decoder_hidden_for_aux_slot_prediction(
                decoder_output.hidden_states
            )
            aux_slot_states = aux_hidden[:, 1 : 1 + int(self.num_aux_tokens), :]
        else:
            aux_slot_states = self._select_aux_stream_hidden_for_aux_slot_prediction(
                getattr(decoder_output, "aux_hidden_states", None)
            )
        aux_slot_pred = self.aux_slot_prediction_head(aux_slot_states)
        return x_rec, aux_slot_pred, aux_target.detach(), aux_slot_states

    def forward(
        self,
        x: torch.Tensor,
        return_latent: bool = False,
        return_aux_slot_pred: bool = False,
        normalize_aux_slot_pred_target: bool = True,
    ):
        if return_aux_slot_pred:
            return self.forward_with_aux_slot_prediction(
                x,
                normalize_aux_target=normalize_aux_slot_pred_target,
            )
        if self.decoder_aux_mode == "discard":
            z = self.encode(x)
            x_rec = self.decode(z)
            if return_latent:
                return x_rec, z
            return x_rec

        if self.aux_token_source == "learned":
            z = self.encode(x)
            x_rec = self.decode(z)
            if return_latent:
                return x_rec, z
            return x_rec

        z, aux_tokens = self.encode_for_stage2(x, normalize_aux_tokens=False)
        x_rec = self.decode(z, aux_tokens=aux_tokens, aux_tokens_are_normalized=False)
        if return_latent:
            return x_rec, (z, aux_tokens)
        return x_rec

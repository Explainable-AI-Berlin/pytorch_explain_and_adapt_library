from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch


Stage2State = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


@dataclass(frozen=True)
class AuxStateSpec:
    mode: str
    shape: Optional[Tuple[int, ...]] = None

    @property
    def enabled(self) -> bool:
        return self.mode != "none"


def normalize_stage2_state(state: Any) -> Stage2State:
    """
    Normalize stage-2 state into either:
      - patch-only tensor z
      - tuple (z, aux)

    Supports:
      - tensor
      - dict {"z": z, "cond": cond}
      - dict {"z": z, "aux_tokens": aux_tokens}
      - tuple/list like (z, aux)
    """
    if torch.is_tensor(state):
        return state

    if isinstance(state, dict):
        z = state["z"]
        cond = state.get("cond", None)
        aux_tokens = state.get("aux_tokens", None)
        if cond is not None and aux_tokens is not None:
            raise ValueError("Stage-2 state dict cannot contain both 'cond' and 'aux_tokens'.")
        aux = cond if cond is not None else aux_tokens
        return z if aux is None else (z, aux)

    if isinstance(state, (tuple, list)):
        if len(state) == 0:
            raise ValueError("Empty tuple/list stage-2 state.")
        if len(state) == 1:
            return state[0]
        if len(state) == 2:
            z, aux = state
            return z if aux is None else (z, aux)
        raise ValueError(f"Unsupported tuple/list stage-2 state of length {len(state)}")

    raise TypeError(f"Unsupported stage-2 state type: {type(state)}")


def final_state_from_trajectory(trajectory: Any) -> Stage2State:
    """
    Extract the terminal Stage-2 state from a sampler trajectory.

    ODE sampling with tensor state returns a tensor trajectory [T, ...].
    ODE sampling with tuple state returns a tuple of tensor trajectories:
      ([T, ...], [T, ...]).
    SDE sampling returns a Python list of per-step states.
    """
    if isinstance(trajectory, list):
        if len(trajectory) == 0:
            raise ValueError("Sampler trajectory is empty.")
        return normalize_stage2_state(trajectory[-1])

    if torch.is_tensor(trajectory):
        if trajectory.ndim == 0:
            raise ValueError("Sampler tensor trajectory must include a time dimension.")
        return trajectory[-1]

    if isinstance(trajectory, tuple):
        if len(trajectory) == 0:
            raise ValueError("Sampler tuple trajectory is empty.")
        if all(torch.is_tensor(part) for part in trajectory):
            return tuple(part[-1] for part in trajectory)

    return normalize_stage2_state(trajectory)


def infer_aux_state_spec(rae, latent_size: Optional[Tuple[int, ...]] = None) -> AuxStateSpec:
    decoder_aux_mode = str(getattr(rae, "decoder_aux_mode", "discard"))
    aux_token_source = getattr(rae, "aux_token_source", None)
    latent_dim = int(getattr(rae, "latent_dim", latent_size[0] if latent_size else 0))

    if decoder_aux_mode == "adaln_pool":
        return AuxStateSpec(mode="pooled", shape=(latent_dim,))

    if decoder_aux_mode in {"prepend", "cross_attn"}:
        if aux_token_source == "learned":
            return AuxStateSpec(mode="none", shape=None)
        num_aux_tokens = int(getattr(rae, "num_aux_tokens", 0))
        if num_aux_tokens <= 0:
            raise ValueError(
                f"decoder_aux_mode='{decoder_aux_mode}' requires num_aux_tokens > 0 for Stage-2."
            )
        return AuxStateSpec(mode="tokens", shape=(num_aux_tokens, latent_dim))

    return AuxStateSpec(mode="none", shape=None)


def validate_decode_target_compatibility(
    train_rae,
    decode_rae,
    latent_size: Optional[Tuple[int, ...]] = None,
) -> None:
    train_latent_dim = int(getattr(train_rae, "latent_dim", latent_size[0] if latent_size else 0))
    decode_latent_dim = int(getattr(decode_rae, "latent_dim", latent_size[0] if latent_size else 0))
    if train_latent_dim != decode_latent_dim:
        raise ValueError(
            "decode_stage_1 latent_dim mismatch: "
            f"train={train_latent_dim}, decode={decode_latent_dim}."
        )

    train_base_patches = int(getattr(train_rae, "base_patches", 0))
    decode_base_patches = int(getattr(decode_rae, "base_patches", 0))
    if train_base_patches != decode_base_patches:
        raise ValueError(
            "decode_stage_1 patch-grid mismatch: "
            f"train={train_base_patches}, decode={decode_base_patches}."
        )

    train_resolution = int(getattr(train_rae, "resolution", 0))
    decode_resolution = int(getattr(decode_rae, "resolution", 0))
    if train_resolution != decode_resolution:
        raise ValueError(
            "decode_stage_1 resolution mismatch: "
            f"train={train_resolution}, decode={decode_resolution}."
        )

    train_aux_mode = str(getattr(train_rae, "decoder_aux_mode", "discard"))
    decode_aux_mode = str(getattr(decode_rae, "decoder_aux_mode", "discard"))
    decode_aux_source = getattr(decode_rae, "aux_token_source", None)
    if decode_aux_mode in {"prepend", "cross_attn"} and decode_aux_source != "learned":
        raise ValueError(
            "decode_stage_1 must be patch-only, pooled, or a learned implicit token decoder when used "
            "as a separate Stage-2 decode target. "
            f"Got decoder_aux_mode={decode_aux_mode!r}, aux_token_source={decode_aux_source!r}."
        )

    def _compare_optional_tensor(name: str, a, b):
        if (a is None) != (b is None):
            raise ValueError(
                f"decode_stage_1 {name} normalization mismatch: one model defines it and the other does not."
            )
        if a is not None and a.shape != b.shape:
            raise ValueError(
                f"decode_stage_1 {name} normalization shape mismatch: train={tuple(a.shape)}, decode={tuple(b.shape)}."
            )
        if a is not None and not torch.allclose(a.float(), b.float()):
            raise ValueError(
                f"decode_stage_1 {name} normalization mismatch: training and decode targets differ."
            )

    if bool(getattr(train_rae, "do_normalization", False)) != bool(getattr(decode_rae, "do_normalization", False)):
        raise ValueError("decode_stage_1 patch-latent normalization enablement does not match stage_1.")
    _compare_optional_tensor("latent_mean", getattr(train_rae, "latent_mean", None), getattr(decode_rae, "latent_mean", None))
    _compare_optional_tensor("latent_var", getattr(train_rae, "latent_var", None), getattr(decode_rae, "latent_var", None))

    if decode_aux_source == "learned" and getattr(decode_rae, "learned_input_aux_tokens", None) is None:
        raise ValueError(
            "decode_stage_1 is configured as a learned implicit aux decoder but has no learned_input_aux_tokens."
        )


def make_initial_sample_state(
    n: int,
    latent_size: Tuple[int, ...],
    aux_state_spec: AuxStateSpec,
    device: torch.device,
) -> Stage2State:
    z = torch.randn(n, *latent_size, device=device, dtype=torch.float32)
    if not aux_state_spec.enabled:
        return z

    if aux_state_spec.shape is None:
        raise ValueError("aux_state_spec.shape must be set when aux state is enabled.")

    aux = torch.randn(n, *aux_state_spec.shape, device=device, dtype=torch.float32)
    return (z, aux)


def duplicate_state_for_guidance(state: Stage2State) -> Stage2State:
    if torch.is_tensor(state):
        return torch.cat([state, state], dim=0)

    z, aux = state
    return (
        torch.cat([z, z], dim=0),
        torch.cat([aux, aux], dim=0),
    )


def split_guided_state(state: Stage2State) -> Stage2State:
    if torch.is_tensor(state):
        state, _ = state.chunk(2, dim=0)
        return state

    z, aux = state
    z, _ = z.chunk(2, dim=0)
    aux, _ = aux.chunk(2, dim=0)
    return (z, aux)


def state_float(state: Stage2State) -> Stage2State:
    if torch.is_tensor(state):
        return state.float()

    z, aux = state
    return z.float(), aux.float()


def state_batch_size(state: Stage2State) -> int:
    return state[0].shape[0] if isinstance(state, tuple) else state.shape[0]


def state_to_device(state: Stage2State, device: torch.device) -> Stage2State:
    if torch.is_tensor(state):
        return state.to(device, non_blocking=True)

    z, aux = state
    return z.to(device, non_blocking=True), aux.to(device, non_blocking=True)


def state_duplicate(state: Stage2State) -> Stage2State:
    return duplicate_state_for_guidance(state)


def state_first_half(state: Stage2State) -> Stage2State:
    return split_guided_state(state)


def state_cat(a: Stage2State, b: Stage2State) -> Stage2State:
    if torch.is_tensor(a):
        if not torch.is_tensor(b):
            raise TypeError(f"State type mismatch: {type(a)} vs {type(b)}")
        return torch.cat([a, b], dim=0)

    if not isinstance(b, tuple):
        raise TypeError(f"State type mismatch: {type(a)} vs {type(b)}")
    return (
        torch.cat([a[0], b[0]], dim=0),
        torch.cat([a[1], b[1]], dim=0),
    )


def decode_stage2_state(rae, state: Stage2State, use_aux_for_decode: bool = True) -> torch.Tensor:
    if torch.is_tensor(state):
        return rae.decode(state)

    z, aux = state
    if not use_aux_for_decode:
        decoder_aux_mode = str(getattr(rae, "decoder_aux_mode", "discard"))
        if decoder_aux_mode in {"prepend", "cross_attn"}:
            raise ValueError(
                "use_aux_for_decode=False requires a patch-only or pooled decoder target. "
                f"Got decoder_aux_mode={decoder_aux_mode!r}; provide a separate patch decoder as the decode target."
            )
        return rae.decode(z)

    if aux.ndim == 2:
        try:
            return rae.decode(z, cond=aux, cond_is_normalized=True)
        except TypeError:
            return rae.decode(z, cond=aux)

    if aux.ndim == 3:
        try:
            return rae.decode(z, aux_tokens=aux, aux_tokens_are_normalized=True)
        except TypeError:
            return rae.decode(z, aux_tokens=aux)

    raise ValueError(f"Unsupported auxiliary state shape for decode: {tuple(aux.shape)}")

"""
Edit-Friendly DDPM Inversion for Diffusion Autoencoders
======================================================

Implements the "edit-friendly" DDPM inversion from:
    Huberman-Spiegelglas, Kulikov, Michaeli:
    "An Edit Friendly DDPM Noise Space: Inversion and Manipulations", CVPR 2024.

...adapted for a diffusion autoencoder (Preechakul et al., "Diffusion
Autoencoders", CVPR 2022), where the noise-prediction model is additionally
conditioned on a semantic code x_sem alongside (x_t, t):

    eps_model(x_t, t, x_sem) -> eps_pred
"""

from typing import Callable, Dict, Optional, Tuple


import torch

EpsModel = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def make_beta_schedule(
    T: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    device: str = "cpu",
) -> torch.Tensor:
    """Linear beta schedule as in the original DDPM paper (Ho et al. 2020).
    If your model was trained with a different schedule (e.g. cosine), replace it
    here accordingly; the key requirement is that it matches the trained model."""
    return torch.linspace(beta_start, beta_end, T, device=device, dtype=torch.float64)


def _broadcast(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshapes a (B,) tensor to the form (B, 1, 1, 1, ...) matching x."""
    return t.view(-1, *([1] * (x.dim() - 1)))


class DiffusionSampler:
    """Stores all quantities derived from the betas. Timesteps are internally
    0-indexed (t_idx = 0 corresponds to the first diffusion step t = 1)."""

    def __init__(self, betas: torch.Tensor, timesteps: Optional[torch.Tensor] = None):
        self.T = betas.shape[0]
        self.betas = betas.double()
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.alpha_bars_prev = torch.cat(
            [
                torch.ones(1, dtype=torch.float64, device=betas.device),
                self.alpha_bars[:-1],
            ]
        )
        # Mapping to the ORIGINAL training timesteps (0..999).
        # Important for the network call if respacing was used.
        if timesteps is None:
            self.timesteps = torch.arange(
                self.T, device=betas.device, dtype=torch.long
            )
        else:
            self.timesteps = timesteps.to(device=betas.device, dtype=torch.long)

    @property
    def device(self) -> torch.device:
        return self.betas.device

    def _ensure_device(self, device: torch.device) -> None:
        if self.device != device:
            self.to(device)

    def to(self, device: torch.device) -> "DiffusionSampler":
        for name in ("betas", "alphas", "alpha_bars", "alpha_bars_prev", "timesteps"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    @classmethod
    def respaced(
        cls,
        full_schedule: "DiffusionSampler",
        num_steps: int,
        spacing: str = "trailing",
    ) -> "DiffusionSampler":
        """Creates a new schedule with num_steps < T steps whose marginal
        distributions q(x_t | x_0) exactly match the original."""
        T = full_schedule.T

        if spacing == "uniform":
            step_ratio = T / num_steps
            idx = (torch.arange(num_steps) * step_ratio).round().long()
        elif spacing == "trailing":
            # TODO seems wrong
            # more resolution near t=0 -> usually better for DDPM inversion
            idx = torch.linspace(T - 1, 0, num_steps + 1).round().long()[:-1].flip(0)
        else:
            raise ValueError(spacing)

        idx = idx.clamp(0, T - 1).to(full_schedule.alpha_bars.device)

        ab_sel = full_schedule.alpha_bars[idx]  # ᾱ'_i
        ab_prev_sel = torch.cat(
            [torch.ones(1, dtype=torch.float64, device=ab_sel.device), ab_sel[:-1]]
        )  # ᾱ'_{i-1}
        betas_new = 1.0 - ab_sel / ab_prev_sel  # β'_i

        return cls(betas_new, timesteps=idx)

    @torch.no_grad()
    def ddpm_edit_friendly_invert(
        self,
        x0: torch.Tensor,
        x_sem: torch.Tensor,
        eps_model: EpsModel,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Inverts x0 under conditioning on x_sem into the "edit-friendly" noise space.

        Returns:
            x_T: starting latent for the reverse process (torch.Tensor, same shape as x0)
            zs: Dict {t: z_t} for t=1..T, with the per-step noise maps extracted.
                 Together with x_sem and x_T, they exactly reproduce x0; with a modified
                 x_sem' they yield an edit.
        """
        self._ensure_device(x0.device)
        device = x0.device
        T = self.T
        B = x0.shape[0]

        xs = [None] * (T + 1)
        xs[0] = x0
        for t in range(1, T + 1):
            t_idx = torch.full((B,), t - 1, device=device, dtype=torch.long)
            noise = torch.randn(
                x0.shape, device=device, dtype=x0.dtype, generator=generator
            )
            xs[t], _ = q_sample(x0, t_idx, self, noise=noise)

        zs: Dict[int, torch.Tensor] = {}
        for t in range(T, 0, -1):
            t_idx = torch.full((B,), t - 1, device=device, dtype=torch.long)
            x_t = xs[t]
            x_prev = xs[t - 1]

            t_orig = self.timesteps[t_idx]
            eps_pred = eps_model(x_t, t_orig, cond=x_sem).pred
            x0_hat = predict_x0_from_eps(x_t, t_idx, eps_pred, self)
            mu = posterior_mean(x_t, x0_hat, t_idx, self)

            var = posterior_variance(t_idx, self)
            sigma = _broadcast(var, x_t).sqrt().to(x_t.dtype)

            zs[t] = (x_prev - mu) / sigma.clamp_min(1e-8)

        return xs[T], zs

    @torch.no_grad()
    def ddpm_edit_friendly_sample(
        self,
        x_T: torch.Tensor,
        zs: Dict[int, torch.Tensor],
        x_sem: torch.Tensor,
        eps_model: EpsModel,
        t_start: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Reverse process using the z_t extracted during inversion.

        - x_sem == original x_sem -> exact reconstruction of x0 (up to numerical
          rounding errors).
        - x_sem == modified x_sem' -> edit: structure remains (carried by the z_t),
          while semantic content changes according to x_sem'.
        """
        self._ensure_device(x_T.device)
        device = x_T.device
        B = x_T.shape[0]
        T = self.T
        t_start = T if t_start is None else t_start

        x_t = x_T
        for t in range(t_start, 0, -1):
            t_idx = torch.full((B,), t - 1, device=device, dtype=torch.long)

            t_orig = self.timesteps[t_idx]
            eps_pred = eps_model(x_t, t_orig, cond=x_sem).pred
            x0_hat = predict_x0_from_eps(x_t, t_idx, eps_pred, self)
            mu = posterior_mean(x_t, x0_hat, t_idx, self)

            var = posterior_variance(t_idx, self)
            sigma = _broadcast(var, x_t).sqrt().to(x_t.dtype)

            x_t = mu + sigma * zs[t]

        return x_t


def q_sample(
    x0: torch.Tensor,
    t_idx: torch.Tensor,
    sched: DiffusionSampler,
    noise: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """x_t ~ q(x_t | x0), using direct reparameterization (NO sequential process)."""
    if noise is None:
        noise = torch.randn_like(x0)
    ab = _broadcast(sched.alpha_bars[t_idx], x0).to(x0.dtype)
    x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
    return x_t, noise


def predict_x0_from_eps(
    x_t: torch.Tensor, t_idx: torch.Tensor, eps: torch.Tensor, sched: DiffusionSampler
) -> torch.Tensor:
    ab = _broadcast(sched.alpha_bars[t_idx], x_t).to(x_t.dtype)
    return (x_t - (1 - ab).sqrt() * eps) / ab.sqrt().clamp_min(1e-8)


def posterior_mean(
    x_t: torch.Tensor,
    x0_hat: torch.Tensor,
    t_idx: torch.Tensor,
    sched: DiffusionSampler,
) -> torch.Tensor:
    """mu_tilde_t(x_t, x0_hat): mean of the true posterior q(x_{t-1}|x_t, x0),
    evaluated here using the estimated x0_hat instead of the true x0. This is
    exactly what your trained DDPM sampling already does."""
    beta_t = _broadcast(sched.betas[t_idx], x_t).to(x_t.dtype)
    alpha_t = _broadcast(sched.alphas[t_idx], x_t).to(x_t.dtype)
    ab_t = _broadcast(sched.alpha_bars[t_idx], x_t).to(x_t.dtype)
    ab_prev = _broadcast(sched.alpha_bars_prev[t_idx], x_t).to(x_t.dtype)

    coef_x0 = ab_prev.sqrt() * beta_t / (1 - ab_t).clamp_min(1e-8)
    coef_xt = alpha_t.sqrt() * (1 - ab_prev) / (1 - ab_t).clamp_min(1e-8)
    return coef_x0 * x0_hat + coef_xt * x_t


def posterior_variance(
    t_idx: torch.Tensor, sched: DiffusionSampler, min_var_at_t1: Optional[float] = None
) -> torch.Tensor:
    """beta_tilde_t. When t_idx == 0 (i.e. t = 1, the final step to x0), the true
    posterior variance is exactly 0 (alpha_bar_prev = 1), which would make z_1
    undefined (division by zero). As is standard in the paper, a small non-zero
    value is used instead (default: beta_1)."""
    beta_t = sched.betas[t_idx]
    ab_t = sched.alpha_bars[t_idx]
    ab_prev = sched.alpha_bars_prev[t_idx]
    var = (1 - ab_prev) / (1 - ab_t).clamp_min(1e-8) * beta_t
    if min_var_at_t1 is None:
        min_var_at_t1 = float(sched.betas[0])
    fallback = torch.full_like(var, min_var_at_t1)
    return torch.where(t_idx == 0, fallback, var)

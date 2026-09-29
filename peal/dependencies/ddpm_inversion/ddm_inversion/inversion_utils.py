import torch
import os
from tqdm import tqdm
from peal.dependencies.PathLDM.ldm.modules.diffusionmodules.util import (
    make_ddim_timesteps,
)
from torchvision.transforms import ToPILImage


def load_real_image(folder="data/", img_name=None, idx=0, img_size=512, device="cuda"):
    from .utils import pil_to_tensor
    from PIL import Image
    from glob import glob

    if img_name is not None:
        path = os.path.join(folder, img_name)
    else:
        path = glob(folder + "*")[idx]

    img = Image.open(path).resize((img_size, img_size))

    img = pil_to_tensor(img).to(device)

    if img.shape[1] == 4:
        img = img[:, :3, :, :]
    return img


def mu_tilde(model, xt, x0, timestep):
    "mu_tilde(x_t, x_0) DDPM paper eq. 7"
    prev_timestep = timestep - model.scheduler.config.num_train_timesteps // len(
        model.scheduler.timesteps
    )
    alpha_prod_t_prev = (
        model.scheduler.alphas_cumprod[prev_timestep]
        if prev_timestep >= 0
        else model.scheduler.final_alpha_cumprod
    )
    alpha_t = model.scheduler.alphas[timestep]
    beta_t = 1 - alpha_t
    alpha_bar = model.scheduler.alphas_cumprod[timestep]
    return ((alpha_prod_t_prev**0.5 * beta_t) / (1 - alpha_bar)) * x0 + (
        (alpha_t**0.5 * (1 - alpha_prod_t_prev)) / (1 - alpha_bar)
    ) * xt


def sample_xts_from_x0(model, x0, num_inference_steps=50):
    """
    Samples from P(x_1:T|x_0)
    """
    # torch.manual_seed(43256465436)
    alpha_bar = model.scheduler.alphas_cumprod
    sqrt_one_minus_alpha_bar = (1 - alpha_bar) ** 0.5
    alphas = model.scheduler.alphas
    betas = 1 - alphas
    variance_noise_shape = (
        num_inference_steps,
        model.unet.in_channels,
        model.unet.sample_size,
        model.unet.sample_size,
    )

    timesteps = model.scheduler.timesteps.to(model.device)
    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    xts = torch.zeros([len(timesteps) + 1] + list(x0.shape)).to(x0.device)
    xts[0] = x0
    for t in reversed(timesteps):
        idx = len(timesteps) - t_to_idx[int(t)]
        xts[idx] = (
            x0 * (alpha_bar[t] ** 0.5)
            + torch.randn_like(x0) * sqrt_one_minus_alpha_bar[t]
        )

    return xts


def sample_xts_from_x0_pathldm(model, x0, timesteps):
    """PathLDM-compatible latent trajectory sampling for stochastic inversion."""
    sqrt_alpha_bar = model.sqrt_alphas_cumprod
    sqrt_one_minus_alpha_bar = model.sqrt_one_minus_alphas_cumprod

    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    xts = torch.zeros([len(timesteps) + 1] + list(x0.shape)).to(x0.device)
    xts[0] = x0
    for t in reversed(timesteps):
        noise = torch.randn_like(x0)
        idx = len(timesteps) - t_to_idx[int(t)]
        x = x0 * (sqrt_alpha_bar[t] ** 0.5) + sqrt_one_minus_alpha_bar[t] * noise
        xts[idx] = x
        t = torch.full((x0.shape[0],), int(t), device=x0.device, dtype=torch.long)
        x = model.q_sample(x0, t)
        xts[idx] = x
    return xts


def encode_text(model, prompts):
    text_input = model.tokenizer(
        prompts,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        text_encoding = model.text_encoder(text_input.input_ids.to(model.device))[0]

    return text_encoding


def forward_step(model, model_output, timestep, sample):
    next_timestep = min(
        model.scheduler.config.num_train_timesteps - 2,
        timestep
        + model.scheduler.config.num_train_timesteps
        // model.scheduler.num_inference_steps,
    )

    # 2. compute alphas, betas
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_original_sample = (
        sample - beta_prod_t ** (0.5) * model_output
    ) / alpha_prod_t ** (0.5)

    # 5. TODO: simple noising implementatiom
    next_sample = model.scheduler.add_noise(
        pred_original_sample, model_output, torch.LongTensor([next_timestep])
    )
    return next_sample


def forward_step_pathldm_builtin(model, x_start, timestep, noise=None):
    """Direct PathLDM forward noising using the built-in q_sample API."""
    if not torch.is_tensor(timestep):
        timestep = torch.full(
            (x_start.shape[0],), int(timestep), device=x_start.device, dtype=torch.long
        )
    return model.q_sample(x_start, timestep, noise=noise)


def forward_step_pathldm(model, model_output, timestep, sample, timesteps):
    """PathLDM-compatible forward step that mirrors the inversion math."""
    next_timestep = min(
        model.num_timesteps - 2,
        timestep + model.num_timesteps // len(timesteps),
    )
    alpha_prod_t = model.alphas_cumprod[timestep]
    beta_prod_t = 1 - alpha_prod_t
    pred_original_sample = (
        sample - beta_prod_t ** (0.5) * model_output
    ) / alpha_prod_t ** (0.5)
    return forward_step_pathldm_builtin(
        model, pred_original_sample, next_timestep, noise=model_output
    )


def get_variance(model, timestep):  # , prev_timestep):
    """Get variance for Diffusers pipeline."""
    """Get variance for Diffusers pipeline."""
    prev_timestep = timestep - model.scheduler.config.num_train_timesteps // len(
        model.scheduler.timesteps
    )
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = (
        model.scheduler.alphas_cumprod[prev_timestep]
        if prev_timestep >= 0
        else model.scheduler.final_alpha_cumprod
    )
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    return variance


def get_variance_pathldm(model, timestep, timesteps):
    """Get variance for PathLDM pipeline.
    Computes the DDPM posterior variance formula.

    """
    prev_timestep = timestep - model.num_timesteps // len(timesteps)

    alpha_prod_t = model.alphas_cumprod[timestep]
    alpha_prod_t_prev = (
        model.alphas_cumprod[prev_timestep]
        if prev_timestep >= 0
        else torch.tensor(1.0, device=model.device)
    )

    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    return variance


# def get_variance_pathldm(model, timestep, timesteps):
#     """Get variance for PathLDM pipeline.

#     Uses PathLDM's built-in posterior_variance if available,
#     otherwise computes it from alpha_cumprod values directly.
#     """
#     prev_timestep = timestep - model.num_timesteps // len(timesteps)

#     alpha_prod_t = model.alphas_cumprod[timestep]
#     alpha_prod_t_prev = (
#         model.alphas_cumprod[prev_timestep]
#         if prev_timestep >= 0
#         else model.final_alpha_cumprod
#     )

#     beta_prod_t = 1 - alpha_prod_t
#     beta_prod_t_prev = 1 - alpha_prod_t_prev
#     variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
#     return variance


def inversion_forward_process(
    model,
    x0,
    etas=None,
    prog_bar=False,
    prompt="",
    cfg_scale=3.5,
    num_inference_steps=50,
    eps=None,
    encoder_hidden_states=None,
):
    if not encoder_hidden_states is None:
        text_embeddings = encoder_hidden_states.to(model.unet.dtype)

    elif not prompt == "":
        text_embeddings = encode_text(model, prompt).to(model.unet.dtype)

    else:
        text_embeddings = None

    uncond_embedding = encode_text(model, [""] * x0.shape[0]).to(model.unet.dtype)
    timesteps = model.scheduler.timesteps.to(model.device)

    """variance_noise_shape = (
        num_inference_steps,
        model.unet.in_channels,
        model.unet.sample_size,
        model.unet.sample_size,
    )"""
    variance_noise_shape = [num_inference_steps] + list(x0.shape)

    if etas is None or (type(etas) in [int, float] and etas == 0):
        eta_is_zero = True
        zs = None
        xts = None  # Initialize to None; will be allocated if eta > 0

    else:
        eta_is_zero = False
        if type(etas) in [int, float]:
            # Use len(timesteps) to ensure consistency with the loop
            etas = [etas] * len(timesteps)
        xts = sample_xts_from_x0(model, x0, num_inference_steps=num_inference_steps)
        alpha_bar = model.scheduler.alphas_cumprod
        zs = torch.zeros(size=variance_noise_shape, device=model.device)

    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    xt = x0.to(model.unet.dtype)
    # op = tqdm(reversed(timesteps)) if prog_bar else reversed(timesteps)
    op = tqdm(timesteps) if prog_bar else timesteps

    for t in op:
        # idx = t_to_idx[int(t)]
        idx = num_inference_steps - t_to_idx[int(t)] - 1
        # 1. predict noise residual
        if not eta_is_zero:
            xt = xts[idx + 1]  # [None]
            # xt = xts_cycle[idx+1][None]

        with torch.no_grad():
            out = model.unet.forward(
                xt.to(model.unet.dtype),
                timestep=t,
                encoder_hidden_states=uncond_embedding,
            )
            if not text_embeddings is None:
                cond_out = model.unet.forward(
                    xt.to(model.unet.dtype),
                    timestep=t,
                    encoder_hidden_states=text_embeddings,
                )

        if not text_embeddings is None:
            ## classifier free guidance
            noise_pred = out.sample + cfg_scale * (cond_out.sample - out.sample)
        else:
            noise_pred = out.sample
        if eta_is_zero:
            # 2. compute more noisy image and set x_t -> x_t+1
            xt = forward_step(model, noise_pred, t, xt)

        else:
            # xtm1 =  xts[idx+1][None]
            xtm1 = xts[idx][None]
            # pred of x0
            pred_original_sample = (
                xt - (1 - alpha_bar[t]) ** 0.5 * noise_pred
            ) / alpha_bar[t] ** 0.5

            # direction to xt
            prev_timestep = t - model.scheduler.config.num_train_timesteps // len(
                model.scheduler.timesteps
            )
            alpha_prod_t_prev = (
                model.scheduler.alphas_cumprod[prev_timestep]
                if prev_timestep >= 0
                else model.scheduler.final_alpha_cumprod
            )

            variance = get_variance(model, t)
            pred_sample_direction = (1 - alpha_prod_t_prev - etas[idx] * variance) ** (
                0.5
            ) * noise_pred

            mu_xt = (
                alpha_prod_t_prev ** (0.5) * pred_original_sample
                + pred_sample_direction
            )

            z = (xtm1 - mu_xt) / (etas[idx] * variance**0.5 + 10e-8)
            zs[idx] = z

            # correction to avoid error accumulation
            xtm1 = mu_xt + (etas[idx] * variance**0.5) * z
            xts[idx] = xtm1

    if not zs is None:
        zs[0] = torch.zeros_like(zs[0])

    return xt, zs, xts


def reverse_step(model, model_output, timestep, sample, eta=0, variance_noise=None):
    # 1. get previous step value (=t-1)
    prev_timestep = timestep - model.scheduler.config.num_train_timesteps // len(
        model.scheduler.timesteps
    )
    # 2. compute alphas, betas
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = (
        model.scheduler.alphas_cumprod[prev_timestep]
        if prev_timestep >= 0
        else model.scheduler.final_alpha_cumprod
    )
    beta_prod_t = 1 - alpha_prod_t
    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_original_sample = (
        sample - beta_prod_t ** (0.5) * model_output
    ) / alpha_prod_t ** (0.5)
    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    # variance = self.scheduler._get_variance(timestep, prev_timestep)
    variance = get_variance(model, timestep)  # , prev_timestep)
    std_dev_t = eta * variance ** (0.5)
    # Take care of asymetric reverse process (asyrp)
    model_output_direction = model_output
    # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    # pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_output_direction
    pred_sample_direction = (1 - alpha_prod_t_prev - eta * variance) ** (
        0.5
    ) * model_output_direction
    # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    prev_sample = (
        alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
    )
    # 8. Add noice if eta > 0
    if eta > 0:
        if variance_noise is None:
            variance_noise = torch.randn(model_output.shape, device=model.device)
        sigma_z = eta * variance ** (0.5) * variance_noise
        prev_sample = prev_sample + sigma_z

    return prev_sample


def inversion_reverse_process(
    model,
    xT,
    etas=0,
    prompts="",
    cfg_scales=None,
    prog_bar=False,
    zs=None,
    controller=None,
    f=None,
    classifier=None,
    classifier_scale=0.0,
    asyrp=False,
    encoder_hidden_states=None,
):
    batch_size = len(prompts)

    cfg_scales_tensor = torch.Tensor(cfg_scales).view(-1, 1, 1, 1).to(model.device)

    uncond_embedding = encode_text(model, [""] * batch_size).to(model.unet.dtype)
    if not encoder_hidden_states is None:
        text_embeddings = encoder_hidden_states.to(model.unet.dtype)

    elif prompts == "":
        text_embeddings = uncond_embedding

    else:
        text_embeddings = encode_text(model, prompts).to(model.unet.dtype)

    if etas is None:
        etas = 0

    timesteps = model.scheduler.timesteps.to(model.device)

    if type(etas) in [int, float]:
        # Use len(timesteps) to ensure consistency with the loop
        etas = [etas] * len(timesteps)
    assert len(etas) == len(timesteps)

    xt = xT.expand(batch_size, -1, -1, -1).to(model.unet.dtype)
    op = tqdm(timesteps[-zs.shape[0] :]) if prog_bar else timesteps[-zs.shape[0] :]

    t_to_idx = {int(v): k for k, v in enumerate(timesteps[-zs.shape[0] :])}
    for t in op:
        idx = len(timesteps) - t_to_idx[int(t)] - (len(timesteps) - zs.shape[0] + 1)
        ## Unconditional embedding
        with torch.no_grad():
            uncond_out = model.unet.forward(
                xt.to(model.unet.dtype),
                timestep=t,
                encoder_hidden_states=uncond_embedding,
            )

        ## Conditional embedding
        with torch.no_grad():
            # TODO introduce option that only uses classifier
            if not classifier is None:
                if not f is None:
                    cond_out = model.unet.forward(
                        xt.to(model.unet.dtype),
                        timestep=t,
                        encoder_hidden_states=text_embeddings,
                        f=f,
                    )

                else:
                    x_noise = xt.detach().requires_grad_()
                    pred_noise = model.unet.forward(
                        xt, timestep=t, encoder_hidden_states=text_embeddings
                    )
                    alpha_prod_t = model.scheduler.alphas_cumprod[t]
                    beta_prod_t = 1 - alpha_prod_t
                    pred_x0 = (
                        xt.detach() - beta_prod_t ** (0.5) * pred_noise
                    ) / alpha_prod_t ** (0.5)
                    log_probs = classifier(pred_x0)
                    grad_classifier = torch.autograd.grad(
                        log_probs.sum(), x_noise, retain_graph=False
                    )[0].detach()
                    x_noise_adapted = (
                        xt.detach() - grad_classifier
                    )  # * (1 - a_t).sqrt()
                    cond_out = model.unet.forward(
                        x_noise_adapted.to(model.unet.dtype),
                        timestep=t,
                        encoder_hidden_states=text_embeddings,
                    )

            else:
                cond_out = model.unet.forward(
                    xt.to(model.unet.dtype),
                    timestep=t,
                    encoder_hidden_states=text_embeddings,
                )
        z = zs[idx] if not zs is None else None
        z = z.expand(batch_size, -1, -1, -1)
        ## classifier free guidance
        noise_pred = uncond_out.sample + cfg_scales_tensor * (
            cond_out.sample - uncond_out.sample
        )

        # 2. compute less noisy image and set x_t -> x_t-1
        xt = reverse_step(model, noise_pred, t, xt, eta=etas[idx], variance_noise=z)
        if controller is not None:
            xt = controller.step_callback(xt)

    return xt, zs


def inversion_forward_process_pathldm(
    model,
    x0,
    etas=None,
    prog_bar=False,
    prompt="",
    cfg_scale=0.0,
    num_inference_steps=50,
    eps=None,
    encoder_hidden_states=None,
    debug: bool = False,
    debug_dir: str = "debug_pathldm",
):
    """PathLDM-compatible version of inversion_forward_process.

    Adapts the SD-based inversion to work with PathLDM's API:
    - Uses model.apply_model() instead of model.unet.forward()
    - Uses model.get_learned_conditioning() instead of encode_text()
    - Handles PathLDM's different return structure
    """
    if not encoder_hidden_states is None:
        text_embeddings = encoder_hidden_states
    elif not prompt == "":
        text_embeddings = model.get_learned_conditioning(prompt)
    else:
        text_embeddings = None
    uncond_embedding = model.get_learned_conditioning([""] * x0.shape[0])
    timesteps = make_ddim_timesteps(
        "uniform",
        num_ddim_timesteps=num_inference_steps,
        num_ddpm_timesteps=model.num_timesteps,
    )
    timesteps = reversed(torch.tensor(timesteps).to(model.device))

    variance_noise_shape = [num_inference_steps] + list(x0.shape)
    sqrt_one_minus_alpha_bar = model.sqrt_one_minus_alphas_cumprod
    sqrt_alpha_bar = model.sqrt_alphas_cumprod
    # breakpoint()
    # Setup debugging
    if debug:
        import os
        from torchvision.transforms import ToPILImage

        os.makedirs(debug_dir, exist_ok=True)
        log_path = os.path.join(debug_dir, "forward_debug_log.txt")

        def _log(s):
            print(s)
            with open(log_path, "a") as f:
                f.write(s + "\n")

        def _save_decoded(tensor, name):
            # tensor is assumed to be latent-like for PathLDM first_stage; decode then scale to [0,1]
            try:
                with torch.no_grad():
                    dec = model.decode_first_stage(tensor.to(model.device))
                # normalize to [0,1] if likely in [-1,1]
                dec = ((dec / 2.0) + 0.5).clamp(0, 1)
                img = ToPILImage()(dec[0].detach().cpu())
                img.save(os.path.join(debug_dir, name))
            except Exception as e:
                _log(f"[DEBUG] Failed to save decoded {name}: {e}")

        # initial debug dump
        try:
            _log(
                f"[DEBUG] inversion_forward_process_pathldm start: x0 shape={x0.shape}, min={x0.min():.6f}, max={x0.max():.6f}, mean={x0.mean():.6f}"
            )
            _save_decoded(x0, "step_00_x0_decoded.png")
        except Exception as e:
            print(f"[DEBUG] init logging failed: {e}")
    if etas is None or (type(etas) in [int, float] and etas == 0):
        eta_is_zero = True
        zs = None
        xts = sample_xts_from_x0_pathldm(model, x0, timesteps=timesteps)
    else:
        eta_is_zero = False
        if type(etas) in [int, float]:
            # Use len(timesteps) to ensure consistency with the loop
            etas = [etas] * len(timesteps)
        if isinstance(etas, (int, float)):
            eta_values = [etas] * len(timesteps)
        else:
            eta_values = list(etas)
        xts = sample_xts_from_x0_pathldm(model, x0, timesteps=timesteps)
        alpha_bar = model.alphas_cumprod
        zs = torch.zeros(size=variance_noise_shape, device=model.device)

    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    xt = x0
    op = tqdm(timesteps) if prog_bar else timesteps
    # breakpoint()
    for t in op:
        t_idx = int(t)
        idx = num_inference_steps - t_to_idx[t_idx] - 1
        # 1. predict noise residual
        if not eta_is_zero:
            xt = xts[idx + 1]

        with torch.no_grad():
            # PathLDM uses apply_model and returns noise directly (not .sample)
            t_batch = torch.full((xt.size(0),), t_idx, device=xt.device).long()
            out_uncond = model.apply_model(xt, t_batch, uncond_embedding)
            if not text_embeddings is None:
                out_cond = model.apply_model(xt, t_batch, text_embeddings)

        if not text_embeddings is None:
            ## classifier free guidance
            noise_pred = out_uncond + cfg_scale * (out_cond - out_uncond)
        else:
            noise_pred = out_uncond

        if debug:
            try:
                _log(
                    f"[DEBUG] t={t_idx} idx={idx} xt shape={xt.shape} xt min={xt.min():.6f} max={xt.max():.6f} mean={xt.mean():.6f} std={xt.std():.6f}"
                )
            except Exception:
                _log(f"[DEBUG] t={t_idx} idx={idx} xt stats unavailable")

        if eta_is_zero:
            # 2. compute more noisy image and set x_t -> x_t+1
            xt = forward_step_pathldm(model, noise_pred, t_idx, xt, timesteps)
            if debug:
                try:
                    _save_decoded(xt, f"step_{t_idx:03d}_xt_decoded.png")
                except Exception as e:
                    _log(f"[DEBUG] save xt failed at t={t_idx}: {e}")
        else:
            xtm1 = xts[idx][None]
            # pred of x0
            pred_original_sample = (
                xt - sqrt_one_minus_alpha_bar[t_idx] * noise_pred
            ) / sqrt_alpha_bar[t_idx]

            # direction to xt
            prev_timestep = t_idx - model.num_timesteps // len(timesteps)
            alpha_prod_t_prev = (
                alpha_bar[prev_timestep]
                if prev_timestep >= 0
                else torch.tensor(1.0, device=model.device)
            )

            variance = get_variance_pathldm(model, t_idx, timesteps)
            # protect against small negative numerical values inside sqrt
            sqrt_arg = 1 - alpha_prod_t_prev - eta_values[idx] * variance
            if debug:
                try:
                    _log(
                        f"[DEBUG] t={t_idx} sqrt_arg={float(sqrt_arg):.6f} variance={float(variance):.6f} alpha_prev={float(alpha_prod_t_prev):.6f} eta={float(eta_values[idx]):.6f}"
                    )
                except Exception:
                    _log(f"[DEBUG] t={t_idx} sqrt_arg stats unavailable")
            # clamp to >=0 to avoid NaNs from sqrt of tiny negative numbers
            if isinstance(sqrt_arg, torch.Tensor):
                sqrt_arg = torch.clamp(sqrt_arg, min=0.0)
            else:
                sqrt_arg = max(sqrt_arg, 0.0)
            pred_sample_direction = (sqrt_arg**0.5) * noise_pred

            mu_xt = (
                alpha_prod_t_prev ** (0.5) * pred_original_sample
                + pred_sample_direction
            )

            denom = eta_values[idx] * variance**0.5
            if debug:
                try:
                    _log(f"[DEBUG] t={t_idx} denom={float(denom):.6f}")
                except Exception:
                    _log(f"[DEBUG] t={t_idx} denom stats unavailable")

            # If the variance collapses to zero (commonly at the first step),
            # fall back to the deterministic update instead of dividing by zero.
            if torch.is_tensor(denom):
                denom_ok = torch.all(torch.abs(denom) > 1e-8).item()
            else:
                denom_ok = abs(denom) > 1e-8

            if denom_ok:
                z = (xtm1 - mu_xt) / denom
                xtm1 = mu_xt + denom * z
            else:
                z = torch.zeros_like(xtm1)
                xtm1 = mu_xt

            zs[idx] = z
            xts[idx] = xtm1
            # breakpoint()
            if debug:
                try:
                    _log(
                        f"[DEBUG] produced xts[{idx}] stats min={xts[idx].min():.6f} max={xts[idx].max():.6f} mean={xts[idx].mean():.6f}"
                    )
                    _save_decoded(
                        xts[idx], f"step_{t_idx:03d}_xts_idx{idx}_decoded.png"
                    )
                except Exception as e:
                    _log(f"[DEBUG] save xts failed at idx={idx}: {e}")
    if not zs is None:
        zs[0] = torch.zeros_like(zs[0])

    return xt, zs, xts


def reverse_step_pathldm(
    model, model_output, timestep, sample, timesteps, eta=0, variance_noise=None
):
    """PathLDM-compatible reverse DDPM step.

    Uses PathLDM's direct scheduler fields on the model instead of the
    Diffusers UNet / scheduler return conventions.
    """
    prev_timestep = timestep - model.num_timesteps // len(timesteps)

    alpha_prod_t = model.alphas_cumprod[timestep]
    alpha_prod_t_prev = (
        model.alphas_cumprod[prev_timestep]
        if prev_timestep >= 0
        else torch.tensor(1.0, device=model.device)
    )
    beta_prod_t = 1 - alpha_prod_t

    pred_original_sample = (
        sample - beta_prod_t ** (0.5) * model_output
    ) / alpha_prod_t ** (0.5)

    variance = get_variance_pathldm(model, timestep, timesteps)
    model_output_direction = model_output
    pred_sample_direction = (1 - alpha_prod_t_prev - eta * variance) ** (
        0.5
    ) * model_output_direction

    prev_sample = (
        alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
    )

    if eta > 0:
        if variance_noise is None:
            variance_noise = torch.randn(model_output.shape, device=model.device)
        sigma_z = eta * variance ** (0.5) * variance_noise
        prev_sample = prev_sample + sigma_z

    return prev_sample


def inversion_reverse_process_pathldm(
    model,
    xT,
    etas=0,
    prompts="",
    cfg_scales=None,
    prog_bar=False,
    zs=None,
    controller=None,
    num_inference_steps: int = 50,
    f=None,
    classifier=None,
    classifier_scale=0.0,
    asyrp=False,
    encoder_hidden_states: torch.Tensor = None,
    debug: bool = False,
    debug_dir: str = "debug_pathldm",
):
    """PathLDM-compatible reverse inversion process.

    This mirrors `inversion_reverse_process` but uses PathLDM's API:
    - `model.get_learned_conditioning(...)` for text embeddings
    - `model.apply_model(...)` for noise prediction
    - direct model-level scheduler fields for DDPM math
    """
    batch_size = len(prompts)

    # Setup debugging
    if debug:
        import os
        from torchvision.transforms import ToPILImage

        os.makedirs(debug_dir, exist_ok=True)
        log_path = os.path.join(debug_dir, "reverse_debug_log.txt")

        def _log(s):
            print(s)
            with open(log_path, "a") as f:
                f.write(s + "\n")

        def _save_decoded(tensor, name):
            # tensor is assumed to be latent-like for PathLDM first_stage; decode then scale to [0,1]
            try:
                with torch.no_grad():
                    dec = model.decode_first_stage(tensor.to(model.device))
                # normalize to [0,1] if likely in [-1,1]
                dec = ((dec / 2.0) + 0.5).clamp(0, 1)
                img = ToPILImage()(dec[0].detach().cpu())
                img.save(os.path.join(debug_dir, name))
            except Exception as e:
                _log(f"[DEBUG] Failed to save decoded {name}: {e}")

        # initial debug dump
        try:
            _log(
                f"[DEBUG] inversion_reverse_process_pathldm start: xT shape={xT.shape}, min={xT.min():.6f}, max={xT.max():.6f}, mean={xT.mean():.6f}"
            )
            _save_decoded(xT, "reverse_00_xT_start.png")
        except Exception as e:
            print(f"[DEBUG] init logging failed: {e}")

    cfg_scales_tensor = torch.Tensor(cfg_scales).view(-1, 1, 1, 1).to(model.device)

    uncond_embedding = model.get_learned_conditioning([""] * batch_size)
    if encoder_hidden_states is not None:
        text_embeddings = encoder_hidden_states
    elif prompts == "":
        text_embeddings = uncond_embedding
    else:
        text_embeddings = model.get_learned_conditioning(prompts)

    if etas is None:
        etas = 0

    timesteps = make_ddim_timesteps(
        "uniform",
        num_ddim_timesteps=num_inference_steps,
        num_ddpm_timesteps=model.num_timesteps,
    )
    timesteps = reversed(torch.tensor(timesteps).to(model.device))

    if isinstance(etas, (int, float)):
        eta_values = [etas] * len(timesteps)
    else:
        eta_values = list(etas)
    assert len(eta_values) == len(timesteps)

    xt = xT.expand(batch_size, -1, -1, -1)
    if zs is None:
        reverse_timesteps = timesteps
        zs = [None] * len(timesteps)
    else:
        reverse_timesteps = timesteps[-zs.shape[0] :]

    op = tqdm(reverse_timesteps) if prog_bar else reverse_timesteps

    t_to_idx = {int(v): k for k, v in enumerate(reverse_timesteps)}
    for t in op:
        t_idx = int(t)
        idx = (
            len(timesteps)
            - t_to_idx[t_idx]
            - (len(timesteps) - len(reverse_timesteps) + 1)
        )
        t_batch = torch.full((xt.size(0),), t_idx, device=xt.device).long()

        if debug:
            try:
                _log(f"[DEBUG] === Reverse Step === t={t_idx} idx={idx}")
                _log(
                    f"[DEBUG] xt_before shape={xt.shape} min={xt.min():.6f} max={xt.max():.6f} mean={xt.mean():.6f} std={xt.std():.6f}"
                )
            except Exception:
                _log(f"[DEBUG] t={t_idx} idx={idx} xt_before stats unavailable")

        with torch.no_grad():
            uncond_out = model.apply_model(xt, t_batch, uncond_embedding)

        with torch.no_grad():
            if classifier is not None:
                if f is not None:
                    cond_out = model.apply_model(xt, t_batch, text_embeddings, f=f)
                else:
                    x_noise = xt.detach().requires_grad_()
                    pred_noise = model.apply_model(xt, t_batch, text_embeddings)
                    alpha_prod_t = model.alphas_cumprod[t_batch]
                    beta_prod_t = 1 - alpha_prod_t
                    pred_x0 = (
                        xt.detach() - beta_prod_t ** (0.5) * pred_noise
                    ) / alpha_prod_t ** (0.5)
                    log_probs = classifier(pred_x0)
                    grad_classifier = torch.autograd.grad(
                        log_probs.sum(), x_noise, retain_graph=False
                    )[0].detach()
                    x_noise_adapted = xt.detach() - grad_classifier
                    cond_out = model.apply_model(
                        x_noise_adapted, t_batch, text_embeddings
                    )
            else:
                cond_out = model.apply_model(xt, t_batch, text_embeddings)

        z = zs[idx] if zs is not None else None
        if z is not None:
            z = z.expand(batch_size, -1, -1, -1)

        noise_pred = uncond_out + cfg_scales_tensor * (cond_out - uncond_out)

        if debug:
            try:
                _log(
                    f"[DEBUG] uncond_out min={uncond_out.min():.6f} max={uncond_out.max():.6f}"
                )
                _log(
                    f"[DEBUG] cond_out min={cond_out.min():.6f} max={cond_out.max():.6f}"
                )
                _log(
                    f"[DEBUG] noise_pred (after guidance) min={noise_pred.min():.6f} max={noise_pred.max():.6f} mean={noise_pred.mean():.6f}"
                )
                if z is not None:
                    _log(
                        f"[DEBUG] variance_noise z min={z.min():.6f} max={z.max():.6f}"
                    )
            except Exception:
                _log(f"[DEBUG] t={t_idx} noise stats unavailable")

        xt = reverse_step_pathldm(
            model,
            noise_pred,
            t_idx,
            xt,
            timesteps,
            eta=eta_values[idx],
            variance_noise=z,
        )
        if debug:
            try:
                _log(
                    f"[DEBUG] xt_after shape={xt.shape} min={xt.min():.6f} max={xt.max():.6f} mean={xt.mean():.6f} std={xt.std():.6f}"
                )
                _save_decoded(
                    xt,
                    f"reverse_step_{len(reverse_timesteps) - list(reverse_timesteps).index(t):03d}_t{t_idx:03d}.png",
                )
            except Exception as e:
                _log(f"[DEBUG] save xt failed at t={t_idx}: {e}")

        if controller is not None:
            xt = controller.step_callback(xt)

    if debug:
        try:
            _log(f"[DEBUG] === Reverse Process Complete ===")
            _log(
                f"[DEBUG] Final xt shape={xt.shape} min={xt.min():.6f} max={xt.max():.6f} mean={xt.mean():.6f} std={xt.std():.6f}"
            )
            _save_decoded(xt, "reverse_final_output.png")
            _log(f"[DEBUG] All debug outputs saved to {debug_dir}")
        except Exception as e:
            print(f"[DEBUG] final logging failed: {e}")

    return xt, zs

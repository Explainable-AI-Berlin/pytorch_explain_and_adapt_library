import wandb
import torch
from functools import partial
import os
import json

def init_wandb(cfg):
    return wandb.init(project=cfg["wandb_project"], name=cfg["name"], config=cfg, reinit=True)

def log_wandb(output, step, wandb_run, index=None):
    log_dict = {
        k: v.item() if isinstance(v, torch.Tensor) and v.dim() == 0 else v
        for k, v in output.items() if isinstance(v, (int, float)) or 
        (isinstance(v, torch.Tensor) and v.dim() == 0)
    }

    if index is not None:
        log_dict = {f"{k}_{index}": v for k, v in log_dict.items()}

    wandb_run.log(log_dict, step=step)

# Hooks for model performance evaluation
def reconstr_hook(activation, hook, sae_out):
    return sae_out

def zero_abl_hook(activation, hook):
    return torch.zeros_like(activation)

def mean_abl_hook(activation, hook):
    return activation.mean([0, 1]).expand_as(activation)

@torch.no_grad()
def log_model_performance(wandb_run, step, model, activations_store, sae, index=None, batch_tokens=None):
    if batch_tokens is None:
        batch_tokens = activations_store.get_batch_tokens()[:sae.config["batch_size"] // sae.config["seq_len"]]
    batch = activations_store.get_activations(batch_tokens).reshape(-1, sae.config["act_size"])

    sae_output = sae(batch)["sae_out"].reshape(batch_tokens.shape[0], batch_tokens.shape[1], -1)

    original_loss = model(batch_tokens, return_type="loss").item()
    reconstr_loss = model.run_with_hooks(
        batch_tokens,

        
        fwd_hooks=[(sae.config["hook_point"], partial(reconstr_hook, sae_out=sae_output))],
        return_type="loss",
    ).item()
    zero_loss = model.run_with_hooks(
        batch_tokens,
        fwd_hooks=[(sae.config["hook_point"], zero_abl_hook)],
        return_type="loss",
    ).item()
    mean_loss = model.run_with_hooks(
        batch_tokens,
        fwd_hooks=[(sae.config["hook_point"], mean_abl_hook)],
        return_type="loss",
    ).item()

    ce_degradation = original_loss - reconstr_loss
    zero_degradation = original_loss - zero_loss
    mean_degradation = original_loss - mean_loss

    log_dict = {
        "performance/ce_degradation": ce_degradation,
        "performance/recovery_from_zero": (reconstr_loss - zero_loss) / zero_degradation,
        "performance/recovery_from_mean": (reconstr_loss - mean_loss) / mean_degradation,
    }

    if index is not None:
        log_dict = {f"{k}_{index}": v for k, v in log_dict.items()}

    wandb_run.log(log_dict, step=step)

def save_checkpoint(wandb_run, sae, cfg, step):
    save_dir = f"checkpoints/{cfg['name']}_{step}"
    os.makedirs(save_dir, exist_ok=True)

    # Save model state
    sae_path = os.path.join(save_dir, "sae.pt")
    torch.save(sae.state_dict(), sae_path)

    # Prepare config for JSON serialization
    json_safe_cfg = {}
    for key, value in cfg.items():
        if isinstance(value, (int, float, str, bool, type(None))):
            json_safe_cfg[key] = value
        elif isinstance(value, (torch.dtype, type)):
            json_safe_cfg[key] = str(value)
        else:
            json_safe_cfg[key] = str(value)

    # Save config
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(json_safe_cfg, f, indent=4)

    # Create and log artifact
    artifact = wandb.Artifact(
        name=f"{cfg['name']}_{step}",
        type="model",
        description=f"Model checkpoint at step {step}",
    )
    artifact.add_file(sae_path)
    artifact.add_file(config_path)
    wandb_run.log_artifact(artifact)

    print(f"Model and config saved as artifact at step {step}")


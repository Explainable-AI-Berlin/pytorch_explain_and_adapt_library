#%%
import torch
import torch.nn.functional as F
from sae_lens import SAE  
import wandb
import json
import os

def load_sae_from_wandb(artifact_name, sae_class):
    # Initialize wandb
    api = wandb.Api()

    # Download the artifact
    artifact = api.artifact(artifact_name)
    artifact_dir = artifact.download()

    # Load the configuration
    config_path = os.path.join(artifact_dir, "config.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    # Convert string representations back to torch.dtype
    if "dtype" in cfg:
        cfg["dtype"] = getattr(torch, cfg["dtype"].split(".")[-1])

    # Convert group_sizes back to a list if it's a string
    if isinstance(cfg['group_sizes'], str):
        cfg['group_sizes'] = json.loads(cfg['group_sizes'])

    if isinstance(cfg["top_k_matryoshka"], str):
        cfg["top_k_matryoshka"] = json.loads(cfg["top_k_matryoshka"])

    sae = sae_class(cfg)

    # Load the state dict
    state_dict_path = os.path.join(artifact_dir, "sae.pt")
    state_dict = torch.load(state_dict_path, map_location=cfg["device"])
    sae.load_state_dict(state_dict)

    return sae, cfg

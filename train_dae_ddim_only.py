"""Train only the DDIM stage of the PEAL DiffusionAutoencoder.

This does the same as the first part of DiffusionAutoencoder.train_model
(peal/generators/diffusion_autoencoder.py): it writes config.yaml with
is_loaded=True and trains square64_ddim until total_samples. It skips the
"infer" eval pass and the latent-DPM stage. The DAEdistill explainer does not
use them, and the January 2026 Camelyon DAE checkpoint did not have them.

Usage: python train_dae_ddim_only.py --config <generator yaml>

Environment: needs lightning 2.1.4 and lmdb, which the python containers lack. The Camelyon
seed DAE was trained with them from /home/space/datasets/peal_ahmed/didae_camelyon17/pyenv_lightning,
put first on PYTHONPATH (see reproduction_scripts/reproduce_didae_results.sh). Config used:
configs/didae_experiments/generators/camelyon_diffusion_autoencoder_seeds.yaml.

Lightning 2.1.x fix: the diffae ModelCheckpoint monitors "FID", which training
never logs, so no top-k file is written. From the second save on, Lightning then
links last.ckpt to itself: it deletes the real file and leaves a self-link. The
patch below makes every "last" save a real save.
"""
import argparse
import os
from pathlib import Path

from lightning.pytorch.callbacks import ModelCheckpoint

_orig_save_last_checkpoint = ModelCheckpoint._save_last_checkpoint


def _save_last_checkpoint_as_file(self, trainer, monitor_candidates):
    last_saved = self._last_checkpoint_saved
    if last_saved and os.path.basename(last_saved).startswith(self.CHECKPOINT_NAME_LAST):
        self._last_checkpoint_saved = ""
    return _orig_save_last_checkpoint(self, trainer, monitor_candidates)


ModelCheckpoint._save_last_checkpoint = _save_last_checkpoint_as_file

from peal.global_utils import load_yaml_config, save_yaml_config, set_random_seed
from peal.generators.generator_factory import get_generator
from peal.dependencies.diffusion_regression_counterfactuals.src.related_work.diffae.templates_latent import (
    square64_autoenc,
    train,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    config.is_loaded = False
    set_random_seed(config.seed)
    print(f"seed={config.seed} base_path={config.base_path} total_samples={config.total_samples}")

    generator = get_generator(config)
    generator.generator_dataset.return_dict = True
    generator.generator_dataset.idx_enabled = True
    Path(config.base_path).mkdir(parents=True, exist_ok=True)

    generator.config.is_loaded = True
    save_yaml_config(generator.config, os.path.join(config.base_path, "config.yaml"))

    conf = square64_autoenc()
    generator.adjust_config(conf)
    train(conf)

    for name in ("last.ckpt", os.path.join("ema", "last.ckpt")):
        path = os.path.join(config.base_path, "square64_ddim", name)
        ok = os.path.isfile(path) and not os.path.islink(path)
        size = os.path.getsize(path) if ok else -1
        print(f"CHECKPOINT {path} real_file={ok} bytes={size}")
        if not ok:
            raise RuntimeError(f"{path} is not a real checkpoint file")
    print("DDIM_TRAINING_DONE")


if __name__ == "__main__":
    main()

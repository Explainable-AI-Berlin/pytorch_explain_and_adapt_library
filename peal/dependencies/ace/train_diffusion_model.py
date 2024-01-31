"""
Train a diffusion model on images.
"""

import argparse
import os

from peal.dependencies.ace.guided_diffusion import logger
from peal.dependencies.ace.guided_diffusion import dist_util
from peal.dependencies.ace.guided_diffusion.image_datasets import load_data_celeba
from peal.dependencies.ace.guided_diffusion.resample import create_named_schedule_sampler
from peal.dependencies.ace.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from peal.dependencies.ace.guided_diffusion.train_util import TrainLoop

from peal.data.dataset_factory import get_datasets
from peal.data.dataloaders import get_dataloader
from peal.global_utils import load_yaml_config
from peal.configs.data.data_config import DataConfig


def main():
    args = create_argparser().parse_args()

    if not args.generator_config is None:
        args = load_yaml_config(args.generator_config)

    dist_util.setup_dist(args.gpus)
    logger.configure(dir=os.path.join(args.base_path, "output"))

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        num_classes=40,
        multiclass=True,
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    if args.data is None:
        data = load_data_celeba(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
            use_hdf5=args.use_hdf5,
            HQ=args.use_celeba_HQ,
        )

    else:
        args.data = load_yaml_config(args.data, DataConfig)
        dataset, _, _ = get_datasets(args.data)
        dataset.return_dict = True
        args.train_batch_size = args.batch_size
        args.steps_per_epoch = args.max_train_steps
        dataloader = get_dataloader(dataset, training_config=args)
        data = iter(dataloader)

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        model_dir=args.base_path,
    ).run_loop(config=args)


def create_argparser():
    defaults = dict(
        data_dir="/data/chercheurs/jeanner211/DATASETS/celeba",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        output_path="/data/chercheurs/jeanner211/RESULTS/DCF-CelebA/ddpm",
        gpus="",
        use_hdf5=False,
        use_celeba_HQ=False,
        generator_config=None,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

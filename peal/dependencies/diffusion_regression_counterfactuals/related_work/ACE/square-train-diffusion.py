"""
Train a diffusion model on images.
"""

import argparse

from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
from diff_cf_ir.squares_dataset import SquaresDataset
from torch.utils.data import DataLoader


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist(args.gpus)
    logger.configure(dir=args.output_path)

    logger.log("creating model and diffusion...")
    # No Class Condition
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    print("Device: ", dist_util.dev())
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating retinaMNIST loader...")

    dataset = SquaresDataset(root=args.data_dir, get_mode="ace")

    def infinite_iterator():
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
        )
        while True:
            yield from dataloader

    data = infinite_iterator()

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
    ).run_loop()


def create_argparser():
    defaults = dict(
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=32,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=1_000,
        save_interval=10_000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        gpus="",
        use_hdf5=False,
        use_celeba_HQ=False,
    )
    defaults.update(model_and_diffusion_defaults())

    # Square settings
    defaults.update(
        {
            "image_size": 64,
        }
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument(
        "data_dir",
        type=str,
        help="Path to the train square dataset directory.",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Where to save the output models.",
    )
    return parser


if __name__ == "__main__":
    main()
    print("main done.")

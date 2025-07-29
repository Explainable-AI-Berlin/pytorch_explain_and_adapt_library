import os
import argparse

from os import path as osp

import torch

from tqdm import tqdm


# Diffusion Model imports
from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)

# core imports
from core.utils import generate_mask
from core.attacks_and_models import JointClassifierDDPM, get_attack

import matplotlib

matplotlib.use("Agg")  # to disable display

from diff_cf_ir.counterfactuals import (
    CFResult,
    save_cf_results,
    update_results_true_latents,
)
from diff_cf_ir.file_utils import (
    assert_paths_exist,
    create_result_dir,
    deterministic_run,
    dump_args,
)
from diff_cf_ir.image_folder_dataset import default_transforms
from diff_cf_ir.models import load_resnet
from diff_cf_ir.squares_dataset import (
    get_experiment_targets,
    SquaresDataset,
    inner_square_color,
)


# =======================================================
# =======================================================
# Functions
# =======================================================
# =======================================================


def create_args():
    defaults = dict(
        clip_denoised=True,  # Clipping noise
        batch_size=1,  # Batch size
        gpu="0",  # GPU index, should only be 1 gpu
        save_images=False,  # Saving all images
        num_samples=10_000,  # useful to sample few examples
        cudnn_deterministic=False,  # setting this to true will slow the computation time but will have identic results when using the checkpoint backwards
        # path args
        model_path="",  # DDPM weights path
        exp_name="exp",  # Experiment name (will store the results at Output/Results/exp_name)
        # attack args
        seed=0,  # Random seed
        attack_method="PGD",  # Attack method (currently 'PGD', 'C&W', 'GD' and 'None' supported)
        attack_iterations=100,  # Attack iterations updates
        attack_epsilon=255,  # L inf epsilon bound (will be devided by 255)
        attack_step=1.0,  # Attack update step (will be devided by 255)
        attack_joint=True,  # Set to false to generate adversarial attacks
        attack_joint_checkpoint=False,  # use checkpoint method for backward. Beware, this will substancially slow down the CE generation!
        attack_checkpoint_backward_steps=1,  # number of DDPM iterations per backward process. We highly recommend have a larger backward steps than batch size (e.g have 2 backward steps and batch size of 1 than 1 backward step and batch size 2)
        attack_joint_shortcut=False,  # Use DiME shortcut to transfer gradients. We do not recommend it.
        # dist args
        dist_l1=0.0,  # l1 scaling factor
        dist_l2=0.0,  # l2 scaling factor
        dist_schedule="none",  # schedule for the distance loss. We did not used any for our results
        # filtering args
        sampling_time_fraction=0.1,  # fraction of noise steps (e.g. 0.1 for 1000 smpling steps would be 100 out of 1000)
        sampling_stochastic=True,  # Set to False to remove the noise when sampling
        # post processing
        sampling_inpaint=0.15,  # Inpainting threshold
        sampling_dilation=15,  # Dilation size for the mask generation
        # query and target label
        # dataset
        image_size=256,  # Dataset image size
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    # Regression args
    parser.add_argument(
        "--rmodel_path",
        type=str,
        required=True,
        help="Path to the regression model.",
    )

    def parse_target(value):
        if value.strip().lower() == "mirror":
            return "mirror"
        else:
            return float(value)

    parser.add_argument(
        "--target",
        type=parse_target,
        required=True,
        help="Target class for the attack, either 'mirror' for mirror experiment or a float value.",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        required=False,
        help="Confidence to goal to stop the attack",
        default=0.05,
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help=(
            "Path to the samples folder. "
            "The folder should contain images and an optional data.csv file with two "
            "fields for filename and age."
            " If available, it will update true labels in the result dict."
        ),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        default="ace_square_results",
        help="Directory to save the results.",
    )

    args = parser.parse_args()
    return args


# =======================================================
# =======================================================
# Custom functions
# =======================================================
# =======================================================


@torch.no_grad()
def filter_fn(
    diffusion,
    attack,
    model,
    steps,
    x,
    stochastic,
    target,
    inpaint,
    dilation,
):

    indices = list(range(steps))[::-1]

    # 1. Generate pre-explanation
    with torch.enable_grad():
        pe, success, steps_done = attack.perturb(x, target)

    # 2. Inpainting: generates masks
    mask, dil_mask = generate_mask(x, pe, dilation)
    boolmask = (dil_mask < inpaint).float()

    ce = (pe.detach() - 0.5) / 0.5
    orig = (x.detach() - 0.5) / 0.5
    noise_fn = torch.randn_like if stochastic else torch.zeros_like

    for idx, t in enumerate(indices):

        # filter the with the diffusion model
        t = torch.tensor([t] * ce.size(0), device=ce.device)

        if idx == 0:
            ce = diffusion.q_sample(ce, t, noise=noise_fn(ce))
            noise_x = ce.clone().detach()

        if inpaint != 0:
            ce = ce * (1 - boolmask) + boolmask * diffusion.q_sample(
                orig, t, noise=noise_fn(ce)
            )

        out = diffusion.p_mean_variance(model, ce, t, clip_denoised=True)

        ce = out["mean"]

        if stochastic and (idx != (steps - 1)):
            noise = torch.randn_like(ce)
            ce += torch.exp(0.5 * out["log_variance"]) * noise

    ce = ce * (1 - boolmask) + boolmask * orig
    ce = (ce * 0.5) + 0.5
    ce = ce.clamp(0, 1)
    noise_x = ((noise_x * 0.5) + 0.5).clamp(0, 1)

    return ce, pe, noise_x, mask, success, steps_done


@torch.no_grad()
def to_ddpm_colorspace(x):
    return (x - 0.5) / 0.5


@torch.no_grad()
def to_normal_colorspace(x):
    return x * 0.5 + 0.5


# =======================================================
# =======================================================
# Main
# =======================================================
# =======================================================


def load_model(args):
    model, respaced_diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    return model, respaced_diffusion


def get_data(args):
    compose = default_transforms(args.image_size)
    dataset = SquaresDataset(root=args.image_folder, transform=compose, mask_mode=True)
    num_samples = (
        len(dataset)
        if args.num_samples is None
        else min(args.num_samples, len(dataset))
    )
    dataset = torch.utils.data.Subset(dataset, range(num_samples))

    return dataset


def main() -> None:
    args = create_args()

    # if args.merge_chunks:
    #     merge_all_chunks(args.chunks, args.output_path, args.exp_name)
    #     return

    respaced_steps = int(args.sampling_time_fraction * int(args.timestep_respacing))
    normal_steps = int(args.sampling_time_fraction * int(args.diffusion_steps))

    print("Using", respaced_steps, "respaced steps and", normal_steps, "normal steps")

    args.respaced_steps = respaced_steps
    args.normal_steps = normal_steps

    # ========================================
    # Setup the environment and results
    # ========================================

    deterministic_run(args.seed)
    assert_paths_exist(
        [
            args.model_path,
            args.rmodel_path,
            args.image_folder,
        ]
    )
    result_dir = create_result_dir(osp.join(args.output_path))
    dump_args(args, result_dir)

    # ========================================
    # load models
    # ========================================

    print("Loading Model and diffusion model")
    # respaced diffusion has the respaced strategy
    model, respaced_diffusion = load_model(args)

    print("Loading Regressor")
    classifier = load_resnet(args.rmodel_path)
    classifier.to(dist_util.dev()).eval()

    if args.attack_joint and not (
        args.attack_joint_checkpoint or args.attack_joint_shortcut
    ):
        joint_classifier = JointClassifierDDPM(
            classifier=classifier,
            ddpm=model,
            diffusion=respaced_diffusion,
            steps=respaced_steps,
            stochastic=args.sampling_stochastic,
        )
        joint_classifier.eval()

    # ========================================
    # load attack
    # ========================================

    def get_dist_fn():

        any_loss = False
        if args.dist_l2 != 0.0:
            l2_loss = (
                lambda x, x_adv: args.dist_l2
                * torch.linalg.norm((x - x_adv).view(x.size(0), -1), dim=1).sum()
            )
            any_loss = True

        if args.dist_l1 != 0.0:
            l1_loss = lambda x, x_adv: args.dist_l1 * (x - x_adv).abs().sum()
            any_loss = True

        if not any_loss:
            return None

        def dist_fn(x, x_adv):
            loss = 0
            if args.dist_l2 != 0.0:
                loss += l2_loss(x, x_adv)
            if args.dist_l1 != 0.0:
                loss += l1_loss(x, x_adv)
            return loss

        return dist_fn

    dist_fn = get_dist_fn()

    attack_args = {
        "predict": (
            joint_classifier
            if args.attack_joint
            and not (args.attack_joint_checkpoint or args.attack_joint_shortcut)
            else classifier
        ),
        "predictor": classifier,
        "loss_fn": "mse",  # we can implement here a custom loss fn
        "dist_fn": dist_fn,
        "eps": args.attack_epsilon / 255,
        "nb_iter": args.attack_iterations,
        "dist_schedule": args.dist_schedule,
        "binary": False,
        "step": args.attack_step / 255,
        "confidence_threshold": args.confidence_threshold,
        "steps_dir": osp.join(args.output_path, "steps"),
    }

    attack = get_attack(
        args.attack_method,
        args.attack_joint and args.attack_joint_checkpoint,
        args.attack_joint and args.attack_joint_shortcut,
    )

    if args.attack_joint and (
        args.attack_joint_checkpoint or args.attack_joint_shortcut
    ):
        attack = attack(
            diffusion=respaced_diffusion,
            ddpm=model,
            steps=respaced_steps,
            stochastic=args.sampling_stochastic,
            backward_steps=args.attack_checkpoint_backward_steps,
            **attack_args,
        )
    else:
        attack = attack(**attack_args)  # Constructor

    dataset = get_data(args)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False
    )

    diffeocf_results: list[CFResult] = []
    masks = []

    # Other Ace Results
    pe_path = osp.join(result_dir, "pe")
    noise_path = osp.join(result_dir, "noise_x")
    mask_path = osp.join(result_dir, "mask")
    os.makedirs(pe_path, exist_ok=True)
    os.makedirs(noise_path, exist_ok=True)
    os.makedirs(mask_path, exist_ok=True)

    with tqdm(dataloader, desc="Running ACE") as pbar:
        for f, x, ys, mask in pbar:
            x = x.to(dist_util.dev())
            x_reconstructed, y_initial = joint_classifier.initial(x)

            pbar.set_postfix_str(f"Processing: {f}")

            # Hack for intermediate images
            image_names = [f.split("/")[-1].split(".")[0] for f in f]
            attack.current_image = image_names

            targets = get_experiment_targets(ys).to(dist_util.dev())
            ce, pe, noise, pe_mask, success, steps_done = filter_fn(
                diffusion=respaced_diffusion,
                attack=attack,
                model=model,
                steps=respaced_steps,
                x=x.to(dist_util.dev()),
                stochastic=args.sampling_stochastic,
                target=targets[: x.size(0)],
                inpaint=args.sampling_inpaint,
                dilation=args.sampling_dilation,
            )

            masks.append(mask.cpu())
            with torch.no_grad():
                y_final = joint_classifier.classifier(ce)
                x = x.detach().cpu()
                x_prime = ce.detach().cpu()

                for j in range(x.size(0)):
                    cf_result = CFResult(
                        image_path=f[j],
                        x=x[j].unsqueeze(0),
                        x_reconstructed=x_reconstructed[j].unsqueeze(0),
                        x_prime=x_prime[j].unsqueeze(0),
                        y_target=targets[j].item(),
                        y_initial_pred=y_initial[j].item(),
                        y_final_pred=y_final[j].item(),
                        success=success[j],
                        steps=steps_done[j],
                    )
                    cf_result.update_y_true_initial(ys[j].item())

                    diffeocf_results.append(cf_result)

            # save_img_threaded(pe[j], osp.join(pe_path, cf_result.image_name))
            # save_img_threaded(
            #     noise[j], osp.join(noise_path, cf_result.image_name)
            # )
            # save_img_threaded(
            #     pe_mask[j], osp.join(mask_path, cf_result.image_name)
            # )

    # Save the results for the diffeo_cf attacks
    del model
    del joint_classifier

    xs = [res.x[0] for res in diffeocf_results]
    x_cfs = [res.x_prime[0] for res in diffeocf_results]
    y_initials = torch.Tensor([res.y_initial_pred for res in diffeocf_results])
    y_ends = torch.Tensor([res.y_final_pred for res in diffeocf_results])

    # Update with true final predictions
    masks = torch.cat(masks)
    update_results_true_latents(
        [inner_square_color(x_cf, mask).item() for x_cf, mask in zip(x_cfs, masks)],
        diffeocf_results,
        args.confidence_threshold,
    )
    save_cf_results(diffeocf_results, result_dir)

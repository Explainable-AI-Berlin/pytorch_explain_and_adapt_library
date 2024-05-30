# I based the embedding learning on this repo
# https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_textual_inversion_training.ipynb
import math

import torchmetrics
import torchvision.utils
import yaml
import os
import tqdm
import random
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.linalg as linalg
import torch.utils.data as data
import torch.nn.functional as F

from diffusers import StableDiffusionPipeline, DDPMScheduler
from torchvision.transforms import ToTensor

from peal.dependencies.time.core.dataset import get_dataset, TextualDataset
from peal.dependencies.time.core.utils import (
    Print,
    add_new_tokens,
    load_tokens_and_embeddings,
    save_tokens_and_embeddings,
)
from peal.dependencies.time.core.phrases import get_phrase_generator


def freeze(m, names):
    for n in names:
        for p in getattr(m, n).parameters():
            p.requires_grad = False


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd_model", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--embedding-files", type=str, nargs="+", default=[])

    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--mini_batch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument(
        "--data_dir", type=str, default="/home/2017025/gjeann01/save/celeba"
    )
    parser.add_argument("--partition", type=str, default="train")
    parser.add_argument("--dataset", type=str, default="CelebAHQ")
    parser.add_argument("--seed", type=int, default=99999999)

    # classifier arguments
    parser.add_argument("--label_query", type=int, default=31)
    parser.add_argument(
        "--training_label",
        type=int,
        default=-1,
        help="Only used for binary classification",
    )

    # token related args
    parser.add_argument("--custom_tokens", type=str, nargs="+", required=True)
    parser.add_argument("--custom_tokens_init", type=str, nargs="+", required=True)
    parser.add_argument(
        "--phase", type=str, default="context", choices=["context", "class"]
    )
    parser.add_argument(
        "--base_prompt",
        type=str,
        default="A picture of",
        help='Used only in the "class" phase. It will be the base of the phrase to train the model',
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true"
    )

    return parser.parse_args()


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def run_epoch(
    data_loader,
    pipeline,
    noise_scheduler,
    optimizer,
    device,
    torch_dtype,
    args,
    iterations,
    num_chunks,
    tokenizer,
    text_encoder,
    index_no_updates,
):
    losses = []
    for image, text in tqdm.tqdm(
        data_loader(iterations), desc="Iterations", total=args.iterations
    ):
        image = image.to(device, dtype=torch_dtype)
        image = torchvision.transforms.Resize([512,512])(image)
        text_inputs = tokenizer(
            text,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device, dtype=torch.long)
        B = image.size(0)

        for img, text_ids in zip(
            image.chunk(num_chunks), text_input_ids.chunk(num_chunks)
        ):
            # Image encoding
            with torch.no_grad():
                latents = (
                    pipeline.vae.encode(img).latent_dist.sample().detach()
                )  # this encodes 256x256 images!
                latents = latents * pipeline.vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                if optimizer is None:
                    torch.manual_seed(0)

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device
                )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = pipeline.text_encoder(text_ids)[0].to(
                dtype=torch_dtype
            )

            # Predict the noise residual
            model_pred = pipeline.unet(
                noisy_latents, timesteps, encoder_hidden_states
            ).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise

            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)

            else:
                raise ValueError(
                    f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                )

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="sum") / B
            if not optimizer is None:
                loss.backward()

            losses.append(loss.item())

        if not optimizer is None:
            # modify gradients of tokens that we don't want to change
            with torch.no_grad():
                grad = text_encoder.get_input_embeddings().weight.grad
                # weight decay
                try:
                    grad = grad.add(
                        text_encoder.get_input_embeddings().weight.data,
                        alpha=args.weight_decay,
                    )

                except Exception:
                    import pdb

                    pdb.set_trace()
                # zero-out gradients of those embeddings we don't want to modify
                grad = grad * (1 - index_no_updates.unsqueeze(1))
                text_encoder.get_input_embeddings().weight.grad = grad

            optimizer.step()
            optimizer.zero_grad()

        if iterations > args.iterations:
            return torch.mean(torch.tensor(losses))

        iterations += 1


def training(args=None):
    # =================================================================
    # Custom variables

    if args is None:
        args = arguments()
        with open("training.yaml", "w") as f:
            yaml.dump(vars(args), f, default_flow_style=False)

    phase = args.phase
    phase += str(args.training_label) if args.training_label >= 0 else ""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch_dtype = torch.float16 if args.use_fp16 else torch.float32
    device = torch.device("cuda")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    # =================================================================
    # Instantiate Pipeline
    Print("Initializing Stable Diffusion")
    if not hasattr(args, "pipeline"):
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.sd_model,
            torch_dtype=torch_dtype,
        )
        pipeline.to(device)

    else:
        pipeline = args.pipeline

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.sd_model, subfolder="scheduler"
    )
    # =================================================================
    # Initialize token(s)

    # load previous tokens
    load_tokens_and_embeddings(sd_model=pipeline, files=args.embedding_files)

    # generate new tokens
    index_no_updates, placeholder_token_ids, initializer_token_ids = add_new_tokens(
        new_tokens=args.custom_tokens,
        init_tokens=args.custom_tokens_init,
        sd_model=pipeline,
        dtype=torch_dtype,
    )

    # =================================================================
    # Get tokenizer and text encoder

    text_encoder = pipeline.text_encoder
    text_encoder.requires_grad_(True)
    tokenizer = pipeline.tokenizer

    # freeze all network parameters except the embeddings
    freeze(pipeline, ["vae", "unet"])
    freeze(pipeline.text_encoder.text_model, ["encoder", "final_layer_norm"])
    freeze(text_encoder.text_model.embeddings, ["position_embedding"])
    text_encoder.get_input_embeddings().weight.requires_grad = True

    if args.enable_xformers_memory_efficient_attention:
        """
        Must have 0.0.17>xformer.__version__
        """
        pipeline.unet.enable_xformers_memory_efficient_attention()

    # =================================================================
    # Optimizer
    Print("Initializing optimizer and dataset")
    # Here, there is a problem, we are optimizing the other embeddings as well with the weight decay as well.
    # We will compute the wd manually
    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=0,  # manually implemented
        eps=args.adam_epsilon,
    )

    # =================================================================
    # Dataset
    if isinstance(args.dataset, torch.utils.data.Dataset):
        dataset_raw = args.dataset

    else:
        dataset_raw = get_dataset(args)

    if args.training_label >= 0:
        dataset_raw.enable_class_restriction(args.training_label)
        print("len restricted dataset")
        print("len restricted dataset")
        print("len restricted dataset")
        print(len(dataset_raw))

    dataset = TextualDataset(
        custom_tokens=args.custom_tokens,
        base_prompt_generator=get_phrase_generator(args),
        dataset=dataset_raw,
    )
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=5,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    if args.training_label >= 0:
        args.generator_dataset_val.enable_class_restriction(args.training_label)

    dataset_val = TextualDataset(
        custom_tokens=args.custom_tokens,
        base_prompt_generator=get_phrase_generator(args),
        dataset=args.generator_dataset_val,
    )
    loader_val = data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=5,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    # =================================================================
    # Training Loop

    num_chunks = args.batch_size // args.mini_batch_size

    Print("Training!")
    differences = []
    iterations = 0

    fid = torchmetrics.image.fid.FrechetInceptionDistance(
        feature=192, reset_real_features=False
    )
    fid.to(device)
    real_images = []
    for i in range(min(len(args.generator_dataset_val), 100)):
        real_images.append(args.generator_dataset_val[i][0])

    real_images = torch.stack(real_images, dim=0).to(device) * 0.5 + 0.5
    fid.update(torch.tensor(255 * real_images, dtype=torch.uint8), real=True)

    images = pipeline(5 * [args.prompt]).images
    images_torch = torch.stack([ToTensor()(image) for image in images])
    images_torch_resized = torchvision.transforms.Resize(real_images.shape[-2:])(images_torch)
    concatenated_imgs = torch.cat([real_images[:5].cpu() , images_torch_resized], dim=0)
    torchvision.utils.save_image(concatenated_imgs, args.output_path[:-4] + "_start.png",nrow=5)

    fid.update(
        torch.tensor(
            255 * concatenated_imgs,
            dtype=torch.uint8,
        ).to(device),
        real=False,
    )
    fid_score = float(fid.compute())
    args.writer.add_scalar(phase + "_fid", fid_score, -1)

    for epoch in range(args.max_epoch):

        def data_loader(iterations):
            while True:
                yield from loader

        def data_loader_val(iterations):
            while True:
                yield from loader_val

        train_loss = run_epoch(
            data_loader,
            pipeline,
            noise_scheduler,
            optimizer,
            device,
            torch_dtype,
            args,
            iterations,
            num_chunks,
            tokenizer,
            text_encoder,
            index_no_updates,
        )
        args.writer.add_scalar(phase + "_train_loss", train_loss, epoch)
        val_loss = run_epoch(
            data_loader_val,
            pipeline,
            noise_scheduler,
            None,
            device,
            torch_dtype,
            args,
            iterations,
            num_chunks,
            tokenizer,
            text_encoder,
            index_no_updates,
        )
        args.writer.add_scalar(phase + "_val_loss", val_loss, epoch)
        with torch.no_grad():
            embeddings = text_encoder.get_input_embeddings().weight.data
            d = (
                (embeddings[placeholder_token_ids] - embeddings[initializer_token_ids])
                .abs()
                .mean(dim=1, keepdim=True)
            )
            differences.append(d.detach().cpu())
            Print(f"Mean difference at epoch {epoch}:", differences[-1])

        matplotlib.use("Agg")
        differences_th = torch.cat(differences, dim=1).numpy()
        for idx, token in enumerate(args.custom_tokens):
            plt.plot(differences_th[idx, :], label=token)

        plt.title("L_1 difference")
        plt.legend()
        plt.savefig(args.output_path[:-4] + f"_differences-{epoch}.png")
        plt.close()

        save_tokens_and_embeddings(
            sd_model=pipeline,
            tokens=args.custom_tokens,
            output=args.output_path[:-4] + f"-ckpt-{epoch}.pth",
        )
        save_tokens_and_embeddings(
            sd_model=pipeline,
            tokens=args.custom_tokens,
            output=args.output_path,
        )
        images = pipeline(5 * [args.prompt]).images
        images_torch = torch.stack([ToTensor()(image) for image in images])
        images_torch_resized = torchvision.transforms.Resize(real_images.shape[-2:])(images_torch)
        concatenated_imgs = torch.cat([real_images[:5].cpu() , images_torch_resized], dim=0)
        torchvision.utils.save_image(concatenated_imgs, args.output_path[:-4] + f"_image_{epoch}.png",nrow=5)

        fid.update(
            torch.tensor(
                255 * concatenated_imgs,
                dtype=torch.uint8,
            ).to(device),
            real=False,
        )
        fid_score = float(fid.compute())
        args.writer.add_scalar(phase + "_fid", fid_score, epoch)

    pipeline.to("cpu")
    save_tokens_and_embeddings(
        sd_model=pipeline,
        tokens=args.custom_tokens,
        output=args.output_path,
    )
    dataset_raw.disable_class_restriction()
    print("len unrestricted dataset")
    print("len unrestricted dataset")
    print("len unrestricted dataset")
    print(len(dataset_raw))
    args.generator_dataset_val.disable_class_restriction()


if __name__ == "__main__":
    training()

import os
from pathlib import Path

import torchmetrics
from tqdm import tqdm
import numpy as np
from PIL import Image
from math import log, sqrt, pi

import argparse

import torch
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from peal.dependencies.glow.model import Glow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Glow trainer")
parser.add_argument("--batch", default=16, type=int, help="batch size")
parser.add_argument("--iter", default=200000, type=int, help="maximum iterations")
parser.add_argument(
    "--n_flow", default=32, type=int, help="number of flows in each block"
)
parser.add_argument("--n_block", default=4, type=int, help="number of blocks")
parser.add_argument(
    "--no_lu",
    action="store_true",
    help="use plain convolution instead of LU decomposed version",
)
parser.add_argument(
    "--affine", action="store_true", help="use affine coupling instead of additive"
)
parser.add_argument("--n_bits", default=5, type=int, help="number of bits")
parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
parser.add_argument("--img_size", default=64, type=int, help="image size")
parser.add_argument("--temp", default=0.7, type=float, help="temperature of sampling")
parser.add_argument("--n_sample", default=20, type=int, help="number of samples")
parser.add_argument(
    "--base_path",
    default="glow_run",
    metavar="PATH",
    type=str,
    help="Path to image directory",
)


def sample_data(path, batch_size, image_size):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.ImageFolder(path, transform=transform)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = DataLoader(
                dataset, shuffle=True, batch_size=batch_size, num_workers=4
            )
            loader = iter(loader)
            yield next(loader)


def calc_z_shapes(n_channel, input_size, n_flow, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes


def calc_loss(log_p, logdet, image_size, n_bins):
    # log_p = calc_log_p([z_list])
    n_pixel = image_size * image_size * 3

    loss = -log(n_bins) * n_pixel
    loss = loss + logdet + log_p

    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )


def train(args, model, optimizer, model_single):
    if not hasattr(args, "train_dataloader"):
        dataset = iter(sample_data(args.path, args.batch, args.img_size))

    else:
        dataset = args.train_dataloader

    n_bins = 2.0**args.n_bits

    z_sample = []
    z_shapes = calc_z_shapes(3, args.img_size, args.n_flow, args.n_block)
    for z in z_shapes:
        z_new = torch.randn(args.n_sample, *z) * args.temp
        z_sample.append(z_new.to(device))

    fid = torchmetrics.image.fid.FrechetInceptionDistance(
        feature=192, reset_real_features=False
    )
    fid.to(device)
    real_images = []
    for i in range(min(len(args.val_dataloader.dataset), 100)):
        real_images.append(args.val_dataloader.dataset[i][0])

    real_images = torch.stack(real_images, dim=0).to(device) * 0.5 + 0.5
    fid.update(torch.tensor(255 * real_images, dtype=torch.uint8), real=True)

    with tqdm(range(args.iter)) as pbar:
        for i in pbar:
            image, _ = next(iter(dataset))
            image = image.to(device)

            image = image * 255

            if args.n_bits < 8:
                image = torch.floor(image / 2 ** (8 - args.n_bits))

            image = image / n_bins - 0.5

            if i == 0:
                with torch.no_grad():
                    log_p, logdet, _ = model.module(
                        image + torch.rand_like(image) / n_bins
                    )

                    continue

            else:
                log_p, logdet, _ = model(image + torch.rand_like(image) / n_bins)

            logdet = logdet.mean()

            loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins)
            model.zero_grad()
            loss.backward()
            # warmup_lr = args.lr * min(1, i * batch_size / (50000 * 10))
            warmup_lr = args.lr
            optimizer.param_groups[0]["lr"] = warmup_lr
            optimizer.step()

            pbar.set_description(
                f"Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; lr: {warmup_lr:.7f}"
            )
            args.writer.add_scalar("Loss", loss.detach().item(), i)
            args.writer.add_scalar("logP", log_p.detach().item(), i)
            args.writer.add_scalar("logdet", log_det.detach().item(), i)

            if i % 100 == 0:
                with torch.no_grad():
                    img_samples = model_single.reverse(z_sample).data.detach() / 2 + 0.5
                    utils.save_image(
                        img_samples.cpu(),
                        os.path.join(
                            args.base_path, "outputs", f"{str(i + 1).zfill(6)}.png"
                        ),
                        normalize=True,
                        nrow=10,
                        # range=(-0.5, 0.5),
                    )
                    fid.update(
                        torch.tensor(
                            255 * torch.clamp(img_samples, 0, 1), dtype=torch.uint8
                        ),
                        real=False,
                    )
                    fid_score = float(fid.compute())
                    args.writer.add_scalar("fid", fid_score, i)

            if i % 10000 == 0:
                torch.save(
                    model_single.state_dict(),
                    os.path.join(
                        args.base_path, "checkpoint", f"model_{str(i + 1).zfill(6)}.pt"
                    ),
                )
                torch.save(
                    model_single.state_dict(), os.path.join(args.base_path, f"final.pt")
                )
                model_single.load_state_dict(
                    torch.load(os.path.join(args.base_path, "final.pt")),
                )
                torch.save(
                    optimizer.state_dict(),
                    os.path.join(
                        args.base_path, "checkpoint", f"optim_{str(i + 1).zfill(6)}.pt"
                    ),
                )


def training(args=None):
    if args is None:
        args = parser.parse_args()

    if not hasattr(args, "glow"):
        model_single = Glow(
            3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu
        )

    else:
        model_single = args.glow

    model = nn.DataParallel(model_single)
    # model = model_single
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    Path(os.path.join(args.base_path, "outputs")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.base_path, "checkpoint")).mkdir(parents=True, exist_ok=True)
    train(args, model, optimizer, model_single)


if __name__ == "__main__":
    training()

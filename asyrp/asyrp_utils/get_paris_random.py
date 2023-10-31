import torch
import os
import time
import numpy as np
import torchvision.utils as tvu
import torchvision.transforms as transforms

from tqdm import tqdm
from PIL import Image

from asyrp.utils.diffusion_utils import denoising_step
from asyrp.datasets.imagenet_dic import IMAGENET_DIC
from asyrp.datasets.data_utils import get_dataset, get_dataloader
from asyrp.configs.paths_config import DATASET_PATHS

@torch.no_grad()
def random_noise_pairs(runner, model, saved_noise=False, save_imgs=False):
    print("Prepare random latent")
    seq_inv = np.linspace(0, 1, runner.args.n_inv_step) * runner.args.t_0
    seq_inv = [int(s + 1e-6) for s in list(seq_inv)]
    seq_inv_next = [-1] + list(seq_inv[:-1])

    n = 1
    img_lat_pairs_dic = {}

    if saved_noise:
        for mode in ["train", "test"]:
            img_lat_pairs = []
            if runner.config.data.dataset == "IMAGENET":
                if runner.args.target_class_num is not None:
                    pairs_path = os.path.join(
                        runner.args.exp,
                        "precomputed/",
                        f"{runner.config.data.category}_{IMAGENET_DIC[str(runner.args.target_class_num)][1]}_" +
                        f"{mode}_random_noise_nim{runner.args.n_precomp_img}_ninv{runner.args.n_inv_step}_pairs.pth",
                    )

                else:
                    pairs_path = os.path.join(
                        runner.args.exp,
                        "precomputed/",
                        f"{runner.config.data.category}_{mode}_random_noise_nim{runner.args.n_precomp_img}" +
                        f"_ninv{runner.args.n_inv_step}_pairs.pth",
                    )

            else:
                if mode == "train":
                    pairs_path = os.path.join(
                        runner.args.exp,
                        "precomputed/",
                        f"{runner.config.data.category}_{mode}_random_noise_nim{runner.args.n_train_img}" +
                        f"_ninv{runner.args.n_inv_step}_pairs.pth",
                    )

                else:
                    pairs_path = os.path.join(
                        runner.args.exp,
                        "precomputed/",
                        f"{runner.config.data.category}_{mode}_random_noise_nim{runner.args.n_test_img}" +
                        f"_ninv{runner.args.n_inv_step}_pairs.pth",
                    )

            print(pairs_path)
            if os.path.exists(pairs_path):
                print(f"{mode} pairs exists")
                img_lat_pairs_dic[mode] = torch.load(
                    pairs_path, map_location=torch.device("cpu")
                )
                if save_imgs:
                    for step, (_, x_id, x_lat) in enumerate(
                        img_lat_pairs_dic[mode]
                    ):
                        tvu.save_image(
                            (x_id + 1) * 0.5,
                            os.path.join(
                                runner.args.image_folder,
                                f"{mode}_{step}_1_rec_ninv{runner.args.n_inv_step}.png",
                            ),
                        )
                        if step == runner.args.n_precomp_img - 1:
                            break
                continue

            step = 0
            while True:
                with torch.no_grad():
                    x_lat = torch.randn(
                        (
                            1,
                            runner.config.data.channels,
                            runner.config.data.image_size,
                            runner.config.data.image_size,
                        )
                    ).to(runner.device)

                    if save_imgs:
                        tvu.save_image(
                            (x_lat + 1) * 0.5,
                            os.path.join(
                                runner.args.image_folder,
                                f"{mode}_{step}_1_lat_ninv{runner.args.n_inv_step}.png",
                            ),
                        )

                    with tqdm(
                        total=len(seq_inv), desc=f"Generative process {mode} {step}"
                    ) as progress_bar:
                        time_s = time.time()
                        x = x_lat
                        for it, (i, j) in enumerate(
                            zip(reversed((seq_inv)), reversed((seq_inv_next)))
                        ):
                            t = (torch.ones(n) * i).to(runner.device)
                            t_next = (torch.ones(n) * j).to(runner.device)

                            x, _, _, _ = denoising_step(
                                x,
                                t=t,
                                t_next=t_next,
                                models=model,
                                logvars=runner.logvar,
                                sampling_type=runner.args.sample_type,
                                b=runner.betas,
                                learn_sigma=runner.learn_sigma,
                            )
                            progress_bar.update(1)
                        time_e = time.time()
                        print(f"{time_e - time_s} seconds")

                    # img_lat_pairs.append([None, x.detach().clone(), x_lat.detach().clone()])
                    img_lat_pairs.append(
                        [
                            x.detach().clone(),
                            x.detach().clone(),
                            x_lat.detach().clone(),
                        ]
                    )

                if save_imgs:
                    tvu.save_image(
                        (x + 1) * 0.5,
                        os.path.join(
                            runner.args.image_folder,
                            f"{mode}_{step}_1_rec_ninv{runner.args.n_inv_step}.png",
                        ),
                    )
                if (mode == "train" and step == runner.args.n_train_img - 1) or (
                    mode == "test" and step == runner.args.n_test_img - 1
                ):
                    break
                step += 1

            img_lat_pairs_dic[mode] = img_lat_pairs
            torch.save(img_lat_pairs, pairs_path)

    else:
        train_lat = []
        for i in range(runner.args.n_train_img):
            lat = torch.randn(
                (
                    1,
                    runner.config.data.channels,
                    runner.config.data.image_size,
                    runner.config.data.image_size,
                )
            ).to(runner.device)
            # train_lat.append([None, None, lat])
            train_lat.append([torch.zeros_like(lat), torch.zeros_like(lat), lat])

        img_lat_pairs_dic["train"] = train_lat

        test_lat = []
        for i in range(runner.args.n_test_img):
            lat = torch.randn(
                (
                    1,
                    runner.config.data.channels,
                    runner.config.data.image_size,
                    runner.config.data.image_size,
                )
            ).to(runner.device)
            # test_lat.append([None, None, lat])
            test_lat.append([torch.zeros_like(lat), torch.zeros_like(lat), lat])

        img_lat_pairs_dic["test"] = test_lat

    return img_lat_pairs_dic
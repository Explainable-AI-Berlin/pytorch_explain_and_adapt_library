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

# ----------- Get random latent -----------#
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
                        "precomputed/",
                        f"{runner.config.data.category}_{IMAGENET_DIC[str(runner.args.target_class_num)][1]}_" +
                        f"{mode}_random_noise_nim{runner.args.n_precomp_img}_ninv{runner.args.n_inv_step}_pairs.pth",
                    )

                else:
                    pairs_path = os.path.join(
                        "precomputed/",
                        f"{runner.config.data.category}_{mode}_random_noise_nim{runner.args.n_precomp_img}" +
                        f"_ninv{runner.args.n_inv_step}_pairs.pth",
                    )

            else:
                if mode == "train":
                    pairs_path = os.path.join(
                        "precomputed/",
                        f"{runner.config.data.category}_{mode}_random_noise_nim{runner.args.n_train_img}" +
                        f"_ninv{runner.args.n_inv_step}_pairs.pth",
                    )

                else:
                    pairs_path = os.path.join(
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


# ----------- Pre-compute -----------#
@torch.no_grad()
def precompute_pairs(runner, model, save_imgs=False):
    print("Prepare identity latent")
    seq_inv = np.linspace(0, 1, runner.args.n_inv_step) * runner.args.t_0
    seq_inv = [int(s + 1e-6) for s in list(seq_inv)]
    seq_inv_next = [-1] + list(seq_inv[:-1])

    n = 1
    img_lat_pairs_dic = {}

    for mode in ["train", "test"]:
        img_lat_pairs = []
        if runner.config.data.dataset == "IMAGENET":
            if runner.args.target_class_num is not None:
                pairs_path = os.path.join(
                    "precomputed/",
                    f"{runner.config.data.category}_{IMAGENET_DIC[str(runner.args.target_class_num)][1]}_{mode}_t{runner.args.t_0}_nim{runner.args.n_precomp_img}_ninv{runner.args.n_inv_step}_pairs.pth",
                )
            else:
                pairs_path = os.path.join(
                    "precomputed/",
                    f"{runner.config.data.category}_{mode}_t{runner.args.t_0}_nim{runner.args.n_precomp_img}_ninv{runner.args.n_inv_step}_pairs.pth",
                )

        else:
            if mode == "train":
                pairs_path = os.path.join(
                    "precomputed/",
                    f"{runner.config.data.category}_{mode}_t{runner.args.t_0}_nim{runner.args.n_train_img}_ninv{runner.args.n_inv_step}_pairs.pth",
                )
            else:
                pairs_path = os.path.join(
                    "precomputed/",
                    f"{runner.config.data.category}_{mode}_t{runner.args.t_0}_nim{runner.args.n_test_img}_ninv{runner.args.n_inv_step}_pairs.pth",
                )
        print(pairs_path)
        if os.path.exists(pairs_path) and not runner.args.re_precompute:
            print(f"{mode} pairs exists")
            img_lat_pairs_dic[mode] = torch.load(
                pairs_path, map_location=torch.device("cpu")
            )
            if save_imgs:
                for step, (x0, x_id, x_lat) in enumerate(img_lat_pairs_dic[mode]):
                    tvu.save_image(
                        (x0 + 1) * 0.5,
                        os.path.join(
                            runner.args.image_folder, f"{mode}_{step}_0_orig.png"
                        ),
                    )
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
        else:
            exist_num = 0
            for exist_precompute_num in reversed(
                range(
                    runner.args.n_train_img
                    if mode == "train"
                    else runner.args.n_test_img
                )
            ):
                tmp_path = os.path.join(
                    "precomputed/",
                    f"{runner.config.data.category}_{mode}_t{runner.args.t_0}_nim{exist_precompute_num}_ninv{runner.args.n_inv_step}_pairs.pth",
                )
                if os.path.exists(tmp_path):
                    print(
                        f"latest {mode} pairs are exist. Continue precomputing..."
                    )
                    img_lat_pairs = img_lat_pairs + torch.load(
                        tmp_path, map_location=torch.device("cpu")
                    )
                    exist_num = exist_precompute_num
                    break

            train_dataset, test_dataset = get_dataset(
                runner.config.data.dataset,
                DATASET_PATHS,
                runner.config,
                target_class_num=runner.args.target_class_num,
            )

            loader_dic = get_dataloader(
                train_dataset,
                test_dataset,
                bs_train=1,  # runner.args.bs_train,
                num_workers=runner.config.data.num_workers,
                shuffle=runner.args.shuffle_train_dataloader,
            )
            loader = loader_dic[mode]

            if runner.args.save_process_origin:
                save_process_folder = os.path.join(
                    runner.args.image_folder, f"inversion_process"
                )
                if not os.path.exists(save_process_folder):
                    os.makedirs(save_process_folder)

        for step, img in enumerate(loader):
            if (mode == "train" and step == runner.args.n_train_img) or (
                mode == "test" and step == runner.args.n_test_img
            ):
                break
            if exist_num != 0:
                exist_num = exist_num - 1
                continue
            x0 = img.to(runner.config.device)
            if save_imgs:
                tvu.save_image(
                    (x0 + 1) * 0.5,
                    os.path.join(
                        runner.args.image_folder, f"{mode}_{step}_0_orig.png"
                    ),
                )

            x = x0.clone()
            model.eval()
            time_s = time.time()
            with torch.no_grad():
                with tqdm(
                    total=len(seq_inv), desc=f"Inversion process {mode} {step}"
                ) as progress_bar:
                    for it, (i, j) in enumerate(
                        zip((seq_inv_next[1:]), (seq_inv[1:]))
                    ):
                        t = (torch.ones(n) * i).to(runner.device)
                        t_prev = (torch.ones(n) * j).to(runner.device)

                        x, _, _, _ = denoising_step(
                            x,
                            t=t,
                            t_next=t_prev,
                            models=model,
                            logvars=runner.logvar,
                            sampling_type="ddim",
                            b=runner.betas,
                            eta=0,
                            learn_sigma=runner.learn_sigma,
                        )
                        progress_bar.update(1)

                time_e = time.time()
                print(f"{time_e - time_s} seconds")
                x_lat = x.clone()
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
                    for it, (i, j) in enumerate(
                        zip(reversed((seq_inv)), reversed((seq_inv_next)))
                    ):
                        t = (torch.ones(n) * i).to(runner.device)
                        t_next = (torch.ones(n) * j).to(runner.device)

                        x, x0t, _, _ = denoising_step(
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
                        if runner.args.save_process_origin:
                            tvu.save_image(
                                (x + 1) * 0.5,
                                os.path.join(
                                    save_process_folder,
                                    f"xt_{step}_{it}_{t[0]}.png",
                                ),
                            )
                            tvu.save_image(
                                (x0t + 1) * 0.5,
                                os.path.join(
                                    save_process_folder,
                                    f"x0t_{step}_{it}_{t[0]}.png",
                                ),
                            )
                    time_e = time.time()
                    print(f"{time_e - time_s} seconds")

                img_lat_pairs.append(
                    [x0, x.detach().clone(), x_lat.detach().clone()]
                )

            if save_imgs:
                tvu.save_image(
                    (x + 1) * 0.5,
                    os.path.join(
                        runner.args.image_folder,
                        f"{mode}_{step}_1_rec_ninv{runner.args.n_inv_step}.png",
                    ),
                )

        img_lat_pairs_dic[mode] = img_lat_pairs
        # pairs_path = os.path.join('precomputed/',
        #                           f'{runner.config.data.category}_{mode}_t{runner.args.t_0}_nim{runner.args.n_precomp_img}_ninv{runner.args.n_inv_step}_pairs.pth')
        torch.save(img_lat_pairs, pairs_path)

    return img_lat_pairs_dic


@torch.no_grad()
def precompute_pairs_with_h(runner, model, img_path):
    if not os.path.exists("./precomputed"):
        os.mkdir("./precomputed")

    save_path = "_".join(img_path.split(".")[-2].split("/")[-2:])
    save_path = (
        runner.config.data.category
        + "_inv"
        + str(runner.args.n_inv_step)
        + "_"
        + save_path
        + ".pt"
    )
    save_path = os.path.join("precomputed", save_path)

    n = 1

    print("Precompute multiple h and x_T")
    seq_inv = np.linspace(0, 1, runner.args.n_inv_step) * runner.args.t_0
    seq_inv = [int(s + 1e-6) for s in list(seq_inv)]
    seq_inv_next = [-1] + list(seq_inv[:-1])

    if os.path.exists(save_path):
        print("Precomputed pairs already exist")
        img_lat_pair = torch.load(save_path)
        return img_lat_pair
    else:
        tmp_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        image = Image.open(img_path).convert("RGB")

        width, height = image.size
        if width > height:
            image = transforms.CenterCrop(height)(image)
        else:
            image = transforms.CenterCrop(width)(image)

        image = tmp_transform(image)

        h_dic = {}

        x0 = image.unsqueeze(0).to(runner.device)

        x = x0.clone()
        model.eval()
        time_s = time.time()

        with torch.no_grad():
            with tqdm(
                total=len(seq_inv), desc=f"Inversion processing"
            ) as progress_bar:
                for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                    t = (torch.ones(n) * i).to(runner.device)
                    t_prev = (torch.ones(n) * j).to(runner.device)

                    x, _, _, h = denoising_step(
                        x,
                        t=t,
                        t_next=t_prev,
                        models=model,
                        logvars=runner.logvar,
                        sampling_type="ddim",
                        b=runner.betas,
                        eta=0,
                        learn_sigma=runner.learn_sigma,
                    )
                    progress_bar.update(1)
                    h_dic[i] = h.detach().clone().cpu()

            time_e = time.time()
            progress_bar.set_description(
                f"Inversion processing time: {time_e - time_s:.2f}s"
            )
            x_lat = x.clone()
        print("Generative process is skipped")

        img_lat_pairs = [x0, 0, x_lat.detach().clone().cpu(), h_dic]

        torch.save(img_lat_pairs, save_path)
        print("Precomputed pairs are saved to ", save_path)

        return img_lat_pairs


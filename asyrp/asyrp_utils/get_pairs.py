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



def precompute(loader, mode, runner, save_imgs, model, seq_inv, seq_inv_next, img_lat_pairs, exist_num, save_process_folder, n):
    for step, img in enumerate(loader):
        if isinstance(img, tuple) or isinstance(img, list):
            target = img[1]
            img = img[0]

        else:
            target = None

        if (mode == "train" and step == runner.args.n_train_img) or (
            mode == "test" and step == runner.args.n_test_img
        ):
            break
        if exist_num != 0:
            exist_num = exist_num - 1
            continue

        try:
            x0 = img.to(runner.config.device)

        except Exception:
            import pdb; pdb.set_trace()

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
                    """if runner.args.save_process_origin:
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
                        )"""
                time_e = time.time()
                print(f"{time_e - time_s} seconds")

            img_latent_pair = [x0, x.detach().clone(), x_lat.detach().clone(), target]

            img_lat_pairs.append(
                img_latent_pair
            )

        if save_imgs:
            tvu.save_image(
                (x + 1) * 0.5,
                os.path.join(
                    runner.args.image_folder,
                    f"{mode}_{step}_1_rec_ninv{runner.args.n_inv_step}.png",
                ),
            )


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
        '''if runner.config.data.dataset == "IMAGENET":
            if runner.args.target_class_num is not None:
                pairs_path = os.path.join(
                    runner.args.exp,
                    "precomputed/",
                    f"{runner.config.data.category}_{IMAGENET_DIC[str(runner.args.target_class_num)][1]}_{mode}_t{runner.args.t_0}_nim{runner.args.n_precomp_img}_ninv{runner.args.n_inv_step}_pairs.pth",
                )

            else:
                pairs_path = os.path.join(
                    runner.args.exp,
                    "precomputed/",
                    f"{runner.config.data.category}_{mode}_t{runner.args.t_0}_nim{runner.args.n_precomp_img}_ninv{runner.args.n_inv_step}_pairs.pth",
                )

        else:
            if mode == "train":
                pairs_path = os.path.join(
                    runner.args.exp,
                    "precomputed/",
                    f"{runner.config.data.category}_{mode}_t{runner.args.t_0}_nim{runner.args.n_train_img}_ninv{runner.args.n_inv_step}_pairs.pth",
                )

            else:
                pairs_path = os.path.join(
                    runner.args.exp,
                    "precomputed/",
                    f"{runner.config.data.category}_{mode}_t{runner.args.t_0}_nim{runner.args.n_test_img}_ninv{runner.args.n_inv_step}_pairs.pth",
                )'''

        pairs_path = os.path.join(
            runner.args.exp,
            "precomputed/",
            f"{runner.config.data.category}_{mode}_t{runner.args.t_0}_nim{runner.args.n_train_img}_ninv{runner.args.n_inv_step}_pairs.pth",
        )
        if not os.path.exists(os.path.dirname(pairs_path)):
            os.makedirs(os.path.dirname(pairs_path))

        print(pairs_path)
        '''if os.path.exists(pairs_path) and not runner.args.re_precompute:
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

        else:'''

        exist_num = 0
        for exist_precompute_num in reversed(
            range(
                runner.args.n_train_img
                if mode == "train"
                else runner.args.n_test_img
            )
        ):
            tmp_path = os.path.join(
                runner.args.exp,
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

        if not hasattr(runner.args, "datasets"):
            train_dataset, test_dataset = get_dataset(
                runner.config.data.dataset,
                DATASET_PATHS,
                runner.config,
                target_class_num=runner.args.target_class_num,
            )

        else:
            train_dataset, test_dataset = runner.args.datasets

        loader_dic = get_dataloader(
            train_dataset,
            test_dataset,
            bs_train=1,  # runner.args.bs_train,
            num_workers=runner.config.data.num_workers,
            shuffle=runner.args.shuffle_train_dataloader,
        )
        loader = loader_dic[mode]

        save_process_folder = "process_origin"
        """if runner.args.save_process_origin:
            save_process_folder = os.path.join(
                runner.args.image_folder, f"inversion_process"
            )
            if not os.path.exists(save_process_folder):
                os.makedirs(save_process_folder)"""

        precompute(loader, mode, runner, save_imgs, model, seq_inv, seq_inv_next, img_lat_pairs, exist_num, save_process_folder, n)

        img_lat_pairs_dic[mode] = img_lat_pairs
        # pairs_path = os.path.join('precomputed/',
        #                           f'{runner.config.data.category}_{mode}_t{runner.args.t_0}_nim{runner.args.n_precomp_img}_ninv{runner.args.n_inv_step}_pairs.pth')
        torch.save(img_lat_pairs, pairs_path)

    return img_lat_pairs_dic


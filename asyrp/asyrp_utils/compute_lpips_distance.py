import torch
import os
import time
import numpy as np
import torchvision.utils as tvu

from tqdm import tqdm

from asyrp.configs.paths_config import DATASET_PATHS
from asyrp.datasets.data_utils import get_dataset, get_dataloader
from asyrp.utils.diffusion_utils import denoising_step

@torch.no_grad()
def compute_lpips_distance(runner):
    import pickle

    print("Get lpips distance...")
    runner.args.bs_train = 1

    # ----------- Model -----------#

    model = runner.load_pretrained_model()

    model = model.to(runner.device)
    model = torch.nn.DataParallel(model)

    import lpips

    loss_fn_alex = lpips.LPIPS(net="alex")
    loss_fn_alex = loss_fn_alex.to(runner.device)

    # ----------- Pre-compute -----------#
    print("Prepare identity latent")
    seq_inv = np.linspace(0, 1, runner.args.n_inv_step) * runner.args.t_0
    seq_inv = [int(s + 1e-6) for s in list(seq_inv)]
    seq_inv_next = [-1] + list(seq_inv[:-1])

    print("the list is Unique? :", len(seq_inv) == len(set(seq_inv)))

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
    )
    loader = loader_dic["train"]
    print("Load dataset done")

    lpips_distance_list = {}
    lpips_distance_list_x0_t = {}
    for seq in seq_inv[1:]:
        lpips_distance_list[seq] = []
        lpips_distance_list_x0_t[seq] = []

    lpips_distance_std_list = {}
    lpips_distance_std_list_x0_t = {}
    for seq in seq_inv[1:]:
        lpips_distance_std_list[seq] = []
        lpips_distance_std_list_x0_t[seq] = []

    save_imgs = True

    for step, img in enumerate(loader):
        x0 = img.to(runner.device)
        if save_imgs:
            tvu.save_image(
                (x0 + 1) * 0.5,
                os.path.join(runner.args.image_folder, f"LPIPS_{step}_0_orig.png"),
            )

        x = x0.clone()
        model.eval()
        time_s = time.time()
        with torch.no_grad():
            with tqdm(
                total=len(seq_inv), desc=f"Inversion process {step}"
            ) as progress_bar:
                for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                    t = (torch.ones(runner.args.bs_train) * i).to(runner.device)
                    t_prev = (torch.ones(runner.args.bs_train) * j).to(runner.device)

                    x, x0_t, _, _ = denoising_step(
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
                    lpips_x = loss_fn_alex(x, x0)
                    lpips_x0 = loss_fn_alex(x0_t, x0)
                    lpips_distance_list[j].append(lpips_x.item())
                    lpips_distance_list_x0_t[j].append(lpips_x0.item())
                    if save_imgs:
                        tvu.save_image(
                            (x + 1) * 0.5,
                            os.path.join(
                                runner.args.image_folder, f"LPIPS_{step}_{j}.png"
                            ),
                        )
                        tvu.save_image(
                            (x0_t + 1) * 0.5,
                            os.path.join(
                                runner.args.image_folder, f"X0_t_LPIPS_{step}_{j}.png"
                            ),
                        )
                    progress_bar.update(1)

            time_e = time.time()
            print(f"{time_e - time_s} seconds")

        save_imgs = False
        if runner.args.n_train_img == step:
            break

    result_x_tsv = ""
    result_x_std_tsv = ""
    result_x0_tsv = ""
    result_x0_std_tsv = ""
    for seq in seq_inv[1:]:
        lpips_distance_std_list[seq] = np.std(lpips_distance_list[seq])
        lpips_distance_list[seq] = np.mean(lpips_distance_list[seq])

        # print(f"{seq} : {lpips_distance_list[seq]}")
        lpips_distance_std_list_x0_t[seq] = np.std(lpips_distance_list_x0_t[seq])
        lpips_distance_list_x0_t[seq] = np.mean(lpips_distance_list_x0_t[seq])

        # print(f"{seq} : {lpips_distance_list_x0_t[seq]}")
        result_x_tsv += f"{seq}\t{lpips_distance_list[seq]}\n"
        result_x_std_tsv += f"{seq}\t{lpips_distance_std_list[seq]}\n"
        result_x0_tsv += f"{seq}\t{lpips_distance_list_x0_t[seq]}\n"
        result_x0_std_tsv += f"{seq}\t{lpips_distance_std_list_x0_t[seq]}\n"

    with open(
        os.path.join(
            runner.args.exp,
            f"{(runner.args.config).split('.')[0]}_LPIPS_distance_x.tsv",
        ),
        "w",
    ) as f:
        f.write(result_x_tsv)

    with open(
        os.path.join(
            runner.args.exp,
            f"{(runner.args.config).split('.')[0]}_LPIPS_distance_x_std.tsv",
        ),
        "w",
    ) as f:
        f.write(result_x_std_tsv)

    with open(
        os.path.join(
            runner.args.exp,
            f"{(runner.args.config).split('.')[0]}_LPIPS_distance_x0_t.tsv",
        ),
        "w",
    ) as f:
        f.write(result_x0_tsv)

    with open(
        os.path.join(
            runner.args.exp,
            f"{(runner.args.config).split('.')[0]}_LPIPS_distance_x0_t_std.tsv",
        ),
        "w",
    ) as f:
        f.write(result_x0_std_tsv)
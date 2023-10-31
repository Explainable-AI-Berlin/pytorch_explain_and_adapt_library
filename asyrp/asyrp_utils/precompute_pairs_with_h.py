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
def precompute_pairs_with_h(runner, model, img_path):
    if not os.path.exists(os.path.join(runner.args.exp, "precomputed")):
        os.mkdir(os.path.exists(os.path.join(runner.args.exp, "precomputed")))

    save_path = "_".join(img_path.split(".")[-2].split("/")[-2:])
    save_path = (
        runner.config.data.category
        + "_inv"
        + str(runner.args.n_inv_step)
        + "_"
        + save_path
        + ".pt"
    )
    save_path = os.path.join(runner.args.exp, "precomputed", save_path)

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
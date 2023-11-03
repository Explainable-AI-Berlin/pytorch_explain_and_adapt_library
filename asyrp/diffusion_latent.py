import time
import os
import numpy as np
import torch

from torch import nn
from tqdm import tqdm

from asyrp.utils.diffusion_utils import get_beta_schedule, denoising_step
from asyrp.utils.text_dic import SRC_TRG_TXT_DIC
from asyrp.losses import id_loss
from asyrp.configs.paths_config import DATASET_PATHS
from asyrp.asyrp_utils.set_t_edit_t_addnoise import set_t_edit_t_addnoise
from asyrp.asyrp_utils.save_image import save_image
from asyrp.asyrp_utils.get_pairs import precompute_pairs
from asyrp.asyrp_utils.load_model import load_pretrained_model
from asyrp.asyrp_utils.create_sequence import create_sequence


class Asyrp(object):
    def __init__(self, args, config, device=None):
        # ----------- predefined parameters -----------#
        self.args = args
        self.config = config
        if self.config.data.category == "CUSTOM":
            DATASET_PATHS["custom_train"] = self.args.custom_train_dataset_dir
            DATASET_PATHS["custom_test"] = self.args.custom_test_dataset_dir

        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)

        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.alphas_cumprod = alphas_cumprod

        if self.model_var_type == "fixedlarge":
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))

        elif self.model_var_type == "fixedsmall":
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))

        self.learn_sigma = False  # it will be changed in load_pretrained_model()

        # ----------- Editing txt -----------#
        if self.args.edit_attr is None:
            self.src_txts = self.args.src_txts
            self.trg_txts = self.args.trg_txts

        elif self.args.edit_attr == "attribute":
            pass

        else:
            self.src_txts = SRC_TRG_TXT_DIC[self.args.edit_attr][0]
            self.trg_txts = SRC_TRG_TXT_DIC[self.args.edit_attr][1]

    def run_training(self):
        print("Running Training...")

        # ----------- Losses -----------#
        # We tried to use ID loss and it works well.
        # But it is not used in the paper because it is not necessary.
        # We just leave the code here for future research.
        if self.args.use_id_loss:
            id_loss_func = id_loss.IDLoss().to(self.device)

        else:
            id_loss_func = None

        # Set self.t_edit & self.t_addnoise & return cosine similarity of attribute
        cosine, clip_loss_func = set_t_edit_t_addnoise(
            runner=self,
            LPIPS_th=self.args.lpips_edit_th,
            LPIPS_addnoise_th=self.args.lpips_addnoise_th,
            return_clip_loss=True,
        )
        if not hasattr(self.args, "classifier"):
            counterfactual_loss_func = clip_loss_func.to(self.device)

            # For memory
            for p in counterfactual_loss_func.parameters():
                p.requires_grad = False

            for p in counterfactual_loss_func.model.parameters():
                p.requires_grad = False

        else:
            counterfactual_loss_func = lambda x, target: torch.nn.CrossEntropyLoss()(self.args.classifier(x), target)

        (
            seq_train,
            seq_train_next,
            seq_test,
            seq_test_next,
            seq_test_edit,
        ) = create_sequence(self)

        # ----------- Model -----------#
        model = load_pretrained_model(runner=self)
        optim_param_list = []
        delta_h_dict = {}
        for i in seq_train:
            delta_h_dict[i] = None

        if self.args.train_delta_block:
            model.setattr_layers(self.args.get_h_num)
            print("Setattr layers")
            model = model.to(self.device)
            model = torch.nn.DataParallel(model)

            for i in range(self.args.get_h_num):
                get_h = getattr(model.module, f"layer_{i}")
                optim_param_list = optim_param_list + list(get_h.parameters())

        elif self.args.train_delta_h:
            # h_dim is hard coded to be 512
            # It can be converted to get automatically
            if self.args.ignore_timesteps:
                delta_h_dict[0] = torch.nn.Parameter(
                    torch.randn((512, 8, 8)) * 0.2
                )  # initialization of delta_h

            else:
                for i in seq_train:
                    delta_h_dict[i] = torch.nn.Parameter(
                        torch.randn((512, 8, 8)) * 0.2
                    )  # initialization of delta_h

            model = model.to(self.device)
            model = torch.nn.DataParallel(model)

            for key in delta_h_dict.keys():
                optim_param_list = optim_param_list + [delta_h_dict[key]]

        # optim_ft = torch.optim.Adam(optim_get_h_list, weight_decay=0, lr=self.args.lr_latent_clr)
        optim_ft = torch.optim.SGD(
            optim_param_list, weight_decay=0, lr=self.args.lr_training
        )

        scheduler_ft = torch.optim.lr_scheduler.StepLR(
            optim_ft, step_size=self.args.scheduler_step_size, gamma=self.args.sch_gamma
        )
        print(f"Setting optimizer with lr={self.args.lr_training}")

        # hs_coeff[0] is for original h, hs_coeff[1] is for delta_h
        # if you want to train multiple delta_h at once, you have to modify this part.
        # TODO is this a problem that it is set like this?
        hs_coeff = (1.0, 1.0)

        # ----------- Pre-compute -----------#
        print("Prepare identity latent...")
        """if self.args.load_random_noise:
            # get Random noise xT
            img_lat_pairs_dic = random_noise_pairs(
                runner=self,
                model=model,
                saved_noise=self.args.saved_random_noise,
                save_imgs=self.args.save_precomputed_images,
            )

        else:"""

        # get Real image xT
        img_lat_pairs_dic = precompute_pairs(
            self, model, self.args.save_precomputed_images
        )

        """if self.args.just_precompute:
            # if you just want to precompute, you can stop here.
            print("Pre-computed done.")
            return"""

        """"# if you want to train with specific image, you can use this part.
        if self.args.target_image_id:
            self.args.target_image_id = self.args.target_image_id.split(" ")
            self.args.target_image_id = [int(i) for i in self.args.target_image_id]"""

        save_image_iter = 0
        outer_train_loop(
            self,
            model,
            optim_ft,
            scheduler_ft,
            delta_h_dict,
            hs_coeff,
            cosine,
            counterfactual_loss_func,
            id_loss_func,
            seq_train,
            seq_train_next,
            img_lat_pairs_dic,
            seq_test,
            seq_test_next,
            save_image_iter,
        )

        # ------------------ Test ------------------#
        counterfactual_list = []
        x_lat_tensor = None
        x0_tensor = None

        for step, (x0, _, x_lat, target) in enumerate(img_lat_pairs_dic["test"]):
            if x_lat_tensor is None:
                x_lat_tensor = x_lat
                if self.args.use_x0_tensor:
                    x0_tensor = x0

            else:
                x_lat_tensor = torch.cat((x_lat_tensor, x_lat), dim=0)
                if self.args.use_x0_tensor:
                    x0_tensor = torch.cat((x0_tensor, x0), dim=0)

            if (step + 1) % self.args.bs_train != 0:
                continue

            counterfactual_list.append(
                save_image(
                    self,
                    model,
                    x_lat_tensor,
                    seq_test,
                    seq_test_next,
                    save_x0=self.args.save_x0,
                    save_x_origin=self.args.save_x_origin,
                    x0_tensor=x0_tensor,
                    delta_h_dict=delta_h_dict,
                    folder_dir=self.args.test_image_folder,
                    file_name=f"test_{step}_{self.args.n_iter - 1}",
                    hs_coeff=hs_coeff,
                )
            )

            if step == self.args.n_test_img - 1:
                break

            save_image_iter += 1
            x_lat_tensor = None
            x0_tensor = None

        return counterfactual_list


def outer_train_loop(
    runner,
    model,
    optim_ft,
    scheduler_ft,
    delta_h_dict,
    hs_coeff,
    cosine,
    counterfactual_loss_func,
    id_loss_func,
    seq_train,
    seq_train_next,
    img_lat_pairs_dic,
    seq_test,
    seq_test_next,
    save_image_iter,
):
    # ----------- Training -----------#
    for it_out in range(
        runner.args.start_iter_when_you_use_pretrained, runner.args.n_iter
    ):
        exp_id = os.path.split(runner.args.exp)[-1]
        save_name = os.path.join(runner.args.exp, f"checkpoint_{it_out}.pth")

        # train set

        save_model_iter_from_noise = 0
        """if runner.args.retrain == 0 and os.path.exists(save_name):
            # load checkpoint
            print(f"{save_name} already exists. load checkpoint")
            runner.args.retrain = 0
            optim_ft.load_state_dict(torch.load(save_name)["optimizer"])
            scheduler_ft.load_state_dict(torch.load(save_name)["scheduler"])
            scheduler_ft.step()
            # print lr of now
            print(f"Loaded lr={optim_ft.param_groups[0]['lr']}")
            # get_h_num default is 0;
            if runner.args.train_delta_block:
                for i in range(runner.args.get_h_num):
                    get_h = getattr(model.module, f"layer_{i}")
                    get_h.load_state_dict(torch.load(save_name)[f"{i}"])

            if runner.args.train_delta_h:
                for i in delta_h_dict.keys():
                    delta_h_dict[i] = torch.load(save_name)[f"{i}"]

            continue

        else:"""
        # Unfortunately, ima_lat_pairs_dic does not match with batch_size
        # I'm sorry but you have to get ima_lat_pairs_dic with batch_size == 1
        x_lat_tensor = None
        x0_tensor = None
        inner_train_loop(
            runner,
            model,
            optim_ft,
            scheduler_ft,
            delta_h_dict,
            hs_coeff,
            cosine,
            counterfactual_loss_func,
            id_loss_func,
            seq_train,
            seq_train_next,
            img_lat_pairs_dic,
            seq_test,
            seq_test_next,
            it_out,
            save_image_iter,
            save_model_iter_from_noise,
            save_name,
            exp_id,
            x_lat_tensor,
            x0_tensor,
        )


def inner_train_loop(
    runner,
    model,
    optim_ft,
    scheduler_ft,
    delta_h_dict,
    hs_coeff,
    cosine,
    counterfactual_loss_func,
    id_loss_func,
    seq_train,
    seq_train_next,
    img_lat_pairs_dic,
    seq_test,
    seq_test_next,
    it_out,
    save_image_iter,
    save_model_iter_from_noise,
    save_name,
    exp_id,
    x_lat_tensor,
    x0_tensor,
):
    for step, (x0, _, x_lat, target) in enumerate(img_lat_pairs_dic["train"]):
        if runner.args.target_image_id:
            assert (
                runner.args.bs_train == 1
            ), "target_image_id is only supported for batch_size == 1"
            if not step in runner.args.target_image_id:
                continue

        if x_lat_tensor is None:
            x_lat_tensor = x_lat
            if runner.args.use_x0_tensor:
                x0_tensor = x0

        else:
            x_lat_tensor = torch.cat((x_lat_tensor, x_lat), dim=0)
            if runner.args.use_x0_tensor:
                x0_tensor = torch.cat((x0_tensor, x0), dim=0)

        if (step + 1) % runner.args.bs_train != 0:
            continue

        # LoL. now x_lat_tensor has batch_size == bs_train

        # torch.cuda.empty.cache()
        model.train()
        # For memory
        for p in model.module.parameters():
            p.requires_grad = False

        if runner.args.train_delta_block:
            for i in range(runner.args.get_h_num):
                get_h = getattr(model.module, f"layer_{i}")
                for p in get_h.parameters():
                    p.requires_grad = True

        time_in_start = time.time()

        # original DDIM
        x_origin = x_lat_tensor.to(runner.device)
        # editing by Asyrp
        xt_next = x_lat_tensor.to(runner.device)

        train_step(
            runner,
            model,
            optim_ft,
            seq_train,
            seq_train_next,
            delta_h_dict,
            hs_coeff,
            cosine,
            counterfactual_loss_func,
            id_loss_func,
            step,
            it_out,
            x0,
            target,
            xt_next,
            x_origin,
        )

        # save image
        if (
            runner.args.save_train_image
            and save_image_iter % runner.args.save_train_image_step == 0
            and it_out % runner.args.save_train_image_iter == 0
        ):
            save_image(
                runner,
                model,
                x_lat_tensor,
                seq_test,
                seq_test_next,
                save_x0=runner.args.save_x0,
                save_x_origin=runner.args.save_x_origin,
                x0_tensor=x0_tensor,
                delta_h_dict=delta_h_dict,
                folder_dir=runner.args.training_image_folder,
                file_name=f"train_{step}_{it_out}",
                hs_coeff=hs_coeff,
            )

        if (
            runner.args.save_checkpoint_during_iter
            and save_image_iter % runner.args.save_checkpoint_step == 0
        ):
            dicts = {}
            if runner.args.train_delta_block:
                for i in range(runner.args.get_h_num):
                    get_h = getattr(model.module, f"layer_{i}")
                    dicts[f"{i}"] = get_h.state_dict()
            if runner.args.train_delta_h:
                for key in delta_h_dict.keys():
                    dicts[f"{key}"] = delta_h_dict[key]

            save_name_tmp = (
                save_name.split(".pth")[0]
                + "_"
                + str(save_model_iter_from_noise)
                + ".pth"
            )
            torch.save(dicts, save_name_tmp)
            print(f"Model {save_name_tmp} is saved.")

            save_model_iter_from_noise += 1

        time_in_end = time.time()
        print(f"Training for 1 step {time_in_end - time_in_start:.4f}s")
        if step == runner.args.n_train_img - 1:
            break
        save_image_iter += 1
        x_lat_tensor = None
        x0_tensor = None

        # ------------------ Save ------------------#
        dicts = {}
        if runner.args.train_delta_block:
            for i in range(runner.args.get_h_num):
                get_h = getattr(model.module, f"layer_{i}")
                dicts[f"{i}"] = get_h.state_dict()

        if runner.args.train_delta_h:
            for key in delta_h_dict.keys():
                dicts[f"{key}"] = delta_h_dict[key]

        dicts["optimizer"] = optim_ft.state_dict()
        dicts["scheduler"] = scheduler_ft.state_dict()
        torch.save(dicts, save_name)
        print(f"Model {save_name} is saved.")
        scheduler_ft.step()

        if runner.args.save_checkpoint_only_last_iter:
            if os.path.exists(f"checkpoint/{exp_id}_{it_out - 1}.pth"):
                os.remove(f"checkpoint/{exp_id}_{it_out - 1}.pth")


def train_step(
    runner,
    model,
    optim_ft,
    seq_train,
    seq_train_next,
    delta_h_dict,
    hs_coeff,
    cosine,
    counterfactual_loss_func,
    id_loss_func,
    step,
    it_out,
    x0,
    target,
    xt_next,
    x_origin,
):
    # Finally, go into training
    with tqdm(total=len(seq_train), desc=f"training iteration") as progress_bar:
        for t_it, (i, j) in enumerate(
            zip(reversed(seq_train), reversed(seq_train_next))
        ):
            optim_ft.zero_grad()
            t = (torch.ones(runner.args.bs_train) * i).to(runner.device)
            t_next = (torch.ones(runner.args.bs_train) * j).to(runner.device)

            # step 1: Asyrp
            xt_next, x0_t, _, _ = denoising_step(
                xt_next.detach(),
                t=t,
                t_next=t_next,
                models=model,
                logvars=runner.logvar,
                b=runner.betas,
                sampling_type=runner.args.sample_type,
                eta=0.0,
                learn_sigma=runner.learn_sigma,
                index=0
                if not (
                    runner.args.image_space_noise_optim
                    or runner.args.image_space_noise_optim_delta_block
                )
                else None,
                t_edit=runner.t_edit,
                hs_coeff=hs_coeff,
                delta_h=delta_h_dict[0]
                if (runner.args.ignore_timesteps and runner.args.train_delta_h)
                else delta_h_dict[t[0].item()],
                ignore_timestep=runner.args.ignore_timesteps,
            )
            # when train delta_block, delta_h is None (ignored)
            # step 2: DDIM
            with torch.no_grad():
                x_origin, x0_t_origin, _, _ = denoising_step(
                    x_origin.detach(),
                    t=t,
                    t_next=t_next,
                    models=model,
                    logvars=runner.logvar,
                    b=runner.betas,
                    sampling_type=runner.args.sample_type,
                    eta=0.0,
                    learn_sigma=runner.learn_sigma,
                )

            progress_bar.update(1)

            loss = 0
            loss_id = 0
            loss_l1 = 0
            loss_clr = 0
            loss_classifier = 0

            # L1 loss
            loss_l1 += nn.L1Loss()(x0_t, x0_t_origin)

            # Following DiffusionCLIP, we use direction clip loss as below
            if not runner.args.classifier:
                loss_classifier = -torch.log(
                    (
                        2
                        - counterfactual_loss_func(
                            x0, runner.src_txts[0], x0_t, runner.trg_txts[0]
                        )
                    )
                    / 2
                )

            else:
                loss_classifier = counterfactual_loss_func(x0_t, torch.tensor([target]).to(x0_t.device))

            if runner.args.use_id_loss:
                # We don't use this.
                loss_id += torch.mean(id_loss_func(x0_t, x0_t_origin))

            loss += runner.args.id_loss_w * loss_id
            loss += runner.args.l1_loss_w * loss_l1 * cosine
            loss += runner.args.clip_loss_w * loss_classifier

            loss.backward()
            optim_ft.step()

            progress_bar.set_description(
                f"{step}-{it_out}: loss_clr: {loss_clr:.3f} loss_l1: {loss_l1:.3f}"
                + f"loss_id: {loss_id:.3f} loss_classifier:{loss_classifier} loss: {loss:.3f} "
            )

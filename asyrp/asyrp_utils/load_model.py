import torch
import os
import torchvision

from asyrp.models.ddpm.diffusion import DDPM
from asyrp.models.improved_ddpm.script_util import i_DDPM
from asyrp.models.guided_diffusion.script_util import guided_Diffusion
from asyrp.configs.paths_config import MODEL_PATHS
from asyrp.models.guided_diffusion.script_util import create_gaussian_diffusion

def load_pretrained_model(runner):
        # ----------- Model -----------#
        if runner.config.data.dataset == "LSUN":
            if runner.config.data.category == "bedroom":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/bedroom.ckpt"

            elif runner.config.data.category == "church_outdoor":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/church_outdoor.ckpt"

        elif runner.config.data.dataset in ["CelebA_HQ", "CUSTOM", "CelebA_HQ_Dialog"]:
            url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"

        elif runner.config.data.dataset in ["FFHQ", "AFHQ", "IMAGENET", "MetFACE", "CelebA_HQ_P2"]:
            # get the model ["FFHQ", "AFHQ", "MetFACE"] from
            # https://1drv.ms/u/s!AkQjJhxDm0Fyhqp_4gkYjwVRBe8V_w?e=Et3ITH
            # reference : ILVR (https://arxiv.org/abs/2108.02938), P2 weighting (https://arxiv.org/abs/2204.00227)
            # reference github : https://github.com/jychoi118/ilvr_adm , https://github.com/jychoi118/P2-weighting

            # get the model "IMAGENET" from
            # https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt
            # reference : ADM (https://arxiv.org/abs/2105.05233)
            pass

        else:
            # if you want to use LSUN-horse, LSUN-cat -> https://github.com/openai/guided-diffusion
            # if you want to use CUB, Flowers -> https://1drv.ms/u/s!AkQjJhxDm0Fyhqp_4gkYjwVRBe8V_w?e=Et3ITH
            raise ValueError

        if runner.config.data.dataset in ["CelebA_HQ", "LSUN", "CelebA_HQ_Dialog"]:
            model = DDPM(runner.config)
            if runner.args.model_path:
                init_ckpt = torch.load(runner.args.model_path)

            else:
                init_ckpt = torch.hub.load_state_dict_from_url(
                    url, map_location=runner.device
                )

            runner.learn_sigma = False
            print("Original diffusion Model loaded.")

        elif runner.config.data.dataset in ["FFHQ", "AFHQ", "IMAGENET"]:
            model = i_DDPM(
                runner.config.data.dataset
            )  # Get_h(runner.config, model="i_DDPM", layer_num=runner.args.get_h_num) #
            if runner.args.model_path:
                init_ckpt = torch.load(runner.args.model_path)

            else:
                init_ckpt = torch.load(MODEL_PATHS[runner.config.data.dataset])

            runner.learn_sigma = True
            print("Improved diffusion Model loaded.")

        elif runner.config.data.dataset in ["MetFACE", "CelebA_HQ_P2"]:
            model = guided_Diffusion(runner.config.data.dataset)
            #init_ckpt = torch.load(MODEL_PATHS[runner.config.data.dataset])
            init_ckpt = torch.load(runner.args.model_path)
            runner.learn_sigma = True

        else:
            print("Not implemented dataset")
            raise ValueError

        model.load_state_dict(init_ckpt, strict=False)
        print(runner.config.data.dataset)
        diffusion = create_gaussian_diffusion(learn_sigma=True)
        model.h_only = True
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        samples = diffusion.p_sample_loop(model, [3, 3, 256, 256])
        torchvision.utils.save_image(samples, os.path.join(runner.args.exp, "generator_samples.png"))
        model.h_only = False

        return model
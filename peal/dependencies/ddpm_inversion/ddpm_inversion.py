import torch
import torchvision

from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler

from peal.configs.data.data_config import DataConfig
from peal.configs.editors.ddpm_inversion_config import DDPMInversionConfig
from peal.data.datasets import Image2MixedDataset
from peal.dependencies.ddpm_inversion.prompt_to_prompt.ptp_classes import AttentionStore
from peal.dependencies.ddpm_inversion.prompt_to_prompt.ptp_utils import (
    register_attention_control,
)
from peal.dependencies.ddpm_inversion.ddm_inversion.inversion_utils import (
    inversion_forward_process,
    inversion_reverse_process,
)
from peal.global_utils import load_yaml_config


class DDPMInversion:
    def __init__(self, config=DDPMInversionConfig()):
        self.config = load_yaml_config(config)
        #self.config.data = DataConfig(**self.config.data)
        if not self.config.data is None and self.config.data.normalization is None:
            self.project_to_pytorch_default = lambda x: (
                x * torch.tensor(self.config.data.normalization[1])
                + torch.tensor(self.config.data.normalization[0])
            )
            self.project_from_pytorch_default = lambda x: (
                x - torch.tensor(self.config.data.normalization[0])
            ) / torch.tensor(self.config.data.normalization[1])

        else:
            self.project_from_pytorch_default = lambda x: x
            self.project_to_pytorch_default = lambda x: x

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.pipe = StableDiffusionPipeline.from_pretrained(self.config.model_id).to(
            self.device
        )
        self.pipe.scheduler = DDIMScheduler.from_config(
            self.config.model_id, subfolder="scheduler"
        )
        self.pipe.scheduler.set_timesteps(self.config.num_diffusion_steps)

    def run(self, x, prompt_tar_list, prompt_src):
        x = self.project_from_pytorch_default(x)
        # TODO do i have to upsample here?
        x0 = torchvision.transforms.Resize([512, 512])(
            torch.clone(x).to(self.device)
        )  # load_512(image_path, *offsets, device)

        # vae encode image
        w0 = (self.pipe.vae.encode(x0).latent_dist.mode() * 0.18215).float()

        # find Zs and wts - forward process
        wt, zs, wts = inversion_forward_process(
            self.pipe,
            w0,
            etas=self.config.eta,
            prompt=prompt_src,
            cfg_scale=self.config.cfg_scale_src,
            prog_bar=True,
            num_inference_steps=self.config.num_diffusion_steps,
        )

        classifier_loss = None

        # Check if number of words in encoder and decoder text are equal
        src_tar_len_eq = False
        # reverse process (via Zs and wT)
        controller = AttentionStore()
        register_attention_control(self.pipe, controller)
        w0, _ = inversion_reverse_process(
            self.pipe,
            xT=wts[self.config.num_diffusion_steps - self.config.skip],
            etas=self.config.eta,
            prompts=prompt_tar_list,
            cfg_scales=[self.config.cfg_scale_tar],
            prog_bar=True,
            zs=zs[: (self.config.num_diffusion_steps - self.config.skip)],
            controller=controller,
            classifier=classifier_loss,
        )

        # vae decode image
        x0_dec = self.pipe.vae.decode(1 / 0.18215 * w0).sample

        if x0_dec.dim() < 4:
            x0_dec = x0_dec[None, :, :, :]

        x_counterfactuals = torch.clone(x0_dec).detach().cpu()
        x_counterfactuals = torchvision.transforms.Resize(x.shape[2:])(
            x_counterfactuals
        )
        x_counterfactuals = torch.clamp(self.project_to_pytorch_default(
            x_counterfactuals
        ), 0, 1)
        print([x_counterfactuals.min(), x_counterfactuals.max()])

        return x_counterfactuals

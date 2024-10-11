import torch
import torchvision

from peal.global_utils import high_contrast_heatmap


def visualize_step(x_in, z_predictor_original, img_predictor, z, z_cuda, img_default, filename):
    heatmap_high_contrast = []
    for it in range(x_in.shape[0]):
        heatmap_high_contrast.append(
            high_contrast_heatmap(
                x_in[it], z_predictor_original[it].detach().cpu()
            )[0]
        )

    ref = torch.zeros_like(x_in[0])
    gradient_decoded = []
    for it in range(x_in.shape[0]):
        gradient_decoded.append(
            high_contrast_heatmap(
                ref, img_predictor.grad[it].detach().cpu()
            )[0]
        )

    gradient_z = []
    for it in range(x_in.shape[0]):
        gradient_z.append(
            high_contrast_heatmap(
                ref, z[0].grad[it].detach().cpu()
            )[0]
        )

    save_tensor = torch.cat(
        [
            x_in,
            torch.ones_like(x_in),
            torch.stack(heatmap_high_contrast),
            torch.ones_like(x_in),
            z_predictor_original.detach().cpu(),
            torch.ones_like(x_in),
            z_cuda.detach().cpu(),
            torch.ones_like(x_in),
            img_default.detach().cpu(),
            torch.ones_like(x_in),
            torch.stack(gradient_decoded),
            torch.ones_like(x_in),
            torch.stack(gradient_z),
        ]
    )
    torchvision.utils.save_image(save_tensor, fp=filename, nrow=x_in.shape[0])
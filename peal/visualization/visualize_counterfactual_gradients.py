import torch
import torchvision

from peal.global_utils import high_contrast_heatmap


def visualize_step(x, z, z_noisy, img_predictor, boolmask, filename):
    heatmap_high_contrast = []
    z_classifier = z[0].data.detach().cpu()
    for it in range(x.shape[0]):
        heatmap_high_contrast.append(
            high_contrast_heatmap(
                x[it], z_classifier[it]
            )[0]
        )

    ref = torch.zeros_like(x[0])
    gradient_img = []
    for it in range(x.shape[0]):
        gradient_img.append(
            high_contrast_heatmap(
                ref, img_predictor.grad[it].detach().cpu()
            )[0]
        )

    gradient_z = []
    for it in range(x.shape[0]):
        gradient_z.append(
            high_contrast_heatmap(
                ref, z[0].grad[it].detach().cpu()
            )[0]
        )

    save_tensor = torch.cat(
        [
            x,
            torch.ones_like(x),
            torch.stack(heatmap_high_contrast),
            torch.ones_like(x),
            z_classifier,
            torch.ones_like(x),
            torch.stack(z_noisy[0].detach().cpu()),
            torch.ones_like(x),
            torch.stack(img_predictor.cpu().detach()),
            torch.ones_like(x),
            torch.stack(gradient_img),
            torch.ones_like(x),
            torch.stack(gradient_z),
            torch.ones_like(x),
            boolmask.detach().cpu(),
        ]
    )
    torchvision.utils.save_image(save_tensor, fp=filename, nrow=x.shape[0])
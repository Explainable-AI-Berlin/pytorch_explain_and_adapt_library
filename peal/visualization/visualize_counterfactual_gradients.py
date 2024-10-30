import torch
import torchvision

from peal.global_utils import high_contrast_heatmap


def visualize_step(x, z, clean_img_old, z_noisy, img_predictor, pe, boolmask, filename, boolmask_in):
    clean_img_new = 0.5 * z[0].data.detach().cpu() + 0.5
    original_vs_counterfactual = []
    for it in range(x.shape[0]):
        original_vs_counterfactual.append(
            high_contrast_heatmap(
                x[it], clean_img_old[it]
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

    if boolmask.shape[1] == 1:
        boolmask = torch.cat(3 * [boolmask.detach().cpu()], dim=1)

    save_tensor = torch.cat(
        [
            x,
            torch.ones_like(x),
            clean_img_old,
            torch.ones_like(x),
            z_noisy.detach().cpu(),
            torch.ones_like(x),
            img_predictor.cpu().detach(),
            torch.ones_like(x),
            torch.stack(gradient_img),
            torch.ones_like(x),
            torch.stack(gradient_z),
            torch.ones_like(x),
            pe,
            torch.ones_like(x),
            torch.stack(original_vs_counterfactual),
            torch.ones_like(x),
            boolmask_in,
            torch.ones_like(x),
            boolmask,
            torch.ones_like(x),
            clean_img_new,
        ]
    )

    torchvision.utils.save_image(save_tensor, fp=filename, nrow=x.shape[0])
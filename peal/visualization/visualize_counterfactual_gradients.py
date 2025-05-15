import torch
import torchvision

from peal.global_utils import high_contrast_heatmap
from typing import Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def create_label_image(text, image_size, font_size=50):
    """
    Create a tensor containing a text label image.
    Args:
        text: Text to display
        image_size: Tuple (C, H, W) of the target image size
        font_size: Size of the font
    Returns:
        label_tensor: Tensor of shape (1, C, H, W)
    """
    transform = torchvision.transforms.ToTensor()
    C, H, W = image_size
    img = Image.new("RGB" if C == 3 else "L", (W, H), color="white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
        # text_bbox = draw.textbbox((0, 0), text, font=font)
        # text_w = text_bbox[2] - text_bbox[0]
        # text_h = text_bbox[3] - text_bbox[1]
        # x = (W - text_w) / 2
        # y = (H - text_h) / 2
    except IOError:
        font = ImageFont.load_default()
    w = draw.textlength(text, font=font)
    h = font_size
    x = (W - w) / 2
    y = (H - h) / 2

    draw.text((x, y), text, fill="black" if C == 3 else 0, font=font)

    label_tensor = transform(img)
    return label_tensor.unsqueeze(0)


@torch.no_grad()
def visualize_step(
    x_original: torch.Tensor,
    z_encoded: torch.Tensor,
    img_predictor: torch.Tensor,
    pe: torch.Tensor,
    filename: str,
    z: Optional[torch.Tensor] = None,
    clean_img_old: Optional[torch.Tensor] = None,
    boolmask: Optional[torch.Tensor] = None,
    boolmask_in: Optional[torch.Tensor] = None,
    latent_decoder=None,
    latent_encoder=None,
    best_z=None,
    best_mask=None,
):
    transform = torchvision.transforms.Resize(x_original.size()[2])
    original_vs_counterfactual = []
    for it in range(x_original.shape[0]):
        if x_original.size() != pe.size():
            pe = transform(pe)
        original_vs_counterfactual.append(high_contrast_heatmap(x_original[it], pe[it])[0])

    ref = torch.zeros_like(x_original[0])
    gradient_img = []
    for it in range(x_original.shape[0]):
        gradient_img.append(
            high_contrast_heatmap(ref, img_predictor.grad[it].detach().cpu())[0]
        )

    if z:
        gradient_z = []
        ref = torch.zeros_like(z[0][0])
        clean_img_new = z[0].data.detach().cpu()
        clean_img_new = (
            transform(clean_img_new) if z[0].size() != x_original.size() else clean_img_new
        )
        for it in range(x_original.shape[0]):
            grad_heatmap = high_contrast_heatmap(ref, -z[0].grad[it].detach().cpu())[0]
            if z[0].size() != x_original.size():
                if latent_decoder:
                    with torch.no_grad():
                        grad_heatmap = latent_decoder(grad_heatmap.to(z[0].device))

            grad_heatmap = transform(grad_heatmap)

            gradient_z.append(grad_heatmap)

    best_img = best_z.data.detach().cpu()

    if clean_img_old.size() != x_original.size():
        clean_img_old = transform(clean_img_old)

    if boolmask is not None:
        if boolmask.size()[2] != x_original.size()[2]:
            if latent_decoder:
                with torch.no_grad():
                    boolmask = latent_decoder(boolmask)

            boolmask = transform(boolmask)

        if boolmask.shape[1] == 1:
            boolmask = torch.cat(3 * [boolmask.detach().cpu()], dim=1)

    if best_mask.size()[2] != x_original.size()[2]:
        if latent_decoder:
            with torch.no_grad():
                best_mask = latent_decoder(best_mask)

        best_mask = transform(best_mask)

    if best_mask.shape[1] == 1:
        best_mask = torch.cat(3 * [best_mask.detach().cpu()], dim=1)

    if boolmask_in.size()[2] != x_original.size()[2]:
        boolmask_in = transform(boolmask_in)

    if z is None and boolmask is None:
        components = [
            (x_original, "Input (x)"),
            (z_encoded.detach().cpu(), "Encoded Z"),
            (img_predictor.cpu().detach(), "Predictor Img"),
            (torch.stack(gradient_z), "Z Gradients"),
            (pe, "PE"),
            (torch.stack(original_vs_counterfactual), "Original vs CF"),
        ]

    else:
        components = [
            (x_original, "Input (x)"),
            (clean_img_old, "Clean Old"),
            (z_encoded.detach().cpu(), "Encoded Z"),
            (img_predictor.cpu().detach(), "Predictor Img"),
            (torch.stack(gradient_img), "Img Gradients"),
            (torch.stack(gradient_z), "Z Gradients"),
            (pe, "PE"),
            (torch.stack(original_vs_counterfactual), "Original vs CF"),
            (boolmask_in, "Input Mask"),
            (boolmask, "Boolmask"),
            (clean_img_new, "Clean New"),
            (best_mask, "Best Mask"),
            (best_img, "Best Image"),
        ]

    rows = []
    for tensor, name in components:
        B, C, H, W = tensor.shape
        label = create_label_image(name, (C, H, W), font_size=max(H // 15, 50))
        label = label.to(tensor.device)
        row = torch.cat([label, tensor], dim=0)  # Shape: (B+1, C, H, W)
        rows.append(row)

    save_tensor = torch.cat(rows, dim=0)
    torchvision.utils.save_image(save_tensor, fp=filename, nrow=B + 1)

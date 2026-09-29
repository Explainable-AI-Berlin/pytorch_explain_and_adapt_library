"""Debug grids for gradient-based counterfactual search.

While a counterfactual explainer optimises an image (or a generator latent)
against the predictor, :func:`visualize_step` writes one labelled image grid
per step showing the input, the encoded/decoded images, the pixel and latent
gradients as heatmaps, the masks in play and the best counterfactual found so
far. :func:`create_label_image` renders the row captions as image tensors so
the whole grid can be saved with ``torchvision.utils.save_image``.
"""

import torch
import torchvision

from peal.global_utils import high_contrast_heatmap
from typing import Optional
from PIL import Image, ImageDraw, ImageFont


def create_label_image(text, image_size, font_size=50):
    """Render a text caption as an image tensor of a given size.

    The text is drawn centred in black on a white canvas (RGB when ``C == 3``,
    greyscale otherwise) with ``arial.ttf`` if available, else PIL's default
    font.

    Parameters
    ----------
    text : str
        Caption to draw.
    image_size : tuple of int
        ``(C, H, W)`` of the target image.
    font_size : int, optional
        Font size in pixels. Default 50.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(1, C, H, W)`` with values in ``[0, 1]``.
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
    img_predictor_unnormalized: torch.Tensor,
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
    """Save a labelled image grid summarising one counterfactual search step.

    Each row is one quantity (input, encoded image, predictor input, gradient
    heatmaps, masks, best image so far) prefixed by a caption tile; the
    columns are the batch elements. Latent-resolution tensors are decoded with
    ``latent_decoder`` when given and resized to the input resolution.
    Gradients are read from ``.grad`` of ``img_predictor_unnormalized`` and of
    ``z[0]``, so the caller must have run ``backward()`` first. The short grid
    (no image gradients or masks) is written when both ``z`` and ``boolmask``
    are ``None``; note that ``z`` is tested with ``if z:`` and the code that
    computes ``gradient_z`` only runs when ``z`` is truthy.

    Parameters
    ----------
    x_original : torch.Tensor
        Original images, shape ``(B, C, H, W)``.
    z_encoded : torch.Tensor
        Generator encoding of the input, decoded back to image space.
    img_predictor : torch.Tensor
        Current image as fed to the predictor.
    img_predictor_unnormalized : torch.Tensor
        Leaf tensor of the predictor input whose ``.grad`` is drawn as the
        "Img Gradients" row.
    pe : torch.Tensor
        Current counterfactual image, resized to the input size if needed.
    filename : str
        Path the grid is written to.
    z : list of torch.Tensor, optional
        Optimised latents; ``z[0]`` supplies the "Clean New" image and its
        ``.grad`` the "Z Gradients" row.
    clean_img_old : torch.Tensor, optional
        Image before the current update.
    boolmask, boolmask_in : torch.Tensor, optional
        Current and input masks; single-channel masks are repeated to RGB.
    latent_decoder : callable, optional
        Decodes latent-resolution tensors to image resolution.
    latent_encoder : callable, optional
        Unused; kept for call-site symmetry.
    best_z : torch.Tensor
        Best counterfactual image found so far (shown as "Best Image").
    best_mask : torch.Tensor
        Mask belonging to ``best_z``.
    """
    transform = torchvision.transforms.Resize(x_original.size()[2])
    original_vs_counterfactual = []
    for it in range(x_original.shape[0]):
        if x_original.size() != pe.size():
            pe = transform(pe)
        original_vs_counterfactual.append(
            high_contrast_heatmap(x_original[it], pe[it])[0]
        )

    ref = torch.zeros_like(x_original[0])
    gradient_img = []
    for it in range(x_original.shape[0]):
        gradient_img.append(
            high_contrast_heatmap(
                ref, img_predictor_unnormalized.grad[it].detach().cpu()
            )[0]
        )

    if z:
        gradient_z = []
        ref = torch.zeros_like(z[0][0])
        clean_img_new = z[0].data.detach().cpu()
        clean_img_new = (
            transform(clean_img_new)
            if z[0].size() != x_original.size()
            else clean_img_new
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

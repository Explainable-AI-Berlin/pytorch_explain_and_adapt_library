"""PIL/NumPy helpers that assemble PEAL's annotated image grids.

Explainers and teachers render feedback collages as columns: a rotated title,
then one image per row with a caption underneath. Columns of checkbox icons
(``<resource_dir>/imgs/checkbox_right.png`` / ``checkbox_wrong.png``) mark
boolean per-row facts such as "flipped". :func:`make_image_grid` concatenates
such columns into one image; the other functions build the pieces. Image
tensors are ``[N, C, H, W]`` floats in ``[0, 1]``.
"""

import numpy as np
import os
import torch

from PIL import Image, ImageFont, ImageDraw

from peal.global_utils import get_project_resource_dir


def zip_tensors(tensor_list):
    """Interleave several image batches sample by sample along the width.

    Parameters
    ----------
    tensor_list : list of torch.Tensor
        Batches of shape ``[N, 3, H, W_j]`` with a common ``N`` and ``H``.

    Returns
    -------
    torch.Tensor
        ``[N, 3, H, sum(W_j) + 5 * (len(tensor_list) - 1)]``: sample ``i`` of
        every batch placed side by side, separated by 5-pixel white bars.
    """
    padding = torch.ones([3, tensor_list[0].shape[2], 5])
    tensor_list_out = []
    for i in range(tensor_list[0].shape[0]):
        tensor_inner_list = []
        for j in range(len(tensor_list)):
            tensor_inner_list.append(tensor_list[j][i])
            if not j == len(tensor_list) - 1:
                tensor_inner_list.append(padding)

        tensor_list_out.append(torch.cat(tensor_inner_list, axis=2))

    return torch.stack(tensor_list_out, axis=0)


def bool_list_to_checkboxes(bool_list, height):
    """Render a list of booleans as a batch of checkbox icons.

    Parameters
    ----------
    bool_list : list of bool or indexable tensor
        One flag per row; a tensor is converted element-wise.
    height : int
        Row height in pixels; the 24x24 icon is centred vertically with white
        padding.

    Returns
    -------
    torch.Tensor
        ``[N, 3, height, 24]`` floats in ``[0, 1]``; ``True`` rows show
        ``checkbox_right.png``, ``False`` rows ``checkbox_wrong.png`` from the
        project resource directory.
    """
    if not isinstance(bool_list, list):
        bool_list = list(map(lambda idx: bool_list[idx], range(bool_list.shape[0])))

    padding_top = np.ones([int((height - 24) / 2), 24, 3], dtype=np.uint8) * 255
    padding_bottom = (
        np.ones([int((height - 24) / 2 + (height - 24) % 2), 24, 3], dtype=np.uint8)
        * 255
    )
    resource_dir = get_project_resource_dir()
    checkbox_right = Image.open(
        os.path.join(resource_dir, "imgs", "checkbox_right.png")
    )
    checkbox_right = checkbox_right.resize((24, 24))
    checkbox_right = (
        np.concatenate([padding_top, np.array(checkbox_right), padding_bottom], axis=0)
        / 255.0
    )
    checkbox_wrong = Image.open(
        os.path.join(resource_dir, "imgs", "checkbox_wrong.png")
    )
    checkbox_wrong = checkbox_wrong.resize((24, 24))
    checkbox_wrong = (
        np.concatenate([padding_top, np.array(checkbox_wrong), padding_bottom], axis=0)
        / 255.0
    )
    checkbox_list = []
    # convert list of bools to list of checkboxes
    for i in range(len(bool_list)):
        if bool_list[i]:
            checkbox_list.append(checkbox_right)

        else:
            checkbox_list.append(checkbox_wrong)

    return torch.tensor(np.stack(checkbox_list, axis=0).transpose(0, 3, 1, 2))


def embed_text_in_image(text, width, height):
    """Draw ``text`` centred on a white ``width`` x ``height`` RGB canvas.

    Uses PIL's built-in default font, so the text is small and does not scale
    with the canvas.

    Returns
    -------
    PIL.Image.Image
        The canvas with the black text.
    """
    # add title to center of image
    # create Image that contains title
    image = Image.new("RGB", (width, height), (255, 255, 255))

    # Create an ImageDraw object
    draw = ImageDraw.Draw(image)

    # Add text to image
    font = ImageFont.load_default()
    bbox = font.getbbox(text)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    # calculate x,y coordinate for text
    x = (image.width - w) / 2
    y = (image.height - h) / 2

    draw.text((x, y), text, fill=(0, 0, 0), font=font)

    return image


def make_column(images, labels, title):
    """Stack images vertically into one titled, captioned column.

    Parameters
    ----------
    images : torch.Tensor
        ``[N, 3, H, W]`` floats in ``[0, 1]`` on the CPU.
    labels : list of str
        Caption drawn under each image; ``N`` entries.
    title : str
        Column title, rendered rotated by 90 degrees above the first image.

    Returns
    -------
    numpy.ndarray
        ``uint8`` RGB array of shape
        ``[(N + 2) * (H + H // 2) + H // 2, W + 2 * (H // 8), 3]`` with a
        white background. The height depends only on ``N`` and ``H``, which is
        what lets :func:`make_image_grid` concatenate columns.
    """
    # Create a new image with a white background
    x_padding = int(images.shape[2] / 8)
    y_padding = int(images.shape[2] / 2)
    column_height = (images.shape[0] + 2) * (images.shape[2] + y_padding) + y_padding
    column_width = images.shape[3] + 2 * x_padding
    output_image = Image.new("RGB", (column_width, column_height), (255, 255, 255))

    # Set x,y coordinates for each image
    x = x_padding
    y = y_padding

    # add title to center of image
    title_image = embed_text_in_image(title, 2 * images.shape[2], images.shape[3])

    # rotate title image
    title_image = title_image.rotate(90, expand=True)

    # fit title image into whole image
    output_image.paste(title_image, (x, y))
    y += np.array(title_image).shape[0] + y_padding

    # Iterate through the images and labels, displaying each one
    for img_idx in range(images.shape[0]):
        # convert image to PIL format
        img = np.array(255 * images[img_idx].numpy(), dtype=np.uint8)
        img = np.transpose(img, (1, 2, 0))
        im = Image.fromarray(img, "RGB")
        output_image.paste(im, (x, y))
        # print(str([labels[img_idx], int(y_padding / 2), img.shape[0]]))
        label_image = embed_text_in_image(
            labels[img_idx], img.shape[1], int(y_padding / 2)
        )
        output_image.paste(label_image, (x, y + img.shape[0] + int(y_padding / 4)))
        y += img.shape[0] + y_padding

    # save image
    return np.array(output_image)


def make_image_grid(checkbox_dict, image_dicts):
    """Compose checkbox columns and image columns into a single grid image.

    Parameters
    ----------
    checkbox_dict : dict
        Maps a column title to a list of booleans (one per row); each becomes
        a column of checkbox icons without captions.
    image_dicts : dict
        Maps a column title to a pair ``(images, labels)`` where ``images`` is
        an ``[N, 3, H, W]`` tensor or a list of such tensors (interleaved with
        :func:`zip_tensors`) and ``labels`` holds the ``N`` captions.

    Returns
    -------
    PIL.Image.Image
        All columns concatenated horizontally, checkbox columns first. Every
        column must share ``N`` and the image height ``H``.
    """
    # create column for grid
    columns = []
    image_height = list(image_dicts.values())[0][0].shape[2]

    # deal with the checkboxes
    for key in checkbox_dict.keys():
        checkbox_images = bool_list_to_checkboxes(checkbox_dict[key], image_height)
        columns.append(
            make_column(
                checkbox_images,
                list(map(lambda x: "", range(len(checkbox_dict[key])))),
                key,
            )
        )

    for key in image_dicts.keys():
        try:
            columns.append(
                make_column(
                    (
                        image_dicts[key][0]
                        if not isinstance(image_dicts[key][0], list)
                        else zip_tensors(image_dicts[key][0])
                    ),
                    image_dicts[key][1],
                    key,
                )
            )
        except Exception:
            raise

    return Image.fromarray(np.concatenate(columns, axis=1))

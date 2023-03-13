import numpy as np
import os
import torch

from PIL import Image, ImageFont, ImageDraw

from peal.utils import get_project_resource_dir

def zip_tensors(tensor_list):
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
    if not isinstance(bool_list, list):
        bool_list = list(map(lambda idx: bool_list[idx], range(bool_list.shape[0])))
        
    padding_top = np.ones([int((height - 24) / 2), 24, 3], dtype=np.uint8) * 255
    padding_bottom = np.ones(
        [int((height - 24) / 2 + (height - 24) % 2), 24, 3], dtype=np.uint8) * 255
    resource_dir = get_project_resource_dir()
    checkbox_right = Image.open(os.path.join(
        resource_dir, 'imgs', 'checkbox_right.png'))
    checkbox_right = checkbox_right.resize((24, 24))
    checkbox_right = np.concatenate([padding_top, np.array(checkbox_right), padding_bottom], axis=0) / 255.0
    checkbox_wrong = Image.open(os.path.join(resource_dir, 'imgs', 'checkbox_wrong.png'))
    checkbox_wrong = checkbox_wrong.resize((24, 24))
    checkbox_wrong = np.concatenate(
        [padding_top, np.array(checkbox_wrong), padding_bottom], axis=0) / 255.0
    checkbox_list = []
    # convert list of bools to list of checkboxes
    for i in range(len(bool_list)):
        if bool_list[i]:
            checkbox_list.append(checkbox_right)

        else:
            checkbox_list.append(checkbox_wrong)
    
    return torch.tensor(np.stack(checkbox_list, axis=0).transpose(0, 3, 1, 2))


def embed_text_in_image(text, width, height):
    # add title to center of image
    # create Image that contains title
    image = Image.new(
        "RGB",
        (width, height),
        (255, 255, 255)
    )

    # Create an ImageDraw object
    draw = ImageDraw.Draw(image)

    # Add text to image
    font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMono.ttf', size = 10)
    try:
        w, h = draw.textsize(text, font)

    except Exception:
        from IPython.core.debugger import set_trace
        set_trace()

    # calculate x,y cordinate for text
    x = (image.width-w)/2
    y = (image.height-h)/2

    draw.text((x, y), text, fill=(0, 0, 0), font=font)

    return image


def make_column(images, labels, title):
    # Create a new image with a white background
    x_padding = int(images.shape[2] / 8)
    y_padding = int(images.shape[2] / 2)
    column_height = (images.shape[0] + 2) * (images.shape[2] + y_padding) + y_padding
    column_width = images.shape[3] + 2 * x_padding
    output_image = Image.new(
        "RGB",
        (column_width, column_height),
        (255, 255, 255)
    )

    # Set x,y cordinates for each images
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
        im = Image.fromarray(img, 'RGB')
        output_image.paste(im, (x, y))
        #print(str([labels[img_idx], int(y_padding / 2), img.shape[0]]))
        label_image = embed_text_in_image(labels[img_idx], img.shape[1], int(y_padding / 2))
        output_image.paste(label_image, (x, y + img.shape[0] + int(y_padding/4)))
        y += img.shape[0] + y_padding

    # save image
    return np.array(output_image)


def make_image_grid(checkbox_dict, image_dicts):
    # create column for grid
    columns = []
    image_height = list(image_dicts.values())[0][0].shape[2]

    # deal with the checkboxes
    for key in checkbox_dict.keys():
        checkbox_images = bool_list_to_checkboxes(
            checkbox_dict[key],
            image_height
        )
        columns.append(make_column(checkbox_images, list(map(lambda x: '', range(len(checkbox_dict[key])))), key))
    
    for key in image_dicts.keys():
        columns.append(make_column(
            image_dicts[key][0] if not isinstance(image_dicts[key][0], list) else zip_tensors(image_dicts[key][0]),
            image_dicts[key][1],
            key
        ))

    return Image.fromarray(np.concatenate(columns, axis=1))

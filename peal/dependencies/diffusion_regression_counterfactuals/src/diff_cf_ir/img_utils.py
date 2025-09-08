from typing import Union
import PIL.Image
import numpy as np
import scipy
import pandas as pd
import shutil
import os
from tqdm import tqdm
from multiprocessing import Pool


def create_cropped_images_old(root_path, output_path):
    """Processes the dataset to create the image files cropped by square bounding boxes."""
    os.makedirs(output_path, exist_ok=True)

    for anno_file in [
        "csvs/imdb_train_new_1024.csv",
        "csvs/imdb_test_new_1024.csv",
        "csvs/imdb_valid_new_1024.csv",
    ]:
        anno_file_df = pd.read_csv(os.path.join(root_path, anno_file))
        for _, row in tqdm(anno_file_df.iterrows(), total=len(anno_file_df)):
            file_name = row["filename"]
            if os.path.exists(f"{output_path}/{file_name}"):
                continue

            x_min = row["x_min"]
            y_min = row["y_min"]
            x_max = row["x_max"]
            y_max = row["y_max"]

            img = PIL.Image.open(f"{root_path}/data/imdb-clean-1024/{file_name}")
            # Calculate the width and height of the bounding box
            width = x_max - x_min
            height = y_max - y_min

            # Determine the size of the square
            square_size = max(width, height)

            # Calculate the new bounding box coordinates to make it square
            x_center = x_min + width / 2
            y_center = y_min + height / 2

            new_x_min = int(x_center - square_size / 2)
            new_y_min = int(y_center - square_size / 2)
            new_x_max = int(x_center + square_size / 2)
            new_y_max = int(y_center + square_size / 2)

            # Crop and resize the image
            img = img.crop((new_x_min, new_y_min, new_x_max, new_y_max))
            # img = img.resize((target_size, target_size))

            subfolder = file_name.split("/")[0]
            os.makedirs(f"{output_path}/{subfolder}", exist_ok=True)
            img.save(f"{output_path}/{file_name}")

        # Copy the annotation file
        shutil.copy(
            os.path.join(root_path, anno_file), os.path.join(output_path, anno_file)
        )
        print(f"Processed {anno_file}")

    print("Finished processing all files")


# From https://github.com/tkarras/progressive_growing_of_gans/blob/2504c3f3cb98ca58751610ad61fa1097313152bd/dataset_tool.py#L421
def _crop_like_celebahq(
    img_path: str,
    landmarks: np.ndarray,
):
    """
    Crop and align an image to match the CelebA-HQ dataset format using facial landmarks.

    Parameters
    ----------
    img_path : str
        Path to the input image.
    landmarks : np.ndarray
        Array of facial landmarks with shape (5, 2).

    Returns
    -------
    np.ndarray
        Cropped and aligned image as a NumPy array with shape (3, 1024, 1024).
    """

    def rot90(v):
        return np.array([-v[1], v[0]])

    # Load original image.
    # orig_idx = fields["orig_idx"][idx]
    # orig_file = fields["orig_file"][idx]
    # orig_path = os.path.join(celeba_dir, "img_celeba", orig_file)
    img = PIL.Image.open(img_path).convert("RGB")

    # Choose oriented crop rectangle.
    # Original landmarks dims:
    # landmarks = np.float32(landmarks).reshape(-1, 5, 2)
    # lm = landmarks[orig_idx]
    # lm shape: (5, 2)
    lm = landmarks
    eye_avg = (lm[0] + lm[1]) * 0.5 + 0.5
    mouth_avg = (lm[3] + lm[4]) * 0.5 + 0.5
    eye_to_eye = lm[1] - lm[0]
    eye_to_mouth = mouth_avg - eye_avg
    x = eye_to_eye - rot90(eye_to_mouth)
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = rot90(x)
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    zoom = 1024 / (np.hypot(*x) * 2)

    # Shrink.
    shrink = int(np.floor(0.5 / zoom))
    if shrink > 1:
        size = (
            int(np.round(float(img.size[0]) / shrink)),
            int(np.round(float(img.size[1]) / shrink)),
        )
        img = img.resize(size, PIL.Image.LANCZOS)
        quad /= shrink
        zoom *= shrink

    # Crop.
    border = max(int(np.round(1024 * 0.1 / zoom)), 3)
    crop = (
        int(np.floor(min(quad[:, 0]))),
        int(np.floor(min(quad[:, 1]))),
        int(np.ceil(max(quad[:, 0]))),
        int(np.ceil(max(quad[:, 1]))),
    )
    crop = (
        max(crop[0] - border, 0),
        max(crop[1] - border, 0),
        min(crop[2] + border, img.size[0]),
        min(crop[3] + border, img.size[1]),
    )
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Simulate super-resolution.
    superres = int(np.exp2(np.ceil(np.log2(zoom))))
    if superres > 1:
        img = img.resize(
            (img.size[0] * superres, img.size[1] * superres), PIL.Image.LANCZOS
        )
        quad *= superres
        zoom /= superres

    # Pad.
    pad = (
        int(np.floor(min(quad[:, 0]))),
        int(np.floor(min(quad[:, 1]))),
        int(np.ceil(max(quad[:, 0]))),
        int(np.ceil(max(quad[:, 1]))),
    )
    pad = (
        max(-pad[0] + border, 0),
        max(-pad[1] + border, 0),
        max(pad[2] - img.size[0] + border, 0),
        max(pad[3] - img.size[1] + border, 0),
    )
    if max(pad) > border - 4:
        pad = np.maximum(pad, int(np.round(1024 * 0.3 / zoom)))
        img = np.pad(
            np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), "reflect"
        )
        h, w, _ = img.shape
        y, x, _ = np.mgrid[:h, :w, :1]
        mask = 1.0 - np.minimum(
            np.minimum(np.float32(x) / pad[0], np.float32(y) / pad[1]),
            np.minimum(np.float32(w - 1 - x) / pad[2], np.float32(h - 1 - y) / pad[3]),
        )
        blur = 1024 * 0.02 / zoom
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(
            mask * 3.0 + 1.0, 0.0, 1.0
        )
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.round(img), 0, 255)), "RGB")
        quad += pad[0:2]

    # Transform.
    img = img.transform(
        (4096, 4096), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR
    )
    img = img.resize((1024, 1024), PIL.Image.LANCZOS)
    img = np.asarray(img).transpose(2, 0, 1)

    return img


def _extract_imdbwiki_landmarks(row: Union[pd.DataFrame, pd.Series]):
    landmarks = np.array(
        row[
            [
                "lefteye_x",
                "lefteye_y",
                "righteye_x",
                "righteye_y",
                "nose_x",
                "nose_y",
                "leftmouth_x",
                "leftmouth_y",
                "rightmouth_x",
                "rightmouth_y",
            ]
        ]
    ).reshape((5, 2))
    return landmarks


def _process_image(args: tuple[int, str, pd.Series, str]):
    i, img_path, row, output_path = args
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    landmarks = _extract_imdbwiki_landmarks(row)
    img = _crop_like_celebahq(img_path, landmarks)

    # print(f"{i} Saving", output_path)
    PIL.Image.fromarray(img.transpose(1, 2, 0)).save(output_path)


def create_cropped_images_like_celebahq(root_path: str, output_path: str):
    """Processes the dataset to create the image files cropped by square bounding boxes.

    The cropping method is directly taken from the generation fo CelebAHQ"""

    os.makedirs(output_path, exist_ok=True)

    img_landmark_tuples = []

    i = 0
    skipped = 0
    df_landmarks = pd.read_csv(
        os.path.join(root_path, "csvs/imdb_1024_retinaface_predictions.csv")
    )
    for anno_file in [
        "csvs/imdb_train_new_1024.csv",
        "csvs/imdb_test_new_1024.csv",
        "csvs/imdb_valid_new_1024.csv",
    ]:
        anno_file_df = pd.read_csv(os.path.join(root_path, anno_file)).merge(
            df_landmarks, on="filename", how="left", validate="1:1"
        )

        for _, row in tqdm(
            anno_file_df.iterrows(),
            total=len(anno_file_df),
            desc=f"Gathering {anno_file}",
        ):
            file_name = row["filename"]
            img_path = f"{root_path}/data/imdb-clean-1024/{file_name}"

            subfolder = os.path.basename(os.path.dirname(img_path))
            img_name = os.path.basename(img_path)

            output_folder = os.path.join(output_path, subfolder)
            output_img_path = os.path.join(output_folder, img_name)

            if os.path.exists(output_img_path):
                print(f"File {output_img_path} exists. Skipping.")
                skipped += 1
            else:
                img_landmark_tuples.append((i, img_path, row, output_img_path))
                i += 1

    print(f"Skipped {skipped} files out of {i + skipped} files. Processing: {i}")

    def run_imap_multiprocessing(func, argument_list, num_processes):
        pool = Pool(processes=num_processes)

        result_list_tqdm = []
        for result in tqdm(
            pool.imap(func=func, iterable=argument_list),
            total=len(argument_list),
            desc="Cropping images",
        ):
            result_list_tqdm.append(result)

        return result_list_tqdm

    run_imap_multiprocessing(_process_image, img_landmark_tuples, num_processes=32)

    # for args in tqdm(img_landmark_tuples, desc="Cropping images"):
    #     _process_image(args)

    print("Finished processing all files")

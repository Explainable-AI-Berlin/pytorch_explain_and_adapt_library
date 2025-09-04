import sys

from matplotlib.colors import Normalize

import argparse
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import shutil
from diff_cf_ir.file_utils import deterministic_run


def embed_numberstring(number_str, num_digits=7):
    number_str = str(number_str)
    return "0" * (num_digits - len(number_str)) + number_str


def latent_to_square_image(
    color_a,
    color_b,
    position_x=None,
    position_y=None,
    SIZE_INNER=8,
    SIZE_BORDER=2,
    noise: np.ndarray = None,
):
    SIZE_ADDED = SIZE_INNER + 2 * SIZE_BORDER
    img = np.ones([64, 64, 3], dtype=np.float32) * color_b
    if noise is None:
        noise = np.random.randn(*img.shape) * 20

    if position_x is None:
        position_x = int((64 - SIZE_ADDED) / 2)

    if position_y is None:
        position_y = int((64 - SIZE_ADDED) / 2)

    img_base = np.clip(img + noise, 0, 255)
    img = np.copy(img_base)
    img[
        position_x : position_x + SIZE_ADDED,
        position_y : position_y + SIZE_ADDED,
    ] = np.clip(
        127
        + noise[
            position_x : position_x + SIZE_ADDED,
            position_y : position_y + SIZE_ADDED,
        ],
        0,
        255,
    )
    foreground = np.concatenate(
        [
            color_a * np.ones([SIZE_INNER, SIZE_INNER, 1]),
            np.zeros([SIZE_INNER, SIZE_INNER, 2]),
        ],
        axis=-1,
    )
    img[
        position_x + SIZE_BORDER : position_x + SIZE_ADDED - SIZE_BORDER,
        position_y + SIZE_BORDER : position_y + SIZE_ADDED - SIZE_BORDER,
    ] = np.clip(
        foreground
        + noise[
            position_x + SIZE_BORDER : position_x + SIZE_ADDED - SIZE_BORDER,
            position_y + SIZE_BORDER : position_y + SIZE_ADDED - SIZE_BORDER,
        ],
        0,
        255,
    )
    img = Image.fromarray(img.astype(dtype=np.uint8))
    return img, noise


class SquareDatasetGenerator:
    def __init__(
        self,
        dataset_path: str,
        num_samples: int,
    ):
        self.dataset_path = dataset_path
        self.num_samples = num_samples

    def generate_dataset(self):
        if os.path.exists(self.dataset_path):
            datestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # move self.dataset_dir to self.dataset_dir + "_old_ + {datestamp}
            shutil.move(
                self.dataset_path,
                self.dataset_path + "_old_" + datestamp,
            )
            shutil.move(
                self.dataset_path + "_inverse",
                self.dataset_path + "_old_" + datestamp + "_inverse",
            )

        os.makedirs(self.dataset_path)
        os.makedirs(os.path.join(self.dataset_path, "imgs"))
        os.makedirs(os.path.join(self.dataset_path, "masks"))
        os.makedirs(os.path.join(self.dataset_path + "_inverse", "imgs"))
        os.makedirs(os.path.join(self.dataset_path + "_inverse", "masks"))
        lines_out = [
            "Name,ClassA,ClassB,ClassC,ClassD,ColorA,ColorB,PositionX,PositionY"
        ]
        lines_out_inverse = [
            "Name,ClassA,ClassB,ClassC,ClassD,ColorA,ColorB,PositionX,PositionY"
        ]

        SIZE_INNER = 8
        SIZE_BORDER = 2
        SIZE_ADDED = SIZE_INNER + 2 * SIZE_BORDER
        for sample_idx in tqdm(
            range(self.num_samples), desc="Generating Square dataset"
        ):
            if sample_idx % 2 == 0:
                class_a = 1
                color_a = np.random.randint(128, 256)

            else:
                class_a = 0
                color_a = np.random.randint(0, 128)

            if int(sample_idx / 2) % 2 == 0:
                class_b = 1
                color_b = np.random.randint(128, 256)

            else:
                class_b = 0
                color_b = np.random.randint(0, 128)

            num_positions = 64 - SIZE_ADDED
            if int(sample_idx / 4) % 2 == 0:
                class_c = 1
                position_x = np.random.randint(int(num_positions / 2), num_positions)

            else:
                class_c = 0
                position_x = np.random.randint(0, int(num_positions / 2))

            if int(sample_idx / 8) % 2 == 0:
                class_d = 1
                position_y = np.random.randint(int(num_positions / 2), num_positions)

            else:
                class_d = 0
                position_y = np.random.randint(0, int(num_positions / 2))

            sample_name = embed_numberstring(sample_idx, 8) + ".png"
            img, noise = latent_to_square_image(
                position_x=position_x,
                position_y=position_y,
                color_a=color_a,
                color_b=color_b,
            )
            img.save(os.path.join(self.dataset_path, "imgs", sample_name))
            img_inverse, noise = latent_to_square_image(
                position_x=position_x,
                position_y=position_y,
                color_a=color_a,
                color_b=255 - color_b,
                noise=noise,
            )
            """img_inverse = np.abs(img_base - 255)
            img_inverse[
                position_x : position_x + SIZE_ADDED,
                position_y : position_y + SIZE_ADDED,
            ] = np.clip(
                127
                - noise[
                    position_x : position_x + SIZE_ADDED,
                    position_y : position_y + SIZE_ADDED,
                ],
                0,
                255,
            )
            img_inverse[
                position_x + SIZE_BORDER : position_x + SIZE_ADDED - SIZE_BORDER,
                position_y + SIZE_BORDER : position_y + SIZE_ADDED - SIZE_BORDER,
            ] = np.clip(
                color_a
                - noise[
                    position_x + SIZE_BORDER : position_x + SIZE_ADDED - SIZE_BORDER,
                    position_y + SIZE_BORDER : position_y + SIZE_ADDED - SIZE_BORDER,
                ],
                0,
                255,
            )
            img_inverse = Image.fromarray(img_inverse.astype(dtype=np.uint8))"""
            img_inverse.save(
                os.path.join(self.dataset_path + "_inverse", "imgs", sample_name)
            )
            mask = np.zeros([64, 64, 3], dtype=np.uint8)
            mask[
                position_x + SIZE_BORDER : position_x + SIZE_ADDED - SIZE_BORDER,
                position_y + SIZE_BORDER : position_y + SIZE_ADDED - SIZE_BORDER,
            ] = 255
            img_mask = Image.fromarray(mask)
            img_mask.save(os.path.join(self.dataset_path, "masks", sample_name))
            img_mask.save(
                os.path.join(self.dataset_path + "_inverse", "masks", sample_name)
            )

            attributes = [
                sample_name,
                str(class_a),
                str(class_b),
                str(class_c),
                str(class_d),
                str(float(color_a) / 255),
                str(float(color_b) / 255),
                str(float(position_x) / 64),
                str(float(position_y) / 64),
            ]
            lines_out.append(",".join(attributes))
            attributes_inverse = [
                sample_name,
                str(class_a),
                str(class_b),
                str(class_c),
                str(class_d),
                str(float(color_a) / 255),
                str(float(color_b - 255) / 255),
                str(float(position_x) / 64),
                str(float(position_y) / 64),
            ]
            lines_out_inverse.append(",".join(attributes_inverse))
            if (sample_idx + 1) % 100 == 0:
                open(os.path.join(self.dataset_path, "data.csv"), "w").write(
                    "\n".join(lines_out)
                )
                open(
                    os.path.join(self.dataset_path + "_inverse", "data.csv"),
                    "w",
                ).write("\n".join(lines_out_inverse))

        open(os.path.join(self.dataset_path, "data.csv"), "w").write(
            "\n".join(lines_out)
        )
        open(
            os.path.join(self.dataset_path + "_inverse", "data.csv"),
            "w",
        ).write("\n".join(lines_out_inverse))


class LowerUpperSquareDatasetGenerator:
    SIZE_INNER = 8
    SIZE_BORDER = 2
    SIZE_ADDED = SIZE_INNER + 2 * SIZE_BORDER
    NUM_POSITIONS = 64 - SIZE_ADDED

    def __init__(
        self,
        dataset_path,
    ):
        self.create_output_dirs(dataset_path=dataset_path)
        self.dataset_path = dataset_path

    def create_output_dirs(self, dataset_path):
        squares_lower = os.path.join(dataset_path, "squares_lower")
        squares_upper = os.path.join(dataset_path, "squares_upper")

        assert not os.path.exists(squares_lower), f"{squares_lower} already exists."
        assert not os.path.exists(squares_upper), f"{squares_upper} already exists."

        os.makedirs(os.path.join(squares_lower, "imgs"))
        os.makedirs(os.path.join(squares_lower, "masks"))
        os.makedirs(os.path.join(squares_upper, "imgs"))
        os.makedirs(os.path.join(squares_upper, "masks"))

        self.squares_lower_path = squares_lower
        self.squares_upper_path = squares_upper

    def construct_dataset_latents(self):
        """
        Construct the necessary latents for the experiment.
        Fills only the lower half of the latent space in a 5x7 grid to cover the
        inner square and background colors.
        """
        # 5x7 grid for the inner square
        NUM_GRID_X = 5
        NUM_GRID_Y = 7

        # 3x3 = 9 Total Positions
        MAX_POSITIONS = 3

        position_x_list = np.linspace(
            0, self.NUM_POSITIONS, num=MAX_POSITIONS, dtype=int
        )
        position_y_list = np.linspace(
            0, self.NUM_POSITIONS, num=MAX_POSITIONS, dtype=int
        )
        color_inner_list = np.linspace(0, 127, num=NUM_GRID_X, dtype=int)
        color_bg_list = np.linspace(0, 255, num=NUM_GRID_Y, dtype=int)

        # Generate all possible combinations
        latents = [
            (position_x, position_y, color_inner, color_bg)
            for position_x in position_x_list
            for position_y in position_y_list
            for color_inner in color_inner_list
            for color_bg in color_bg_list
        ]

        def plot_color_combinations(color_inner_list, color_bg_list):
            color_inner_list = color_inner_list / 255
            color_bg_list = color_bg_list / 255

            coordinates = np.array(
                np.meshgrid(color_inner_list, color_bg_list)
            ).T.reshape(-1, 2)

            plt.scatter(
                coordinates[:, 0],
                coordinates[:, 1],
                c=coordinates[:, 0],
                cmap="inferno",
                norm=Normalize(0, 1),
            )
            plt.xlabel("Inner Color (normalized)")
            plt.ylabel("Background Color (normalized)")
            plt.title("Color Combinations")
            plt.colorbar(label="Inner Color (normalized)", orientation="horizontal")
            plt.grid(True)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.savefig(
                os.path.join(self.dataset_path, "color_combinations.png"), dpi=200
            )

        plot_color_combinations(color_inner_list, color_bg_list)
        return latents

    def generate_dataset(self):
        lines_out = [
            "Name,ClassA,ClassB,ClassC,ClassD,ColorA,ColorB,PositionX,PositionY"
        ]
        lines_out_inverse = [
            "Name,ClassA,ClassB,ClassC,ClassD,ColorA,ColorB,PositionX,PositionY"
        ]

        latents = self.construct_dataset_latents()
        for sample_idx, (position_x, position_y, color_inner, color_bg) in tqdm(
            enumerate(latents), desc="Generating Square dataset", total=len(latents)
        ):
            sample_name = f"{sample_idx:08}.png"
            # Create initial latent
            img, noise = latent_to_square_image(
                position_x=position_x,
                position_y=position_y,
                color_a=color_inner,
                color_b=color_bg,
            )
            img.save(os.path.join(self.squares_lower_path, "imgs", sample_name))

            # Create the mirror color latent
            color_inner_upper = color_inner + 128
            img_inverse, noise = latent_to_square_image(
                position_x=position_x,
                position_y=position_y,
                color_a=color_inner_upper,
                color_b=color_bg,
                noise=noise,
            )
            img_inverse.save(os.path.join(self.squares_upper_path, "imgs", sample_name))

            # Masks for the latents
            img_mask = self.construct_mask(position_x, position_y)

            # Verify the mask
            self.verify_mask(img, img_mask, color_inner)
            self.verify_mask(img_inverse, img_mask, color_inner_upper)

            img_mask.save(os.path.join(self.squares_lower_path, "masks", sample_name))
            img_mask.save(os.path.join(self.squares_upper_path, "masks", sample_name))

            # Fill the data.csv file
            attributes = self.get_attributes(
                position_x, position_y, color_inner, color_bg, sample_name
            )
            lines_out.append(",".join(attributes))
            attributes_inverse = self.get_attributes(
                position_x, position_y, color_inner_upper, color_bg, sample_name
            )
            lines_out_inverse.append(",".join(attributes_inverse))

        open(os.path.join(self.squares_lower_path, "data.csv"), "w").write(
            "\n".join(lines_out)
        )
        open(
            os.path.join(self.squares_upper_path, "data.csv"),
            "w",
        ).write("\n".join(lines_out_inverse))

    def get_attributes(
        self, position_x, position_y, color_inner, color_bg, sample_name
    ):
        return [
            sample_name,
            str(-1),  # class_a
            str(-1),  # class_b
            str(-1),  # class_c
            str(-1),  # class_d
            str(float(color_inner) / 255),
            str(float(color_bg) / 255),
            str(float(position_x) / 64),
            str(float(position_y) / 64),
        ]

    def construct_mask(self, position_x, position_y):
        mask = np.zeros([64, 64, 3], dtype=np.uint8)

        x_start = position_x + self.SIZE_BORDER
        x_end = position_x + self.SIZE_ADDED - self.SIZE_BORDER
        y_start = position_y + self.SIZE_BORDER
        y_end = position_y + self.SIZE_ADDED - self.SIZE_BORDER
        mask[x_start:x_end, y_start:y_end] = 255

        img_mask = Image.fromarray(mask)
        return img_mask

    def verify_mask(self, x, mask, color):
        pass
        # x = np.array(x)
        # mask = np.array(mask).astype(bool)
        # mask[:, :, [1, 2]] = False  # Only select red channel

        # color_from_mask = x[mask].mean()
        # if not np.isclose(color, color_from_mask, atol=11):
        #     print(f"Color mismatch: {color} !~ {color_from_mask}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate either the train or val set for the Square datset. "
            " For the validation set, two datasets will be generated: one for the lower end of the"
            " color spectrum and one for the higher end."
        )
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="The output folder where the subsets will be saved.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val"],
        help="The split to generate the dataset for.",
    )
    args = parser.parse_args()

    deterministic_run(0)
    if args.split == "train":
        generator = SquareDatasetGenerator(
            dataset_path=args.output_folder, num_samples=16000
        )
        generator.generate_dataset()
    else:
        generator = LowerUpperSquareDatasetGenerator(dataset_path=args.output_folder)
        generator.generate_dataset()

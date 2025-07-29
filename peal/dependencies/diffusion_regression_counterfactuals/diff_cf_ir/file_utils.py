import json
import os
import threading

from lightning import seed_everything
import torch
import torchvision


def create_result_dir(result_dir: str) -> str:
    if os.path.exists(result_dir):
        rename_if_exists(result_dir)

    os.makedirs(result_dir, exist_ok=True)

    return result_dir


def rename_if_exists(file_path: str) -> None:
    if not os.path.exists(file_path):
        return

    file_path = os.path.abspath(file_path)
    timestamp = int(os.path.getmtime(file_path))

    if os.path.isdir(file_path):
        new_name = f"{file_path}_old_{timestamp}"
        if os.path.exists(new_name):
            new_name += "_1"
        os.rename(file_path, new_name)
    else:
        base_folder, file_name = os.path.split(file_path)
        file_name_noext, file_ext = os.path.splitext(file_name)
        new_name = f"{file_name_noext}_old_{timestamp}{file_ext}"

        os.rename(
            file_path,
            os.path.join(base_folder, new_name),
        )
    print(f"Renamed {file_path} to {new_name}")


def save_image(img_tensor: torch.Tensor, image_path: str):
    if all(
        [
            not image_path.lower().endswith(ext)
            for ext in [".png", ".jpg", ".jpeg", ".bmp", ".gif"]
        ]
    ):
        image_path += ".png"  # Default to png, if no image extension is provided

    base_folder = os.path.dirname(image_path)
    os.makedirs(base_folder, exist_ok=True)
    torchvision.utils.save_image(
        img_tensor,
        image_path,
    )


def save_img_threaded(img_tensor: torch.Tensor, image_path: str) -> None:
    def _save_image():
        base_folder = os.path.dirname(image_path)
        os.makedirs(base_folder, exist_ok=True)
        torchvision.utils.save_image(
            img_tensor,
            image_path,
        )

    save_thread = threading.Thread(target=_save_image)
    save_thread.start()


def assert_paths_exist(paths: list[str]) -> None:
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"File/Folder {p} does not exist")


def dump_args(args, result_dir: str) -> None:
    os.makedirs(result_dir, exist_ok=True)
    args_file: str = os.path.join(result_dir, "args.json")
    with open(args_file, "w") as f:
        json.dump(vars(args), f, indent=2)


def deterministic_run(seed: int):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    seed_everything(seed, workers=True)


def is_image_file(path: str) -> bool:
    return path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))

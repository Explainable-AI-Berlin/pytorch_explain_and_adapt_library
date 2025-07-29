import argparse
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from diff_cf_ir.models import load_resnet
from tqdm import tqdm
from pathlib import Path
import threading


def load_image(image_path: str, size: int):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ]
    )
    return transform(image).unsqueeze(0)


def adversarial_attack(
    image: torch.Tensor,
    model: torch.nn.Module,
    target: float,
    epsilon: float,
    n_max: int,
    threshold: float = 0.01,
):
    image_attack = image.clone()
    image_attack.requires_grad = True
    optimizer = torch.optim.Adam([image_attack], lr=epsilon)

    target_pt = torch.tensor([[target]], device=image_attack.device)

    with tqdm(total=n_max) as pbar:
        for _ in range(n_max):
            output = model(image_attack)
            pbar.set_description(f"End: {output.item()}")

            loss = torch.nn.functional.mse_loss(
                output, target_pt
            ) + torch.nn.functional.mse_loss(image_attack, image)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if abs(output.item() - target) < threshold:
                break

            pbar.update(1)

    return image_attack.detach()


def get_globs(folder: Path):
    return (
        list(folder.glob("*.png"))
        + list(folder.glob("*.jpg"))
        + list(folder.glob("*.jpeg"))
    )


def save_perturbed_image(image_path, perturbed_image):
    def save_image():
        perturbed_image_np = perturbed_image.squeeze().cpu().numpy().transpose(1, 2, 0)
        perturbed_image_pil = Image.fromarray(
            (perturbed_image_np * 255).astype(np.uint8)
        )
        perturbed_image_pil.save(image_path)

    save_thread = threading.Thread(target=save_image)
    save_thread.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Adversarial Attack on Images in a Folder"
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        help="Path to the input folder containing images",
        required=True,
    )
    parser.add_argument(
        "--predictor_type", type=str, help="Type of predictor", required=True
    )
    parser.add_argument(
        "--predictor_path", type=str, help="Path to the predictor model", required=True
    )
    parser.add_argument(
        "--size", type=int, help="Target size of the input images", required=True
    )
    parser.add_argument(
        "--target", type=float, help="Target Value for the attack", required=True
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.001, help="Epsilon value for the attack"
    )
    parser.add_argument(
        "--n_max", type=float, default=100, help="Maximum number of iterations"
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_resnet(args.predictor_type, args.predictor_path).model.to(device)

    input_folder = Path(args.input_folder)
    perturbed_path = input_folder.parent / (input_folder.stem + "_perturbed")
    perturbed_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving perturbed images to {perturbed_path}")

    for image_path in tqdm(get_globs(input_folder)):
        image = load_image(str(image_path), args.size).to(device)
        perturbed_image = adversarial_attack(
            image,
            model,
            target=args.target,
            epsilon=args.epsilon,
            n_max=args.n_max,
        )

        save_perturbed_image(perturbed_path / image_path.name, perturbed_image)

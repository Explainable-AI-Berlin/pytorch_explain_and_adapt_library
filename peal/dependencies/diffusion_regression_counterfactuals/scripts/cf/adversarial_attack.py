import argparse
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from diff_cf_ir.models import load_resnet
from tqdm import tqdm


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
    epsilon: float = 0.01,
    stop_at: float = None,
):
    image_attack = image.clone()
    image_attack.requires_grad = True
    optimizer = torch.optim.Adam([image_attack], lr=epsilon)
    n_max = 20

    threshold = 0.01

    target_pt = torch.tensor([[target]], device=image_attack.device)

    with tqdm(total=n_max) as pbar:
        for _ in range(n_max):
            output = model(image_attack)
            pbar.set_description(f"Output: {output.item()}")

            loss = torch.nn.functional.mse_loss(
                output, target_pt
            ) + torch.nn.functional.mse_loss(image_attack, image)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            stop_stop_at = (
                abs(output.item() - stop_at) < threshold if stop_at else False
            )
            stop_target = abs(output.item() - target) < threshold
            if stop_stop_at:
                print("Stopping at stop_at")
                break
            if stop_target:
                print("Stopping at target")
                break

            pbar.update(1)

    print("Final output:", output.item())
    return image_attack.detach()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adversarial Attack on an Image")
    parser.add_argument("--input_image", type=str, help="Path to the input image")
    parser.add_argument("--predictor_type", type=str, help="Type of predictor")
    parser.add_argument(
        "--predictor_path", type=str, help="Path to the predictor model"
    )
    parser.add_argument("--size", type=int, help="Target size of the input image")
    parser.add_argument(
        "--epsilon", type=float, default=0.001, help="Epsilon value for the attack"
    )
    parser.add_argument("--target", type=float, help="Target Value for the attack")
    parser.add_argument("--stop_at", type=float, help="Stop attack at this value")

    args = parser.parse_args()

    # Check if all args are set
    assert all(
        [args.input_image, args.predictor_type, args.predictor_path, args.size]
    ), "Missing arguments"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image = load_image(args.input_image, args.size).to(device)
    model = load_resnet(args.predictor_type, args.predictor_path).model.to(device)
    perturbed_image = adversarial_attack(
        image, model, target=args.target, epsilon=args.epsilon, stop_at=args.stop_at
    )

    perturbed_image_np = perturbed_image.squeeze().cpu().numpy().transpose(1, 2, 0)
    perturbed_image_pil = Image.fromarray((perturbed_image_np * 255).astype(np.uint8))
    perturbed_image_pil.save("perturbed.png")

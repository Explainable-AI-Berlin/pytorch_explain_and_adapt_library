from compute_FVA import compute_FVA, arguments
from eval_utils.resnet50_facevgg2_FVA import resnet50, load_state_dict

import torch
import numpy as np

from diff_cf_ir.image_folder_dataset import PairedImageFolderDataset
from diff_cf_ir.metrics import AceFVA


def run_ace_version() -> tuple[dict[str, float], np.ndarray]:
    weights_path = "/home/tha/ACE/pretrained/resnet50_ft_weight.pkl"
    exp_name = "celeba-attack/explanation/"
    output_path = "imgs_fva/celeba-attack/"
    batch_size = 16

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    oracle = resnet50(num_classes=8631, include_top=False).to(device)
    load_state_dict(oracle, weights_path)
    oracle.eval()

    results = compute_FVA(oracle, output_path, exp_name, batch_size)

    summary = {
        "FVA": np.mean(results[0]),
        "FVA_STD": np.std(results[0]),
        "mean_dist": np.mean(results[1]),
        "std_dist": np.std(results[1]),
    }
    return summary, np.hstack(results)


def run_thesis_version() -> tuple[dict[str, float], np.ndarray]:
    real_folder = "/home/tha/master-thesis-xai/diff_cf_ir/diff_cf_ir/test/imgs_fva/celeba-attack/Original/Correct"
    fake_folder = "/home/tha/master-thesis-xai/diff_cf_ir/diff_cf_ir/test/imgs_fva/celeba-attack/Results/celeba-attack/explanation/CC/CCF/CF"

    paired_image_dataset = PairedImageFolderDataset(real_folder, fake_folder, size=224)

    loader = torch.utils.data.DataLoader(
        paired_image_dataset, batch_size=16, shuffle=False
    )
    ace_fva = AceFVA("/home/tha/ACE/pretrained/resnet50_ft_weight.pkl")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    scores = []
    for real_imgs, fake_imgs, _ in loader:
        real_imgs_pt = real_imgs.to(device)
        fake_imgs_pt = fake_imgs.to(device)

        score = ace_fva.compute_counterfactual_score(real_imgs_pt, fake_imgs_pt)
        scores.append(score)

    scores = torch.vstack(scores)

    results_dict = {
        "FVA": scores[:, 0].mean().item(),
        "FVA_STD": scores[:, 0].std().item(),
        "mean_dist": scores[:, 1].mean().item(),
        "std_dist": scores[:, 1].std().item(),
    }
    return results_dict, scores.cpu().numpy()


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    ace_dict, ace_results = run_ace_version()
    thesis_dict, thesis_results = run_thesis_version()

    print("Results from ACE:")
    print(ace_dict)

    print("Results from Thesis:")
    print(thesis_dict)

    print("Comparing results:")
    print(ace_results[:, 1])
    print(thesis_results[:, 1])

    # Sort the values first, the order might be different due to the loader
    assert np.allclose(
        ace_results, thesis_results, atol=1e-4
    ), "The numpy arrays ace_results and thesis_results are not equal when rounded to 4 digits"

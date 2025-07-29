import torch
import numpy as np

from compute_MNAC import compute_MNAC, CelebaHQOracle
from diff_cf_ir.image_folder_dataset import PairedImageFolderDataset
from diff_cf_ir.metrics import AceMNAC

WEIGHTS_PATH = (
    "/home/tha/ACE/pretrained/oracle/oracle_attribute/celebamaskhq/checkpoint.tar"
)
EXP_NAME = "celeba-attack/explanation/"
OUTPUT_PATH = "imgs_fva/celeba-attack/"
BATCH_SIZE = 16


def run_ace_version():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    oracle = CelebaHQOracle(weights_path=WEIGHTS_PATH, device=device)

    mnacs, dists_real, dists_cf = compute_MNAC(
        oracle, OUTPUT_PATH, EXP_NAME, BATCH_SIZE
    )

    return mnacs, dists_real, dists_cf


def run_thesis_version():
    real_folder = "/home/tha/master-thesis-xai/diff_cf_ir/diff_cf_ir/test/imgs_fva/celeba-attack/Original/Correct"
    fake_folder = "/home/tha/master-thesis-xai/diff_cf_ir/diff_cf_ir/test/imgs_fva/celeba-attack/Results/celeba-attack/explanation/CC/CCF/CF"

    paired_image_dataset = PairedImageFolderDataset(real_folder, fake_folder, size=224)

    loader = torch.utils.data.DataLoader(
        paired_image_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    ace_mnac = AceMNAC(WEIGHTS_PATH)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    mnacs_results = []
    outs_real_results = []
    outs_cf_results = []

    for real_imgs, fake_imgs, _ in loader:
        real_imgs_pt = real_imgs.to(device)
        fake_imgs_pt = fake_imgs.to(device)

        mnacs, outs_real, outs_cf = ace_mnac.compute_MNAC(real_imgs_pt, fake_imgs_pt)
        mnacs_results.append(mnacs)
        outs_real_results.append(outs_real)
        outs_cf_results.append(outs_cf)

    return (
        torch.cat(mnacs_results),
        torch.cat(outs_real_results),
        torch.cat(outs_cf_results),
    )


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    mnacs_results, outs_real_results, outs_cf_results = run_ace_version()
    mnacs_results_thesis, outs_real_results_thesis, outs_cf_results_thesis = (
        run_thesis_version()
    )

    # Sort the results by the mean of outs_real_results
    sorted_indices = np.argsort(outs_real_results.mean(1))
    mnacs_results = torch.from_numpy(mnacs_results[sorted_indices])
    outs_real_results = torch.from_numpy(outs_real_results[sorted_indices])
    outs_cf_results = torch.from_numpy(outs_cf_results[sorted_indices])

    sorted_indices = torch.argsort(outs_real_results_thesis.mean(1))
    mnacs_results_thesis = mnacs_results_thesis[sorted_indices]
    outs_real_results_thesis = outs_real_results_thesis[sorted_indices]
    outs_cf_results_thesis = outs_cf_results_thesis[sorted_indices]

    print("Results")
    print("MNACs ACE:", mnacs_results)
    print("MNACs thesis:", mnacs_results)

    max_dist_real = torch.max(torch.abs(outs_real_results - outs_real_results_thesis))
    max_dist_cf = torch.max(torch.abs(outs_cf_results - outs_cf_results_thesis))

    print("Max Distance Real:", max_dist_real.item())
    print("Max Distance CF:", max_dist_cf.item())

    print("Comparing results:")
    assert torch.allclose(
        mnacs_results, mnacs_results_thesis, atol=1e-4
    ), "MNACs results are not equal"
    assert torch.allclose(
        outs_real_results, outs_real_results_thesis, atol=1e-4
    ), "Dists Real results are not equal"
    assert torch.allclose(
        outs_cf_results, outs_cf_results_thesis, atol=1e-4
    ), "Dists CF results are not equal"

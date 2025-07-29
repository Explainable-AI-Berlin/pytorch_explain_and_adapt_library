import csv
import torch

from diff_cf_ir.metrics import AceFlipRateYoungOld
from diff_cf_ir.image_folder_dataset import ImageFolderDataset


@torch.no_grad()
def predict_images(model: torch.nn.Module, dataloader):

    results = []
    for batch in dataloader:
        filenames, imgs = batch
        predictions = model(imgs.to("cuda")).tolist()
        results.extend(zip(filenames, predictions))
    return results


def save_results_to_csv(results, output_file):
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["img", "pred"])
        writer.writerows(results)


def main():
    image_folder = "/home/tha/datasets/celebahq_samples"
    output_csv = "test_ace_predictions.csv"

    classifier_path = (
        "/home/tha/ACE/pretrained/decision_densenet/celebamaskhq/checkpoint.tar"
    )

    model = AceFlipRateYoungOld(classifier_path=classifier_path).model

    dataset = ImageFolderDataset(image_folder, 256)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, num_workers=4)
    results = predict_images(model, dataloader)
    save_results_to_csv(results, output_csv)
    print(f"Predictions saved to {output_csv}")


if __name__ == "__main__":
    main()

from argparse import Namespace
from typing import Union
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import PeakSignalNoiseRatio
from tqdm import tqdm
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from diff_cf_ir.image_folder_dataset import PairedImageFolderDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------DISTRIBUTION METRICS---------------
class FIDScorer:
    def __init__(
        self,
        paired_dataset: PairedImageFolderDataset,
        feature_dim=2048,
        batch_size=32,
    ):
        super().__init__()
        print("Constructing FID metric")
        self.fid = FrechetInceptionDistance(feature=feature_dim).to(device)

        dataloader = DataLoader(paired_dataset, batch_size=batch_size, num_workers=4)
        for real_imgs, _, _ in tqdm(
            dataloader,
            desc=f"FID: Real images",
            total=len(dataloader),
        ):
            imgs = self.scale_to_255(real_imgs).to(device)
            self.fid.update(imgs, real=True)

        self.fid_real_only = self.fid.clone()

    def update_fake_images(self, fake_images: torch.Tensor):
        fake_images = self.scale_to_255(fake_images.to(self.fid.device))
        self.fid.update(fake_images, real=False)

    def compute_score(self) -> torch.Tensor:
        return self.fid.compute()

    def scale_to_255(self, fake_images: torch.Tensor) -> torch.Tensor:
        if fake_images.max() <= 1.0:
            fake_images = (fake_images * 255).to(torch.uint8)
        return fake_images

    def reset(self):
        self.fid = self.fid_real_only.clone()


# ----------------REFERENCE METRICS---------------
class ReferenceScorer:
    def compute_counterfactual_score(
        self, real_images: torch.Tensor, counterfactuals: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError


class PeakSNR(ReferenceScorer):
    def __init__(
        self, data_range: tuple[float, float] = (0.0, 1.0), reduction=None
    ) -> None:
        self.psnr = PeakSignalNoiseRatio(
            data_range=data_range, reduction=reduction, dim=(1, 2, 3)
        ).to(device)

    def compute_counterfactual_score(
        self, real_images: torch.Tensor, counterfactuals: torch.Tensor
    ) -> torch.Tensor:
        return self.psnr(
            counterfactuals, real_images.to(counterfactuals.device)
        ).unsqueeze(0)


class LPIPS(ReferenceScorer):
    def __init__(self, normalize=True) -> None:
        """
        Initializes LPIPS.

        Args:
            normalize (bool): Whether to normalize the input images (If not in range [-1, 1]). Default is True.
        """
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=normalize).to(
            device
        )

    def compute_counterfactual_score(
        self, real_images: torch.Tensor, counterfactuals: torch.Tensor
    ) -> torch.Tensor:
        results = []
        for real_img, counterfactual_img in zip(real_images, counterfactuals):
            score = self.lpips(
                real_img.unsqueeze(0), counterfactual_img.unsqueeze(0)
            ).unsqueeze(0)
            results.append(score)
        return torch.cat(results).cpu()


class SSIM(ReferenceScorer):
    def __init__(
        self,
        data_range: Union[float, tuple[float, float]] = (0.0, 1.0),
        reduction=None,
    ) -> None:
        self.ssim = StructuralSimilarityIndexMeasure(
            data_range=data_range, reduction=reduction
        ).to(device)

    def compute_counterfactual_score(
        self, real_images: torch.Tensor, counterfactuals: torch.Tensor
    ) -> torch.Tensor:
        self.ssim = self.ssim.to(real_images.device)
        return self.ssim(counterfactuals, real_images.to(counterfactuals.device)).cpu()


# ----------------Metrics From ACE---------------
# Copied and adapted from https://github.com/guillaumejs2403/ACE


class AceFlipRateYoungOld(ReferenceScorer):
    def __init__(
        self,
        classifier_path: str,
    ) -> None:
        from models import get_classifier as get_ace_classifier

        label_query = 39  # YoungOld
        dataset = "CelebAHQ"  # 256x256
        args = Namespace(
            classifier_path=classifier_path,
            dataset=dataset,
            label_query=label_query,
        )
        # Query should be in [20, 31, 39]
        self.model = get_ace_classifier(args).eval().to(device)

    @torch.inference_mode()
    def compute_counterfactual_score(
        self, real_images: torch.Tensor, counterfactuals: torch.Tensor
    ) -> torch.Tensor:
        real_preds = self.model(real_images) > 0
        fake_preds = self.model(counterfactuals) > 0
        flipped = (real_preds != fake_preds).float()
        return flipped.cpu()


class AceFVA(ReferenceScorer):
    mean_bgr = torch.Tensor([91.4953, 103.8827, 131.0912]).view(1, 3, 1, 1)
    resize = transforms.Resize((224, 224), antialias=True)  # Model expects 224x224

    def transform(self, img: torch.Tensor):
        """Transforms the image to the format expected by the FVA model.

        Parameters
        ----------
        img : torch.Tensor
            Input image in the format B x RGB x H x W.

        Returns
        -------
        torch.Tensor
            Output image in B x BGR x H x W format.
        """
        # RGB -> BGR
        img = self.resize(img)
        img = img[:, [2, 1, 0], :, :] * 255  # model operates on [0, 255]
        img -= self.mean_bgr.to(img)
        return img

    def __init__(
        self,
        classifier_path: str,
    ) -> None:
        """Initializes the FVA scorer with the given classifier.

        The method compute_counterfactual_score will return a tensor with two values:

        - the first value is the FVA score (cosine similarity > 0.5)
        - the second value is the FS score, the cosine similarity between the features of the real and counterfactual images.

        Parameters
        ----------
        classifier_path : str
            The path to the classifier checkpoint
        """
        from eval_utils.resnet50_facevgg2_FVA import resnet50, load_state_dict

        oracle = resnet50(num_classes=8631, include_top=False).to(device)
        load_state_dict(oracle, classifier_path)
        oracle.eval()

        self.model = oracle
        self.cosine_similarity = torch.nn.CosineSimilarity()

    @torch.inference_mode()
    def compute_counterfactual_score(
        self, real_images: torch.Tensor, counterfactuals: torch.Tensor
    ) -> torch.Tensor:

        cl = self.transform(real_images.to(device))
        cf = self.transform(counterfactuals.to(device))
        cl_feat = self.model(cl)
        cf_feat = self.model(cf)
        cosine_distance = self.cosine_similarity(cl_feat, cf_feat).view(-1, 1)
        fva = (cosine_distance > 0.5).float()

        return torch.hstack([fva, cosine_distance]).cpu()


class AceMNAC(ReferenceScorer):
    transformations = transforms.Compose(
        [transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )

    def transform(self, img: torch.Tensor):
        return self.transformations(img)

    def __init__(
        self,
        classifier_path: str,
    ) -> None:
        from compute_MNAC import CelebaHQOracle

        self.oracle = CelebaHQOracle(weights_path=classifier_path, device=device)

    def compute_MNAC(self, real_images: torch.Tensor, counterfactuals: torch.Tensor):
        real = self.transform(real_images.to(device, dtype=torch.float))
        cf = self.transform(counterfactuals.to(device, dtype=torch.float))
        out_real = self.oracle(real).cpu()
        out_cf = self.oracle(cf).cpu()

        mnac = ((out_real > 0.5) != (out_cf > 0.5)).sum(dim=1).float().cpu()

        # return np.concatenate(MNACS), np.concatenate([d[0] for d in dists]), np.concatenate([d[1] for d in dists])
        return mnac, out_real, out_cf

    @torch.inference_mode()
    def compute_counterfactual_score(
        self, real_images: torch.Tensor, counterfactuals: torch.Tensor
    ) -> torch.Tensor:
        mnacs, _, _ = self.compute_MNAC(real_images, counterfactuals)
        return mnacs


# ----------------REGRESSION METRICS---------------
def get_regr_confidence(y, y_hat):
    return torch.nn.functional.l1_loss(y, y_hat, reduction="none")

import torch

from torchvision import transforms

from peal.generators.interfaces import Generator


class PealDataset(torch.utils.data.Dataset):
    """
    This is the base class for all datasets in PEAL. It is a wrapper around

    Args:
        torch.utils.data.Dataset (nn.Module): The parent class for all datasets in PEAL
    """

    def generate_contrastive_collage(
        self,
        x_list: list,
        x_counterfactual_list: list,
        y_target_list: list,
        y_source_list: list,
        y_list: list,
        y_target_start_confidence_list: list,
        y_target_end_confidence_list: list,
        base_path: str,
        start_idx: int,
        y_counterfactual_teacher_list=None,
        y_original_teacher_list=None,
        feedback_list=None,
        **kwargs: dict,
    ):
        """
        This function generates a collage of the input and the counterfactual

        Args:
            batch_in (torch.tensor): The input batch
            counterfactual (torch.tensor): The counterfactual batch

        Returns:
            torch.tensor: The collage
        """
        return torch.zeros([3, 64, 64])

    def serialize_dataset(self, output_dir, x_list, y_list, sample_names=None):
        """
        This function serializes the dataset to a given directory

        Args:
            output_dir (Path): The output directory
            x_list (list): The list of inputs
            y_list (list): The list of labels
            sample_names (list, optional): The list of sample names. Defaults to None.
        """
        pass

    def project_to_pytorch_default(self, x):
        """
        This function maps processed data sample back to pytorch default format

        Args:
            x (torch.tensor): The data sample in the processed format

        Returns:
            torch.tensor: The data sample in the pytorch default format
        """
        if hasattr(self, "normalization"):
            x = self.normalization.invert(x)

        return x

    def project_from_pytorch_default(self, x):
        """
        This function maps pytorch default image to the processed format

        Args:
            x (torch.tensor): The data sample in the pytorch default format

        Returns:
            torch.tensor: The data sample in the processed format
        """
        if hasattr(self, "normalization"):
            x = self.normalization(x)

        if list(x.shape[-3:]) != self.config.input_size:
            x = transforms.Resize(self.config.input_size[1:])(x)

        return x

    def track_generator_performance(self, generator: Generator, batch_size=1):
        """
        This function tracks the performance of the generator

        Args:
            generator (Generator): The generator
        """
        return {}

    def distribution_distance(self, x_list):
        pass

    def pair_wise_distance(self, x1, x2):
        pass

    def variance(self, x_list):
        pass

    def flip_rate(self, y_list, y_counterfactual_list):
        pass

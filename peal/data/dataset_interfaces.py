import torch

from peal.generators.interfaces import Generator


class PealDataset(torch.utils.data.Dataset):
    """
    This is the base class for all datasets in PEAL. It is a wrapper around

    Args:
        torch.utils.data.Dataset (nn.Module): The parent class for all datasets in PEAL
    """

    def generate_contrastive_collage(
        self,
        x,
        x_counterfactual,
        target_confidence_goal,
        y_target,
        y_source,
        base_path,
        start_idx,
        classifier=None,
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
        return x

    def project_from_pytorch_default(self, x):
        """
        This function maps pytorch default image to the processed format

        Args:
            x (torch.tensor): The data sample in the pytorch default format

        Returns:
            torch.tensor: The data sample in the processed format
        """
        return x

    def track_generator_performance(self, generator: Generator, batch_size=1):
        """
        This function tracks the performance of the generator

        Args:
            generator (Generator): The generator
        """
        return {}

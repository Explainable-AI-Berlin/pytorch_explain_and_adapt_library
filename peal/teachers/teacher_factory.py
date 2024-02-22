import os
import torch

from torch import nn
from typing import Union

from peal.teachers.teacher_interface import TeacherInterface
from peal.teachers.model2model_teacher import Model2ModelTeacher
from peal.teachers.human2model_teacher import Human2ModelTeacher
from peal.teachers.segmentation_mask_teacher import SegmentationMaskTeacher
from peal.data.dataset_interfaces import PealDataset


def get_teacher(
    teacher: Union[TeacherInterface, nn.Module, str],
    output_size: int,
    adaptor_config: Union[dict, str],
    dataset: PealDataset,
    device: Union[torch.device, str] = torch.device("cpu"),
    tracking_level: int = 0,
) -> TeacherInterface:
    """
    This function returns a teacher. It can be a teacher that is already.

    Args:
        teacher (Union[TeacherInterface, nn.Module, str]): This can be a teacher that is already
        output_size (int): The output size.
        adaptor_config (Union[dict, str]): The adaptor config.
        dataset (PealDataset): The dataset.
        device (Union[torch.device, str], optional): The device. Defaults to torch.device("cpu").

    Returns:
        TeacherInterface: The teacher.
    """
    if isinstance(teacher, TeacherInterface):
        teacher = teacher

    elif isinstance(teacher, nn.Module):
        teacher.eval()
        teacher = Model2ModelTeacher(teacher, dataset, tracking_level=tracking_level)

    elif teacher[:5] == "human":
        if len(teacher) == 10:
            port = int(teacher[-4:])

        else:
            port = 8000

        teacher = Human2ModelTeacher(port)

    elif teacher == "SegmentationMask":
        teacher = SegmentationMaskTeacher(adaptor_config.attribution_threshold, dataset)

    elif teacher[-4:] == ".cpl":
        teacher = Model2ModelTeacher(
            torch.load(teacher, map_location=device),
            dataset,
            tracking_level=tracking_level,
        )

    else:
        raise ValueError(f"Unknown teacher {teacher}")

    return teacher

import os
import torch

from torch import nn
from typing import Union

from peal.teachers.teacher_interface import TeacherInterface
from peal.teachers.model2model_teacher import Model2ModelTeacher
from peal.teachers.human2model_teacher import Human2ModelTeacher
from peal.teachers.segmentation_mask_teacher import SegmentationMaskTeacher
from peal.teachers.virelay_teacher import VirelayTeacher


def get_teacher(
    teacher: Union[TeacherInterface, nn.Module, str],
    output_size: int,
    adaptor_config: Union[dict, str],
) -> TeacherInterface:
    """
    This function returns a teacher. It can be a teacher that is already.

    Args:
        teacher (Union[TeacherInterface, nn.Module, str]): This can be a teacher that is already
        output_size (int): The output size.
        adaptor_config (Union[dict, str]): The adaptor config.

    Returns:
        TeacherInterface: The teacher.
    """
    if isinstance(teacher, TeacherInterface):
        teacher = teacher

    elif isinstance(teacher, nn.Module):
        teacher.eval()
        teacher = Model2ModelTeacher(teacher)

    elif teacher[:5] == "human":
        if len(teacher) == 10:
            port = int(teacher[-4:])

        else:
            port = 8000

        teacher = Human2ModelTeacher(port)

    elif teacher == "SegmentationMask":
        teacher = SegmentationMaskTeacher(adaptor_config.attribution_threshold)

    elif teacher[:7] == "virelay":
        teacher = VirelayTeacher(num_classes=output_size, port=int(teacher[-4:]))

    else:
        teacher = Model2ModelTeacher(torch.load(os.path.join(teacher, "model.cpl")))

    return teacher

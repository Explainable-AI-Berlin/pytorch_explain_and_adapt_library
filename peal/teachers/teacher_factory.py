import os
import torch

from torch import nn

from peal.teachers.teacher_interface import TeacherInterface
from peal.teachers.model2model_teacher import Model2ModelTeacher
from peal.teachers.human2model_teacher import Human2ModelTeacher
from peal.teachers.segmentation_mask_teacher import SegmentationMaskTeacher
from peal.teachers.virelay_teacher import VirelayTeacher


def get_teacher(teacher, output_size, adaptor_config):
    '''
    Creates a teacher object from a string.

    Args:
        teacher (_type_): _description_
        output_size (_type_): _description_
        adaptor_config (_type_): _description_

    Returns:
        _type_: _description_
    '''
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
        teacher = SegmentationMaskTeacher(
            adaptor_config["attribution_threshold"]
        )

    elif teacher[:7] == "virelay":
        teacher = VirelayTeacher(
            num_classes=output_size, port=int(teacher[-4:])
        )

    else:
        teacher = Model2ModelTeacher(
            torch.load(os.path.join(teacher, "model.cpl"))
        )

    return teacher

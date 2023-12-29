import torch

from peal.teachers.teacher_interface import TeacherInterface


class SegmentationMaskTeacher(TeacherInterface):
    def __init__(self, attribution_threshold):
        self.attribution_threshold = attribution_threshold

    def get_feedback(self, x_attribution_list, hint_list, **args):
        feedback = []
        for idx, heatmap in enumerate(x_attribution_list):
            #
            attribution_relative = (heatmap * hint_list[idx]).sum() / heatmap.sum()
            masked_pixels_relative = hint_list[idx].sum() / torch.prod(
                torch.tensor(list(hint_list[idx].shape))
            )
            attribution_relative = attribution_relative / masked_pixels_relative
            #
            if attribution_relative > self.attribution_threshold:
                feedback.append("true")

            else:
                feedback.append("false")

        return feedback

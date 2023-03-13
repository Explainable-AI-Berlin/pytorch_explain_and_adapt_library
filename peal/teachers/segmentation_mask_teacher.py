import torch

from peal.teachers.teacher_interface import TeacherInterface


class SegmentationMaskTeacher(TeacherInterface):
    def __init__(self, attribution_threshold):
        self.attribution_threshold = attribution_threshold

    def get_feedback(self, heatmaps, hints, **args):
        feedback = []
        for idx, heatmap in enumerate(heatmaps):
            #
            attribution_relative = (heatmap * hints[idx]).sum() / heatmap.sum()
            masked_pixels_relative = hints[idx].sum(
            ) / torch.prod(torch.tensor(list(hints[idx].shape)))
            attribution_relative = attribution_relative / masked_pixels_relative
            #
            if attribution_relative > self.attribution_threshold:
                feedback.append('true')

            else:
                feedback.append('false')

        return feedback

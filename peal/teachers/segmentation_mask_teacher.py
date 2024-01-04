import torch

from peal.teachers.teacher_interface import TeacherInterface


class SegmentationMaskTeacher(TeacherInterface):
    def __init__(self, attribution_threshold, dataset):
        self.attribution_threshold = attribution_threshold
        self.dataset = dataset

    def get_feedback(
        self,
        x_attribution_list,
        hint_list,
        base_dir,
        x_counterfactual_list,
        y_source_list,
        y_target_list,
        x_list,
        y_list,
        **kwargs
    ):
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

        if "validation" in base_dir:
            self.dataset.generate_contrastive_collage(
                y_counterfactual_teacher_list=y_list,
                y_original_teacher_list=list(
                    map(
                        lambda x: x[0] if x[1] == "true" else abs(1 - x[0]),
                        zip(y_list, feedback),
                    )
                ),
                feedback_list=feedback,
                x_counterfactual_list=x_counterfactual_list,
                y_source_list=y_source_list,
                y_target_list=y_target_list,
                x_list=x_list,
                y_list=y_list,
                base_path=base_dir,
                **kwargs,
            )

        return feedback

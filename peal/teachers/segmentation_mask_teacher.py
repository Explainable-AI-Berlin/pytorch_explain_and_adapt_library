import torch

from peal.teachers.interfaces import TeacherInterface


class SegmentationMaskTeacher(TeacherInterface):
    def __init__(self, attribution_threshold, dataset, counterfactual_type = '1sided', tracking_level=0):
        self.attribution_threshold = attribution_threshold
        self.dataset = dataset
        self.counterfactual_type = counterfactual_type
        self.tracking_level = tracking_level

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
        y_target_end_confidence_list,
        student=None,
        **kwargs
    ):
        feedback = []
        device = "cuda" if next(student.parameters()).is_cuda else "cpu"
        for idx, heatmap in enumerate(x_attribution_list):
            #
            """
            attribution_relative = (heatmap * hint_list[idx]).sum() / heatmap.sum()
            masked_pixels_relative = hint_list[idx].sum() / torch.prod(
                torch.tensor(list(hint_list[idx].shape))
            )
            attribution_relative = attribution_relative / masked_pixels_relative
            if attribution_relative > self.attribution_threshold:
                feedback.append("true")

            else:
                feedback.append("false")
            """
            #
            y_target_confidence_end = torch.nn.functional.softmax(
                student(x_counterfactual_list[idx].unsqueeze(0).to(device))[0]
            )[y_target_list[idx]]
            if (
                not abs(y_target_confidence_end - y_target_end_confidence_list[idx])
                < 0.01
            ):
                feedback.append("confidence missmatch!")

            elif (
                self.counterfactual_type == "1sided"
                and y_list[idx] != y_source_list[idx]
            ):
                feedback.append("student originally wrong!")

            elif y_target_end_confidence_list[idx] < 0.5:
                feedback.append("student not swapped!")

            else:
                hints = hint_list[idx].float()
                hints = hints - hints.mean()
                joint_map = heatmap * hints
                true_counterfactual_score = joint_map.sum()
                if true_counterfactual_score > 0:
                    feedback.append("true")

                else:
                    feedback.append("false")

        if self.tracking_level > 1:
            self.dataset.generate_contrastive_collage(
                y_counterfactual_teacher_list=y_list,
                y_target_end_confidence_list=y_target_end_confidence_list,
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

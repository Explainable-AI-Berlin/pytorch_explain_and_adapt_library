import numpy as np
import torch

from peal.teachers.interfaces import TeacherInterface


class BaselineTeacher(TeacherInterface):
    def __init__(
        self,
        strategy="random",
        dataset=None,
        tracking_level=0,
        counterfactual_type="1sided",
    ):
        self.strategy = strategy
        self.dataset = dataset
        self.tracking_level = tracking_level
        self.counterfactual_type = counterfactual_type

    def get_feedback(
        self,
        x_counterfactual_list,
        y_source_list,
        x_list,
        y_list,
        y_target_end_confidence_list,
        base_dir=None,
        y_target_list=None,
        student=None,
        **kwargs
    ):
        feedback = []
        teacher_original = []
        teacher_counterfactual = []
        device = "cuda" if next(student.parameters()).is_cuda else "cpu"
        for idx, counterfactual in enumerate(x_counterfactual_list):
            """y_target_confidence_end = torch.nn.functional.softmax(
                student(counterfactual.unsqueeze(0).to(device))[0]
            )[y_target_list[idx]]
            if (
                not abs(y_target_confidence_end - y_target_end_confidence_list[idx])
                < 0.01
            ):
                print("End confidences are not matching!")
                feedback.append("confidence missmatch!")"""

            if (
                self.counterfactual_type == "1sided"
                and y_list[idx] != y_source_list[idx]
            ):
                feedback.append("student originally wrong!")

            elif y_target_end_confidence_list[idx] < 0.5:
                feedback.append("student not swapped!")

            else:
                if self.strategy == "random":
                    f = np.random.randint(0, 2)
                    feedback.append("true" if f else "false")

                elif self.strategy == "false":
                    feedback.append("false")

                elif self.strategy == "true":
                    feedback.append("true")

            teacher_original.append(-1)
            teacher_counterfactual.append(-1)

        if self.tracking_level >= 5:
            self.dataset.generate_contrastive_collage(
                y_counterfactual_teacher_list=teacher_counterfactual,
                y_original_teacher_list=teacher_original,
                feedback_list=feedback,
                x_counterfactual_list=x_counterfactual_list,
                y_source_list=y_source_list,
                y_target_list=y_target_list,
                x_list=x_list,
                y_list=y_list,
                y_target_end_confidence_list=y_target_end_confidence_list,
                base_path=base_dir,
                **kwargs,
            )

        return feedback

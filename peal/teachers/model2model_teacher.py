import torch

from peal.teachers.interfaces import TeacherInterface


class Model2ModelTeacher(TeacherInterface):
    def __init__(self, model, dataset, tracking_level=0, counterfactual_type="1sided"):
        self.model = model
        self.dataset = dataset
        self.tracking_level = tracking_level
        self.device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
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
        mode="train",
        **kwargs
    ):
        feedback = []
        teacher_original = []
        teacher_counterfactual = []
        is_train = self.model.training
        self.model.eval()
        for idx, counterfactual in enumerate(x_counterfactual_list):
            pred_original = (
                self.model(x_list[idx].unsqueeze(0).to(self.device))
                .squeeze(0)
                .detach()
                .cpu()
                .argmax(-1)
            )
            pred_counterfactual = (
                self.model(counterfactual.unsqueeze(0).to(self.device))
                .squeeze(0)
                .detach()
                .cpu()
                .argmax(-1)
            )

            # TODO here has to be somehing added for OOD e.g. with FID score
            # TODO this will be a problem for multiclass
            # TODO is this a numerical problem???
            if (
                self.counterfactual_type == "1sided"
                and y_list[idx] != y_source_list[idx]
            ):
                feedback.append("student originally wrong!")

            elif pred_original != y_list[idx]:
                feedback.append("teacher originally wrong!")

            elif y_target_end_confidence_list[idx] < 0.5:
                feedback.append("student not swapped!")

            else:
                if pred_original != pred_counterfactual:
                    feedback.append("true")

                else:
                    y_counterfactual = y_source_list[idx]
                    prediction = (
                        student(counterfactual.unsqueeze(0).to(self.device))
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .argmax(-1)
                    )
                    print([int(prediction), int(y_counterfactual)])
                    feedback.append("false")

            teacher_original.append(pred_original)
            teacher_counterfactual.append(pred_counterfactual)

        if self.tracking_level >= 2 and mode == "validation" or self.tracking_level >= 4 and mode == "train":
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

        if is_train:
            self.model.train()

        return feedback

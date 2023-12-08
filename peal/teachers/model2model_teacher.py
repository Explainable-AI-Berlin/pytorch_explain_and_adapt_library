from peal.teachers.teacher_interface import TeacherInterface


class Model2ModelTeacher(TeacherInterface):
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"

    def get_feedback(self, x_counterfactual_list, y_source_list, y_target_list, x_list, base_dir, **kwargs):
        feedback = []
        teacher_original = []
        teacher_counterfactual = []
        is_train = self.model.training
        self.model.eval()
        for idx, counterfactual in enumerate(x_counterfactual_list):
            pred_original = self.model(counterfactual.unsqueeze(0).to(self.device)).squeeze(0).detach().cpu().argmax(-1)
            pred_counterfactual = self.model(counterfactual.unsqueeze(0).to(self.device)).squeeze(0).detach().cpu().argmax(-1)
            pred = self.model(counterfactual.unsqueeze(0).to(self.device)).squeeze(0)

            # TODO here has to be somehing added for OOD e.g. with FID score
            if pred[y_target_list[idx]] > pred[y_source_list[idx]]:
                feedback.append("true")

            else:
                feedback.append("false")
            # TODO here has to be somehing added for OOD e.g. with FID score
            """if pred_original == y_source_list[idx]:
                if pred_counterfactual == y_target_list[idx]:
                    feedback.append("true")

                else:
                    feedback.append("false")

            else:
                if pred_counterfactual == y_target_list[idx]:
                    feedback.append("false")

                else:
                    feedback.append("true")"""

            teacher_original.append(pred_original)
            teacher_counterfactual.append(pred_counterfactual)

        #import pdb; pdb.set_trace()
        self.dataset.generate_contrastive_collage(
            y_counterfactual_teacher_list=teacher_counterfactual,
            y_original_teacher_list=teacher_original,
            feedback_list=feedback,
            x_counterfactual_list=x_counterfactual_list,
            y_source_list=y_source_list,
            y_target_list=y_target_list,
            x_list=x_list,
            base_path=base_dir,
            **kwargs,
        )

        if is_train:
            self.model.train()

        return feedback

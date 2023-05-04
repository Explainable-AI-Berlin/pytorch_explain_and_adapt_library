from peal.teachers.teacher_interface import TeacherInterface


class Model2ModelTeacher(TeacherInterface):
    def __init__(self, model):
        self.model = model
        self.device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"

    def get_feedback(self, x_counterfactual_list, y_source_list, y_target_list, **args):
        feedback = []
        is_train = self.model.training
        self.model.eval()
        for idx, counterfactual in enumerate(x_counterfactual_list):
            pred = self.model(counterfactual.unsqueeze(0).to(self.device)).squeeze(0)

            # TODO here has to be somehing added for OOD e.g. with FID score
            if pred[y_target_list[idx]] > pred[y_source_list[idx]]:
                feedback.append("true")

            else:
                feedback.append("false")

        if is_train:
            self.model.train()

        return feedback

from peal.teachers.teacher_interface import TeacherInterface

class Model2ModelTeacher(TeacherInterface):
    def __init__(self, model):
        self.model = model
        self.device = 'cuda' if next(
            self.model.parameters()).is_cuda else 'cpu'

    def get_feedback(self, counterfactuals, source_classes, target_classes, **args):
        feedback = []
        is_train = self.model.training
        self.model.eval()
        for idx, counterfactual in enumerate(counterfactuals):
            pred = self.model(counterfactual.unsqueeze(
                0).to(self.device)).squeeze(0)

            # TODO here has to be somehing added for OOD e.g. with FID score
            if pred[target_classes[idx]] > pred[source_classes[idx]]:
                feedback.append('true')

            else:
                feedback.append('false')

        if is_train:
            self.model.train()

        return feedback

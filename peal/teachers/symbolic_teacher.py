import torch
from peal.teachers.interfaces import TeacherInterface

class SymbolicTeacher(TeacherInterface):
    """
    Teacher that evaluates counterfactuals based on flipping a specific symbolic confounder attribute.
    If the counterfactual was generated only by modifying the confounder, it flags it as 'false'.
    """
    def __init__(self, model, dataset, confounder_name: str, tracking_level=0, counterfactual_type="1sided"):
        self.model = model
        self.dataset = dataset
        self.confounder_name = confounder_name
        self.tracking_level = tracking_level
        self.counterfactual_type = counterfactual_type
        
        self.device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
        
        # Identify the index of the confounder
        features = self.dataset.attributes
        if self.confounder_name in features:
            self.confounder_idx = features.index(self.confounder_name)
        else:
            raise ValueError(f"Confounder '{self.confounder_name}' not found in dataset attributes: {features}")

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
        is_train = self.model.training
        self.model.eval()

        for idx, cf in enumerate(x_counterfactual_list):
            original_x = x_list[idx]
            
            # 1. Model prediction on original
            with torch.no_grad():
                pred_original = self.model(original_x.unsqueeze(0).to(self.device)).squeeze(0).cpu().argmax(-1).item()
                # 2. Model prediction on counterfactual
                pred_cf = self.model(cf.unsqueeze(0).to(self.device)).squeeze(0).cpu().argmax(-1).item()
                student_pred_original = student(original_x.unsqueeze(0).to(self.device)).squeeze(0).cpu().argmax(-1).item()
                student_pred_cf = student(cf.unsqueeze(0).to(self.device)).squeeze(0).cpu().argmax(-1).item()

            if self.counterfactual_type == "1sided" and y_list[idx] != y_source_list[idx]:
                feedback.append("student originally wrong!")
            elif pred_original != y_list[idx]:
                feedback.append("teacher originally wrong!")
            elif y_target_end_confidence_list[idx] < 0.5:
                feedback.append("student not swapped!")
            elif student_pred_original == y_source_list[idx] and student_pred_cf != y_target_list[idx]:
                feedback.append("adversarial counterfactual!")
            else:
                if pred_original == pred_cf:
                    # Counterfactual didn't even flip the class for the teacher
                    feedback.append("not flipped")
                else:
                    # It flipped the class! Let's check why.
                    # We flip the confounder back to the value it had in the original image.
                    cf_reverted = cf.clone()
                    cf_reverted[self.confounder_idx] = original_x[self.confounder_idx]
                    
                    with torch.no_grad():
                        pred_cf_reverted = self.model(cf_reverted.unsqueeze(0).to(self.device)).squeeze(0).cpu().argmax(-1).item()
                        
                    # If reverting the confounder reverts the prediction back to the original class,
                    # it means the change in prediction was SOLELY driven by the confounder.
                    if pred_cf_reverted == pred_original:
                        feedback.append("false")
                    else:
                        feedback.append("true")

        if is_train:
            self.model.train()

        return feedback

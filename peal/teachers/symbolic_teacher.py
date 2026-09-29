"""Teacher for tabular data with a known symbolic confounder feature.

Teachers are PEAL's sources of feedback on counterfactuals. This one needs no
human or oracle image model: it works on feature vectors whose attributes are
listed in ``dataset.attributes`` and decides whether a counterfactual is a
genuine explanation or one that merely toggled the named confounder column.
"""

import torch
from peal.teachers.interfaces import TeacherInterface


class SymbolicTeacher(TeacherInterface):
    """Teacher that judges tabular counterfactuals by reverting one confounder.

    For every counterfactual the confounder column is set back to its original
    value; if that alone restores the reference model's original prediction the
    class change was driven solely by the confounder and the counterfactual is
    flagged ``"false"``, otherwise ``"true"``.

    Parameters
    ----------
    model : torch.nn.Module
        Reference classifier (the "teacher" model) mapping a feature vector of
        shape ``(num_features,)`` to class logits. Its device determines
        ``self.device``.
    dataset
        Dataset exposing ``attributes``, the ordered list of feature names.
    confounder_name : str
        Name of the confounder feature; must be in ``dataset.attributes``.
    tracking_level : int, optional
        Stored only; not used by this teacher.
    counterfactual_type : str, optional
        ``"1sided"`` (default) additionally rejects samples the student already
        misclassified before the edit.

    Attributes
    ----------
    confounder_idx : int
        Column index of the confounder inside the feature vector.

    Raises
    ------
    ValueError
        If ``confounder_name`` is not among the dataset attributes.
    """

    def __init__(
        self,
        model,
        dataset,
        confounder_name: str,
        tracking_level=0,
        counterfactual_type="1sided",
    ):
        """Resolve the confounder column index and the reference model's device.

        Raises
        ------
        ValueError
            If ``confounder_name`` is not in ``dataset.attributes``.
        """
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
            raise ValueError(
                f"Confounder '{self.confounder_name}' not found in dataset attributes: {features}"
            )

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
        **kwargs,
    ):
        """Label every counterfactual as ``"true"``, ``"false"`` or a rejection.

        Parameters
        ----------
        x_counterfactual_list : sequence of torch.Tensor
            Counterfactual feature vectors, each of shape ``(num_features,)``.
        y_source_list : sequence of int
            Class predicted by the student for the original sample.
        x_list : sequence of torch.Tensor
            Original feature vectors matching ``x_counterfactual_list``.
        y_list : sequence of int
            Ground-truth labels of the originals.
        y_target_end_confidence_list : sequence of float
            Student confidence in the target class after the edit.
        base_dir : optional
            Unused; kept for interface compatibility.
        y_target_list : sequence of int, optional
            Target class of every counterfactual.
        student : torch.nn.Module
            The classifier being repaired; required despite the default.
        mode : str, optional
            Unused; kept for interface compatibility.

        Returns
        -------
        list of str
            One entry per counterfactual, in order of precedence:
            ``"student originally wrong!"`` (1-sided mode, student label differs
            from ground truth), ``"teacher originally wrong!"``,
            ``"student not swapped!"`` (confidence below 0.5),
            ``"adversarial counterfactual!"`` (student flipped but the edit did
            not reach ``y_target``), ``"not flipped"`` (teacher model keeps its
            class), ``"false"`` (flip explained by the confounder alone) or
            ``"true"``.

        Notes
        -----
        The model is switched to eval mode for the duration of the call and
        restored afterwards.
        """
        feedback = []
        is_train = self.model.training
        self.model.eval()

        for idx, cf in enumerate(x_counterfactual_list):
            original_x = x_list[idx]

            # 1. Model prediction on original
            with torch.no_grad():
                pred_original = (
                    self.model(original_x.unsqueeze(0).to(self.device))
                    .squeeze(0)
                    .cpu()
                    .argmax(-1)
                    .item()
                )
                # 2. Model prediction on counterfactual
                pred_cf = (
                    self.model(cf.unsqueeze(0).to(self.device))
                    .squeeze(0)
                    .cpu()
                    .argmax(-1)
                    .item()
                )
                student_pred_original = (
                    student(original_x.unsqueeze(0).to(self.device))
                    .squeeze(0)
                    .cpu()
                    .argmax(-1)
                    .item()
                )
                student_pred_cf = (
                    student(cf.unsqueeze(0).to(self.device))
                    .squeeze(0)
                    .cpu()
                    .argmax(-1)
                    .item()
                )

            if (
                self.counterfactual_type == "1sided"
                and y_list[idx] != y_source_list[idx]
            ):
                feedback.append("student originally wrong!")
            elif pred_original != y_list[idx]:
                feedback.append("teacher originally wrong!")
            elif y_target_end_confidence_list[idx] < 0.5:
                feedback.append("student not swapped!")
            elif (
                student_pred_original == y_source_list[idx]
                and student_pred_cf != y_target_list[idx]
            ):
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
                        pred_cf_reverted = (
                            self.model(cf_reverted.unsqueeze(0).to(self.device))
                            .squeeze(0)
                            .cpu()
                            .argmax(-1)
                            .item()
                        )

                    # If reverting the confounder reverts the prediction back to the original class,
                    # it means the change in prediction was SOLELY driven by the confounder.
                    if pred_cf_reverted == pred_original:
                        feedback.append("false")
                    else:
                        feedback.append("true")

        if is_train:
            self.model.train()

        return feedback

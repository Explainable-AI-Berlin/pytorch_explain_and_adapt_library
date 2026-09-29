"""Oracle teacher that judges counterfactuals with a second classifier.

Teachers are PEAL's sources of feedback on explanations. This one replaces the
human annotator by a reference ("teacher") model that is assumed to be
unconfounded: a counterfactual is accepted as a genuine explanation when it
flips the reference model's prediction too, and rejected as a shortcut edit
when the reference model is unimpressed. It is the cheap stand-in used to run
CFKD-style repair loops without a human in the loop; the noisy variant below
simulates an unreliable annotator.
"""

import random


from peal.teachers.interfaces import TeacherInterface
from peal.log import get_logger

_log = get_logger(__name__)


class Model2ModelTeacher(TeacherInterface):
    """Teacher that asks a reference classifier whether a counterfactual is real.

    For every counterfactual the reference model is evaluated on the original
    image and on the edit. Verdicts are returned as strings: ``"true"`` when
    the reference prediction changed, ``"false"`` when it did not, and one of
    several rejection reasons when the sample never qualified in the first
    place (``"student originally wrong!"``, ``"teacher originally wrong!"``,
    ``"student not swapped!"``).

    Parameters
    ----------
    model : torch.nn.Module
        Reference classifier producing class logits. It is switched to eval
        mode for the duration of ``get_feedback`` and restored afterwards; its
        device determines ``self.device``.
    dataset
        Dataset used for ``calculate_outlier_score`` and, at high tracking
        levels, ``generate_contrastive_collage``.
    tracking_level : int, optional
        From level 5 on, a contrastive collage of the judged counterfactuals
        is written below ``base_dir``.
    counterfactual_type : str, optional
        ``"1sided"`` (default) additionally rejects samples whose ground-truth
        label already disagrees with the student's source prediction.
    """

    def __init__(self, model, dataset, tracking_level=0, counterfactual_type="1sided"):
        """Store the reference model and dataset and derive the model's device."""
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
        **kwargs,
    ):
        """Return one verdict per counterfactual.

        Parameters
        ----------
        x_counterfactual_list : sequence of torch.Tensor
            Counterfactual images of shape ``(C, H, W)``, aligned with
            ``x_list``.
        y_source_list : sequence
            The student's predicted class for each original image.
        x_list : sequence of torch.Tensor
            The original images.
        y_list : sequence
            Ground-truth labels of the original images.
        y_target_end_confidence_list : sequence of float
            The student's confidence in the target class after the edit; a
            value below 0.5 means the edit failed to flip the student.
        base_dir : str, optional
            Directory the contrastive collage is written to.
        y_target_list : sequence, optional
            The class each counterfactual was steered towards.
        student : torch.nn.Module, optional
            The classifier under repair; used for the logged before/after
            predictions.
        mode : str, optional
            ``"train"`` or ``"validation"``; only selects whether a collage is
            produced.
        **kwargs
            Forwarded to ``dataset.generate_contrastive_collage``.

        Returns
        -------
        list of str
            One verdict per counterfactual, in input order.
        """
        feedback = []
        teacher_original = []
        teacher_counterfactual = []
        is_train = self.model.training
        self.model.eval()
        #### uncomment and use for sce results, unfortunately the x_list is not matching counterfactual list due to a bug
        # Counterfactuals were generated in batches of 4, with 2 per image
        # cf[0-3, 4-7] → x[0-3], cf[8-11, 12-15] → x[4-7], etc.
        # batch_size = 4
        # counterfactuals_per_batch = batch_size * 2

        # for cf_idx, counterfactual in enumerate(x_counterfactual_list):
        #     # Calculate which original image this counterfactual corresponds to
        #     idx = (cf_idx // counterfactuals_per_batch) * batch_size + (
        #         cf_idx % batch_size
        #     )

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
            outlier_score = float(
                self.dataset.calculate_outlier_score(counterfactual.unsqueeze(0))[
                    "relative"
                ]
            )
            student_pred_original = (
                student(x_list[idx].unsqueeze(0).to(self.device))
                .cpu()
                .argmax(-1)
                .item()
            )
            student_pred = (
                student(counterfactual.unsqueeze(0).to(self.device))
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
                feedback.append(
                    "student not swapped!"
                    + " "
                    + str(student_pred_original)
                    + " -> "
                    + str(student_pred)
                )

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
                    _log.info("%s", [int(prediction), int(y_counterfactual)])
                    feedback.append("false")

            teacher_original.append(pred_original)
            teacher_counterfactual.append(pred_counterfactual)

        if (
            self.tracking_level >= 5
            and mode == "validation"
            or self.tracking_level >= 5
            and mode == "train"
        ):
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


class NoisyModel2ModelTeacher(Model2ModelTeacher):
    """Model-to-model teacher that flips a fraction of its verdicts.

    Simulates an unreliable annotator: whenever the reference model reaches a
    ``"true"``/``"false"`` verdict, it is inverted with probability
    ``noise_prob``. The rejection reasons are left untouched. Unlike the
    parent, this variant also rejects out-of-distribution edits
    (``"ood_<score>"`` above a relative outlier score of 2.5) and edits that
    moved the student somewhere other than the target class
    (``"adversarial counterfactual!"``).

    Parameters
    ----------
    model, dataset, tracking_level, counterfactual_type
        As in :class:`Model2ModelTeacher`.
    noise_prob : float, optional
        Probability of inverting a decided verdict. Default 0.1.
    """

    def __init__(
        self,
        model,
        dataset,
        tracking_level=0,
        counterfactual_type="1sided",
        noise_prob=0.1,
    ):
        """Initialise the base teacher and store the label-flip probability."""
        super().__init__(model, dataset, tracking_level, counterfactual_type)
        self.noise_prob = noise_prob

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
        """Return one verdict per counterfactual.

        Parameters
        ----------
        x_counterfactual_list : sequence of torch.Tensor
            Counterfactual images of shape ``(C, H, W)``, aligned with
            ``x_list``.
        y_source_list : sequence
            The student's predicted class for each original image.
        x_list : sequence of torch.Tensor
            The original images.
        y_list : sequence
            Ground-truth labels of the original images.
        y_target_end_confidence_list : sequence of float
            The student's confidence in the target class after the edit; a
            value below 0.5 means the edit failed to flip the student.
        base_dir : str, optional
            Directory the contrastive collage is written to.
        y_target_list : sequence, optional
            The class each counterfactual was steered towards.
        student : torch.nn.Module, optional
            The classifier under repair; used for the logged before/after
            predictions.
        mode : str, optional
            ``"train"`` or ``"validation"``; only selects whether a collage is
            produced.
        **kwargs
            Forwarded to ``dataset.generate_contrastive_collage``.

        Returns
        -------
        list of str
            One verdict per counterfactual, in input order.
        """
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
            outlier_score = float(
                self.dataset.calculate_outlier_score(counterfactual.unsqueeze(0))[
                    "relative"
                ]
            )
            student_pred_original = (
                student(x_list[idx].unsqueeze(0).to(self.device))
                .cpu()
                .argmax(-1)
                .item()
            )
            student_pred = (
                student(counterfactual.unsqueeze(0).to(self.device))
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

            elif outlier_score > 2.5:
                feedback.append("ood_" + str(round(outlier_score, 2)))

            elif (
                student_pred_original == y_source_list[idx]
                and student_pred != y_target_list[idx]
            ):
                feedback.append("adversarial counterfactual!")

            else:
                if pred_original != pred_counterfactual:
                    fb = "true"

                else:
                    y_counterfactual = y_source_list[idx]
                    prediction = (
                        student(counterfactual.unsqueeze(0).to(self.device))
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .argmax(-1)
                    )
                    _log.info("%s", [int(prediction), int(y_counterfactual)])
                    fb = "false"

                # Add noise
                if random.random() < self.noise_prob:
                    fb = "false" if fb == "true" else "true"

                feedback.append(fb)

            teacher_original.append(pred_original)
            teacher_counterfactual.append(pred_counterfactual)

        if (
            self.tracking_level >= 5
            and mode == "validation"
            or self.tracking_level >= 5
            and mode == "train"
        ):
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

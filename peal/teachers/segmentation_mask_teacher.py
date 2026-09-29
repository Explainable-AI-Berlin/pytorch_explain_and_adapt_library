"""
A teacher that judges counterfactuals with ground-truth segmentation masks.

Teachers are PEAL's sources of feedback on counterfactuals. This one uses the
dataset's hint masks (a binary mask of the region that legitimately carries the
class, e.g. the object rather than the background) to decide whether the
attribution map of a counterfactual concentrates on the right region. It is
the automatic stand-in for a human annotator in CFKD experiments on datasets
with segmentation hints.
"""

from peal.teachers.interfaces import TeacherInterface


class SegmentationMaskTeacher(TeacherInterface):
    """
    Teacher that scores attribution maps against segmentation hints.

    For every counterfactual the (mean-centred) hint mask is multiplied with
    the attribution heatmap; if the mean of that product exceeds
    ``attribution_threshold`` the change lies inside the annotated region and
    the feedback is ``"true"``, otherwise ``"false"``.

    Parameters
    ----------
    attribution_threshold : float
        Decision threshold on the mean of ``heatmap * (hint - hint.mean())``.
    dataset : peal dataset
        Dataset providing ``generate_contrastive_collage`` for tracking.
    counterfactual_type : {"1sided", "2sided"}
        With ``"1sided"`` a counterfactual whose original prediction was
        wrong gets the feedback ``"student originally wrong!"``.
    tracking_level : int
        Verbosity level; ``>= 5`` writes collage images to ``base_dir``.
    """

    def __init__(
        self,
        attribution_threshold,
        dataset,
        counterfactual_type="1sided",
        tracking_level=0,
    ):
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
        **kwargs,
    ):
        """
        Produce one feedback string per counterfactual.

        Parameters
        ----------
        x_attribution_list : list of torch.Tensor
            Attribution heatmap of each counterfactual (same spatial shape as
            the hint mask).
        hint_list : list of torch.Tensor
            Binary segmentation mask of the legitimate region per sample.
        base_dir : str
            Directory the collage is written to when ``tracking_level >= 5``.
        x_counterfactual_list : list of torch.Tensor
            Counterfactual images.
        y_source_list : list
            Class the student predicted for each original sample.
        y_target_list : list
            Target class of each counterfactual.
        x_list : list of torch.Tensor
            Original images.
        y_list : list
            Ground-truth labels of the originals.
        y_target_end_confidence_list : list of float
            Student confidence for the target class on the counterfactual.
        student : torch.nn.Module
            The classifier being explained; only used to determine the device.
        **kwargs
            Forwarded to ``dataset.generate_contrastive_collage``.

        Returns
        -------
        list of str
            One of ``"true"``, ``"false"``, ``"student originally wrong!"``
            or ``"student not swapped!"`` per counterfactual.
        """
        feedback = []
        device = "cuda" if next(student.parameters()).is_cuda else "cpu"
        for idx, heatmap in enumerate(x_attribution_list):
            #
            if (
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
                true_counterfactual_score = joint_map.mean()
                if true_counterfactual_score > self.attribution_threshold:
                    feedback.append("true")

                else:
                    feedback.append("false")

        if self.tracking_level >= 5:
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

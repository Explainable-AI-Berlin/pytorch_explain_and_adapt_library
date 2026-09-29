"""Teacher that grades counterfactuals by a precomputed cluster assignment.

Teachers are PEAL's sources of feedback: given the counterfactuals an explainer
produced, they say whether each one shows a valid reason for the predictor's
decision ("true") or a spurious one ("false"). ``PreclusteredTeacher`` needs
neither a human nor an oracle model: the explainer has already clustered the
counterfactuals (``cluster_list``) and the caller declares which cluster ids
are the correct ones. It simulates feedback in CFKD/DiDAE experiments where the
meaning of every cluster is known beforehand.
"""

from peal.teachers.interfaces import TeacherInterface


class PreclusteredTeacher(TeacherInterface):
    """Feedback source that labels a counterfactual by its cluster membership.

    A counterfactual is ``"true"`` when its cluster id is in
    ``correct_clusters`` and ``"false"`` otherwise; the usual sanity gates
    (student already wrong, not flipped, out of distribution) are applied
    first.

    Parameters
    ----------
    dataset : PealDataset
        Dataset used for the outlier score (``calculate_outlier_score``) and,
        at high tracking levels, for rendering contrastive collages.
    correct_clusters : collection of int
        Cluster ids whose counterfactuals count as valid explanations.
    tracking_level : int, optional
        Verbosity; at 5 or above a collage of every feedback batch is written.
    counterfactual_type : str, optional
        ``"1sided"`` (default) rejects counterfactuals of samples the student
        misclassified in the first place; any other value skips that gate.
    """

    def __init__(
        self, dataset, correct_clusters, tracking_level=0, counterfactual_type="1sided"
    ):
        """Store the dataset, the correct cluster ids and the tracking options."""
        self.dataset = dataset
        self.tracking_level = tracking_level
        self.counterfactual_type = counterfactual_type
        self.correct_clusters = correct_clusters

    def get_feedback(
        self,
        x_counterfactual_list,
        y_source_list,
        x_list,
        y_list,
        y_target_end_confidence_list,
        cluster_list,
        base_dir=None,
        y_target_list=None,
        mode="train",
        **kwargs,
    ):
        """Return one feedback string per counterfactual.

        Parameters
        ----------
        x_counterfactual_list : sequence of torch.Tensor
            Counterfactual inputs, each without batch dimension.
        y_source_list : sequence of int
            Class the predictor assigned to the original sample.
        x_list : sequence of torch.Tensor
            Original samples; only forwarded to the collage writer.
        y_list : sequence of int
            Ground-truth labels of the original samples.
        y_target_end_confidence_list : sequence of float
            Predictor confidence in the target class after the edit.
        cluster_list : sequence of int
            Cluster id of every counterfactual, as stamped by
            ``CounterfactualExplainer.cluster_explanations``.
        base_dir : str, optional
            Directory the collages are written to when ``tracking_level >= 5``.
        y_target_list : sequence of int, optional
            Target classes, forwarded to the collage writer.
        mode : str, optional
            ``"train"`` or ``"validation"``; collages are written for both.
        **kwargs
            Passed through to ``dataset.generate_contrastive_collage``.

        Returns
        -------
        list of str
            Per counterfactual, in order of precedence: ``"student originally
            wrong!"`` (1sided and ``y != y_source``), ``"student not
            swapped!"`` (target confidence below 0.5), ``"ood_<score>"``
            (relative outlier score above 4.0), else ``"true"`` or ``"false"``
            by cluster membership.

        Notes
        -----
        The teacher labels handed to the collage writer are ``-1``
        placeholders; this teacher never relabels the images themselves.
        """
        feedback = []
        teacher_original = []
        teacher_counterfactual = []
        for idx, counterfactual in enumerate(x_counterfactual_list):
            outlier_score = float(
                self.dataset.calculate_outlier_score(counterfactual.unsqueeze(0))[
                    "relative"
                ]
            )

            if (
                self.counterfactual_type == "1sided"
                and y_list[idx] != y_source_list[idx]
            ):
                feedback.append("student originally wrong!")

            elif y_target_end_confidence_list[idx] < 0.5:
                feedback.append("student not swapped!")

            elif outlier_score > 4.0:
                feedback.append("ood_" + str(round(outlier_score, 2)))

            else:
                if cluster_list[idx] in self.correct_clusters:
                    feedback.append("true")

                else:
                    feedback.append("false")

            teacher_original.append(-1)
            teacher_counterfactual.append(-1)

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

        return feedback

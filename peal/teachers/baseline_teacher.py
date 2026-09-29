"""
A teacher that answers with a fixed or random verdict.

Teachers are PEAL's sources of feedback on counterfactuals: given an original
sample and its counterfactual they say whether the change was a valid
("true") or spurious ("false") reason for the classifier's decision. The
``BaselineTeacher`` ignores the images entirely and answers according to a
strategy (random / always true / always false). It serves as a control
condition for adaptors such as CFKD.
"""

import numpy as np

from peal.teachers.interfaces import TeacherInterface


class BaselineTeacher(TeacherInterface):
    """
    Content-agnostic teacher used as a baseline for CFKD experiments.

    The feedback for every counterfactual is decided by ``strategy`` without
    looking at the sample, unless the counterfactual is disqualified first
    (student originally wrong on a one-sided counterfactual, or student not
    swapped to the target class).

    Parameters
    ----------
    strategy : {"random", "true", "false"}
        ``"random"`` draws a fair coin per counterfactual, ``"true"`` and
        ``"false"`` always give that verdict.
    dataset : peal dataset, optional
        Only needed for ``tracking_level >= 5``; its
        ``generate_contrastive_collage`` is used to render the feedback.
    tracking_level : int
        Verbosity level; ``>= 5`` writes collage images to ``base_dir``.
    counterfactual_type : {"1sided", "2sided"}
        With ``"1sided"`` a counterfactual whose original prediction was
        wrong gets the feedback ``"student originally wrong!"``.
    """

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
        **kwargs,
    ):
        """
        Produce one feedback string per counterfactual.

        Parameters
        ----------
        x_counterfactual_list : list of torch.Tensor
            Counterfactual images.
        y_source_list : list
            Class the student predicted for each original sample.
        x_list : list of torch.Tensor
            Original images.
        y_list : list
            Ground-truth labels of the originals.
        y_target_end_confidence_list : list of float
            Student confidence for the target class on the counterfactual.
        base_dir : str, optional
            Directory the collage is written to when ``tracking_level >= 5``.
        y_target_list : list, optional
            Target class of each counterfactual.
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
        teacher_original = []
        teacher_counterfactual = []
        device = "cuda" if next(student.parameters()).is_cuda else "cpu"
        for idx, counterfactual in enumerate(x_counterfactual_list):

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

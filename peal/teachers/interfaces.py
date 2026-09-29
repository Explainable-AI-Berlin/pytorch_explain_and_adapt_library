"""Minimal interface every PEAL teacher implements.

A teacher is the source of feedback in the explain-and-adapt loop: given the
explanations an explainer produced (collage paths, confidences, labels), it
returns one verdict per explanation that an adaptor such as CFKD or DiDAE
then learns from. Concrete teachers (human GUI, LLM, model-to-model, cluster,
segmentation-mask, ...) live in the sibling modules.
"""

from abc import ABC, abstractmethod


class TeacherInterface(ABC):
    """Duck-typed base class of all teachers.

    Subclasses override ``get_feedback``; the base implementation only raises so
    that a mis-configured teacher fails loudly instead of silently returning
    nothing.
    """

    @abstractmethod
    def get_feedback(self, **args):
        """Return one feedback label per explanation.

        Parameters
        ----------
        **args
            Explainer outputs; the exact keywords depend on the teacher (typically
            ``collage_path_list``, ``y_target_end_confidence_list``,
            ``y_source_list`` and ``y_list``).

        Returns
        -------
        list
            One verdict per explanation, in the order the explanations were given.

        Raises
        ------
        Exception
            Always, in this base class: subclasses must implement the method.
        """
        raise NotImplementedError

"""Factory that turns a ``teacher`` config value into a ``TeacherInterface``.

Teachers are PEAL's sources of feedback on counterfactuals (human GUI, an
oracle model, an LLM, symbolic rules, segmentation masks, ...). Adaptor
configs such as CFKD's specify the teacher as a string
(``"human@8000"``, ``"Baseline:<strategy>"``, ``"web:<dir>"``, a ``.cpl``
model path, ...), a dict with a ``type`` key, an ``nn.Module`` or an already
built teacher; ``get_teacher`` resolves all of these to one object.
"""

import torch

from torch import nn
from typing import Union

from peal.teachers.baseline_teacher import BaselineTeacher
from peal.teachers.cluster_teacher import ClusterTeacher
from peal.teachers.interfaces import TeacherInterface
from peal.teachers.human2model_teacher import Human2ModelTeacher
from peal.teachers.model2model_teacher import (
    Model2ModelTeacher,
    NoisyModel2ModelTeacher,
)
from peal.teachers.symbolic_teacher import SymbolicTeacher

from peal.teachers.preclustered_teacher import PreclusteredTeacher
from peal.teachers.segmentation_mask_teacher import SegmentationMaskTeacher
from peal.data.interfaces import PealDataset


def get_teacher(
    teacher: Union[TeacherInterface, nn.Module, str],
    output_size: int,
    adaptor_config: Union[dict, str],
    dataset: PealDataset,
    device: Union[torch.device, str] = torch.device("cpu"),
    tracking_level: int = 0,
    counterfactual_type: str = "1sided",
) -> TeacherInterface:
    """
    Build the teacher described by ``teacher``.

    The accepted specifications are, in dispatch order:

    * a ``TeacherInterface`` instance, returned unchanged;
    * an ``nn.Module``, wrapped as an eval-mode ``Model2ModelTeacher``;
    * a dict with ``type: symbolic`` (``model`` path, ``confounder_name``),
      ``type: llm`` (``LLM2ModelTeacher`` driven by a ``claude -p``
      subprocess), ``type: web`` (``WebFeedbackTeacher`` polling a directory)
      or ``type: NoisyModel2Model`` (``model`` path, ``noise_prob``);
    * strings ``"web:<dir>"``, ``"preclustered"`` (uses
      ``adaptor_config.correct_clusters``), ``"cluster[XXXX]"``,
      ``"virelay[XXXX]"``, ``"human[XXXX]"`` (``XXXX`` an optional four digit
      port, default 8000), ``"SegmentationMask"`` (uses
      ``adaptor_config.attribution_threshold``), ``"Baseline:<strategy>"`` and
      a ``*.cpl`` path to a torch-saved oracle model.

    Saved models are loaded with ``torch.load`` and fall back to
    ``weights_only=False`` when the default load fails.

    Parameters
    ----------
    teacher : TeacherInterface or nn.Module or str or dict
        Teacher specification as listed above.
    output_size : int
        Number of classifier outputs. Currently unused by every branch.
    adaptor_config : object
        Adaptor config; only ``correct_clusters`` and
        ``attribution_threshold`` are read, for the preclustered and
        segmentation-mask teachers.
    dataset : PealDataset
        Dataset the teacher judges counterfactuals of.
    device : torch.device or str, optional
        ``map_location`` for ``torch.load``. Defaults to CPU.
    tracking_level : int, optional
        Verbosity of intermediate artifacts, forwarded to the teacher.
    counterfactual_type : str, optional
        ``"1sided"`` or ``"2sided"``, forwarded to the teacher.

    Returns
    -------
    TeacherInterface
        The constructed teacher.

    Raises
    ------
    ValueError
        If ``teacher`` matches none of the specifications.
    """
    if isinstance(teacher, TeacherInterface):
        teacher = teacher

    elif isinstance(teacher, nn.Module):
        teacher.eval()
        teacher = Model2ModelTeacher(
            teacher,
            dataset,
            tracking_level=tracking_level,
            counterfactual_type=counterfactual_type,
        )

    elif isinstance(teacher, dict) and teacher.get("type") == "symbolic":
        model_path = teacher.get("model")
        try:
            loaded_teacher = torch.load(model_path, map_location=device)
        except Exception:
            loaded_teacher = torch.load(
                model_path, map_location=device, weights_only=False
            )

        teacher = SymbolicTeacher(
            loaded_teacher,
            dataset=dataset,
            confounder_name=teacher.get("confounder_name"),
            tracking_level=tracking_level,
            counterfactual_type=counterfactual_type,
        )

    elif isinstance(teacher, dict) and teacher.get("type") == "llm":
        # LLM-in-the-loop teacher: answers the visual counterfactual questions
        # via a headless `claude -p` subprocess instead of the human Flask GUI.
        # Drop-in for Human2ModelTeacher ("teacher: human@8000").
        from peal.teachers.llm2model_teacher import LLM2ModelTeacher

        teacher = LLM2ModelTeacher(
            dataset=dataset,
            confounder_name=teacher.get("confounder_name"),
            target_name=teacher.get("target_name"),
            model=teacher.get("model"),
            claude_bin=teacher.get("claude_bin", "claude"),
            tracking_level=tracking_level,
            counterfactual_type=counterfactual_type,
            batch_size=teacher.get("batch_size", 16),
            stage_dir=teacher.get("stage_dir"),
        )
    elif (isinstance(teacher, dict) and teacher.get("type") == "web") or (
        isinstance(teacher, str) and teacher.startswith("web:")
    ):
        # File-based feedback for the web demo: the job worker writes the
        # collages it needs verdicts for into the job directory and the web app
        # collects them from the uploader. "web:<dir>" or
        # {"type": "web", "dir": <dir>, "timeout_s": ..., "auto_verdict": ...}.
        from peal.teachers.web_feedback_teacher import WebFeedbackTeacher

        if isinstance(teacher, str):
            teacher = {"type": "web", "dir": teacher[len("web:") :]}
        teacher = WebFeedbackTeacher(
            feedback_dir=teacher["dir"],
            poll_interval=teacher.get("poll_interval", 2.0),
            timeout_s=teacher.get("timeout_s"),
            auto_verdict=teacher.get("auto_verdict"),
            tracking_level=tracking_level,
            counterfactual_type=counterfactual_type,
        )

    elif isinstance(teacher, dict) and teacher.get("type") == "NoisyModel2Model":
        model_path = teacher.get("model")
        try:
            loaded_teacher = torch.load(model_path, map_location=device)

        except Exception:
            loaded_teacher = torch.load(
                model_path, map_location=device, weights_only=False
            )

        teacher = NoisyModel2ModelTeacher(
            loaded_teacher,
            dataset=dataset,
            noise_prob=teacher.get("noise_prob", 0.1),
            tracking_level=tracking_level,
            counterfactual_type=counterfactual_type,
        )

    elif isinstance(teacher, str) and teacher == "preclustered":
        teacher = PreclusteredTeacher(
            dataset=dataset,
            correct_clusters=adaptor_config.correct_clusters,
            tracking_level=tracking_level,
            counterfactual_type=counterfactual_type,
        )

    elif isinstance(teacher, str) and teacher[:7] == "cluster":
        if len(teacher) == 12:
            port = int(teacher[-4:])

        else:
            port = 8000

        teacher = ClusterTeacher(
            port,
            dataset=dataset,
            tracking_level=tracking_level,
        )

    elif isinstance(teacher, str) and teacher[:7] == "virelay":
        if len(teacher) == 12:
            port = int(teacher[-4:])

        else:
            port = 8000

        # Imported here, not at module level: the ViRelAy teacher builds
        # corelay processors at import time, so a module-level import
        # would make corelay, virelay and h5py hard dependencies of
        # every PEAL install rather than the `xai` extra.
        from peal.teachers.virelay_teacher import VirelayTeacher

        teacher = VirelayTeacher(
            port=port,
            dataset=dataset,
            tracking_level=tracking_level,
        )

    elif isinstance(teacher, str) and teacher[:5] == "human":
        if len(teacher) == 10:
            port = int(teacher[-4:])

        else:
            port = 8000

        teacher = Human2ModelTeacher(port)

    elif isinstance(teacher, str) and teacher == "SegmentationMask":
        teacher = SegmentationMaskTeacher(
            adaptor_config.attribution_threshold,
            dataset=dataset,
            tracking_level=tracking_level,
            counterfactual_type=counterfactual_type,
        )

    elif isinstance(teacher, str) and teacher[: len("Baseline")] == "Baseline":
        teacher = BaselineTeacher(
            strategy=teacher.split(":")[1],
            dataset=dataset,
            tracking_level=tracking_level,
            counterfactual_type=counterfactual_type,
        )

    elif isinstance(teacher, str) and teacher[-4:] == ".cpl":
        try:
            loaded_teacher = torch.load(teacher, map_location=device)

        except Exception:
            loaded_teacher = torch.load(
                teacher, map_location=device, weights_only=False
            )

        teacher = Model2ModelTeacher(
            loaded_teacher,
            dataset=dataset,
            tracking_level=tracking_level,
            counterfactual_type=counterfactual_type,
        )

    else:
        raise ValueError(f"Unknown teacher {teacher}")

    return teacher

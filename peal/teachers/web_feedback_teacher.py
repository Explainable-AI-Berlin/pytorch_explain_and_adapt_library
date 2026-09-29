"""
File-based feedback teacher for the web demo.

``Human2ModelTeacher`` runs its own Flask server on a port and blocks until a
person has clicked through every collage. A job worker behind another web app
cannot do that, so this teacher talks to the web app through the job directory:

    <feedback_dir>/feedback_request_<round>.json    written by the teacher
    <feedback_dir>/feedback_response_<round>.json   written by the web app

The request lists the collages that need a verdict (index, path, source and
target class, end confidence); the response maps index -> "true" | "false" |
"ood". The teacher polls for the response file and returns the verdict list in
the order of ``collage_path_list``, with the same sentinels Human2ModelTeacher
emits for pairs that never reach a person ("student incorrect!" when the
student was wrong on the original, "student not swapped!" when the edit did not
cross the boundary).

``PEAL_WEB_FEEDBACK_AUTO`` = "true" | "false" | "ood" answers every request
without a person (tests, smoke runs).
"""

import json
import os
import time
import uuid

from peal.teachers.interfaces import TeacherInterface
from peal.log import get_logger

_log = get_logger(__name__)


VERDICTS = ("true", "false", "ood")


def _atomic_write_json(path, payload):
    """Write ``payload`` as JSON via a temp file + ``os.replace`` so the web app
    never reads a half-written request."""
    tmp = f"{path}.{uuid.uuid4().hex}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


class WebFeedbackTeacher(TeacherInterface):
    """Teacher that collects human verdicts through JSON files in a job directory.

    Used by the web demo's job worker in place of ``Human2ModelTeacher``. Each
    call to :meth:`get_feedback` is one *round*: the collages that need a
    verdict are written to ``feedback_request_<round>.json`` and the teacher
    blocks until ``feedback_response_<round>.json`` contains a complete set
    of valid verdicts.

    Parameters
    ----------
    feedback_dir : str
        Directory shared with the web app; created if missing.
    poll_interval : float
        Seconds between checks for the response file.
    timeout_s : float or None
        Give up (``TimeoutError``) after this many seconds without a complete
        response; ``None`` waits forever.
    auto_verdict : str or None
        ``"true"``, ``"false"`` or ``"ood"`` answers every request without a
        person. Falls back to the ``PEAL_WEB_FEEDBACK_AUTO`` environment
        variable when ``None``.
    min_confidence : float
        Counterfactuals whose target-class end confidence is below this value
        are answered ``"student not swapped!"`` without asking anyone.
    tracking_level : int
        Stored for interface compatibility; unused here.
    counterfactual_type : str
        ``"1sided"`` asks only about pairs where the student's prediction on
        the original equals the label (others get ``"student incorrect!"``);
        any other value asks about every swapped pair.

    Attributes
    ----------
    round : int
        Index of the next feedback round; incremented on every
        :meth:`get_feedback` call, including calls that need no verdict.

    Raises
    ------
    ValueError
        If ``auto_verdict`` is set to something other than the three verdicts.
    """

    def __init__(
        self,
        feedback_dir,
        poll_interval=2.0,
        timeout_s=None,
        auto_verdict=None,
        min_confidence=0.5,
        tracking_level=0,
        counterfactual_type="1sided",
    ):
        """Store the settings, resolve ``auto_verdict`` and create ``feedback_dir``."""
        self.feedback_dir = feedback_dir
        self.poll_interval = float(poll_interval)
        self.timeout_s = timeout_s
        self.auto_verdict = auto_verdict or os.environ.get("PEAL_WEB_FEEDBACK_AUTO")
        if self.auto_verdict is not None and self.auto_verdict not in VERDICTS:
            raise ValueError(
                f"auto_verdict must be one of {VERDICTS}, got {self.auto_verdict!r}"
            )
        self.min_confidence = float(min_confidence)
        self.tracking_level = tracking_level
        self.counterfactual_type = counterfactual_type
        self.round = 0
        os.makedirs(self.feedback_dir, exist_ok=True)

    # ---------------------------------------------------------------- paths
    def request_path(self, round_idx):
        """Path of the request file the teacher writes for round ``round_idx``.

        Parameters
        ----------
        round_idx : int
            Feedback round.

        Returns
        -------
        str
            ``<feedback_dir>/feedback_request_<round_idx>.json``.
        """
        return os.path.join(self.feedback_dir, f"feedback_request_{round_idx}.json")

    def response_path(self, round_idx):
        """Path of the response file the web app writes for round ``round_idx``.

        Parameters
        ----------
        round_idx : int
            Feedback round.

        Returns
        -------
        str
            ``<feedback_dir>/feedback_response_<round_idx>.json``.
        """
        return os.path.join(self.feedback_dir, f"feedback_response_{round_idx}.json")

    # ------------------------------------------------------------- feedback
    def get_feedback(
        self,
        collage_path_list,
        y_target_end_confidence_list,
        y_source_list,
        y_list,
        y_target_list=None,
        **kwargs,
    ):
        """Return one verdict per collage, asking the web app for the open ones.

        Parameters
        ----------
        collage_path_list : list of str
            Paths of the contrastive collages, one per counterfactual pair.
        y_target_end_confidence_list : list of float
            Predictor confidence for the target class on each counterfactual.
        y_source_list : list of int
            Student prediction on each original image.
        y_list : list of int
            Ground-truth label of each original image.
        y_target_list : list of int, optional
            Target class of each counterfactual; passed through to the request.
        **kwargs
            Ignored; accepted for compatibility with other teachers.

        Returns
        -------
        list of str
            Same length and order as ``collage_path_list``. Entries are
            ``"true"``, ``"false"`` or ``"ood"`` for pairs a person (or
            ``auto_verdict``) judged, ``"student incorrect!"`` when the student
            was wrong on the original (1-sided mode only) and
            ``"student not swapped!"`` when the end confidence is below
            ``min_confidence``.

        Notes
        -----
        Writes ``feedback_request_<round>.json`` with the items to judge and
        blocks in :meth:`_wait_for_response` until the matching response file
        holds a valid verdict for every requested index. The round counter is
        advanced even when nothing has to be asked.
        """
        n = len(collage_path_list)
        feedback = [None] * n
        items = []
        for i in range(n):
            student_correct = (
                self.counterfactual_type != "1sided" or y_source_list[i] == y_list[i]
            )
            swapped = float(y_target_end_confidence_list[i]) >= self.min_confidence
            if not student_correct:
                feedback[i] = "student incorrect!"
            elif not swapped:
                feedback[i] = "student not swapped!"
            else:
                items.append(
                    {
                        "index": i,
                        "collage": os.path.abspath(str(collage_path_list[i])),
                        "source_class": int(y_source_list[i]),
                        "target_class": (
                            int(y_target_list[i]) if y_target_list is not None else None
                        ),
                        "confidence": float(y_target_end_confidence_list[i]),
                    }
                )

        round_idx = self.round
        self.round += 1
        if len(items) == 0:
            return feedback

        if self.auto_verdict is not None:
            for it in items:
                feedback[it["index"]] = self.auto_verdict
            return feedback

        request = {
            "round": round_idx,
            "created": time.time(),
            "n_items": len(items),
            "items": items,
        }
        _atomic_write_json(self.request_path(round_idx), request)
        # A person may take hours to answer. Hand the allocator's cached but
        # unused GPU memory back meanwhile, so other processes (the web demo's
        # per-direction analysis) can use the GPU.
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        _log.info(
            "%s",
            f"[WebFeedbackTeacher] round {round_idx}: waiting for {len(items)} verdicts "
            f"in {self.response_path(round_idx)}",
        )
        verdicts = self._wait_for_response(round_idx, {it["index"] for it in items})
        for it in items:
            feedback[it["index"]] = verdicts[it["index"]]
        return feedback

    def _wait_for_response(self, round_idx, expected):
        """Poll the response file until ``payload["verdicts"]`` covers every
        index in ``expected`` with a valid verdict; returns ``{index: verdict}``."""
        path = self.response_path(round_idx)
        started = time.time()
        while True:
            if os.path.isfile(path):
                try:
                    with open(path) as f:
                        payload = json.load(f)
                except (OSError, json.JSONDecodeError):
                    payload = None
                if isinstance(payload, dict) and "verdicts" in payload:
                    verdicts = {int(k): str(v) for k, v in payload["verdicts"].items()}
                    missing = expected - set(verdicts)
                    bad = {k: v for k, v in verdicts.items() if v not in VERDICTS}
                    if not missing and not bad:
                        return verdicts
                    _log.info(
                        "%s",
                        f"[WebFeedbackTeacher] response incomplete "
                        f"(missing {sorted(missing)[:5]}..., invalid {bad}); waiting",
                    )
            if self.timeout_s is not None and time.time() - started > self.timeout_s:
                raise TimeoutError(
                    f"no feedback for round {round_idx} within {self.timeout_s} s"
                )
            time.sleep(self.poll_interval)

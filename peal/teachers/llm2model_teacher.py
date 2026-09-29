"""
LLM-in-the-loop teacher.

`LLM2ModelTeacher` is a drop-in replacement for `Human2ModelTeacher`: it answers
the same visual counterfactual-feedback questions, but instead of a human clicking
the Flask GUI it shells out to a headless Claude Code process (`claude -p
--output-format json`). This lets DiDAE / CFKD runs that would otherwise block on
a browser run fully unattended, while keeping the *shape* of human feedback so a
real human (`teacher: human@8000`) can be swapped back in without any other change.

Each collage handed to the teacher is a side-by-side grid: the LEFT half is the
original image, the RIGHT half is the counterfactual (see
DiDAE._collect_direction_collages, which saves `torch.cat([orig, cf])` with
nrow=2). The teacher's job, per collage, is the same judgement a human makes::

    "true"  – the edit changed the class-defining (target) feature: a valid
              counterfactual
    "false" – the edit changed the spurious / confounding feature instead
    "ood"   – the edited image is not a plausible, in-distribution image

The `claude` binary is expected on PATH inside the run environment. Under the
lab's `appruns` Apptainer wrapper it is bind-mounted to /usr/local/bin/claude and
authenticated from the mounted ~/.claude credentials, so no API key is needed.
"""

import json
import os
import shutil
import tempfile

import subprocess

from peal.teachers.interfaces import TeacherInterface
from peal.log import get_logger

_log = get_logger(__name__)


class LLM2ModelTeacher(TeacherInterface):
    """Teacher that labels counterfactual collages with a headless Claude call.

    Collages are staged into a temporary directory, described in one prompt and
    sent to ``claude -p --output-format json``; the reply must be a JSON array
    with one of ``"true"``, ``"false"`` or ``"ood"`` per collage (anything else
    becomes ``"false"``). ``get_feedback`` returns the same sentinel strings the
    human GUI produces for collages it does not show.

    Parameters
    ----------
    dataset : optional
        Kept for interface parity with ``Human2ModelTeacher``; not used.
    confounder_name : str or None
        Human-readable name of the spurious feature put into the prompt.
    target_name : str or None
        Human-readable name of the class-defining feature put into the prompt.
    model : str or None
        Passed as ``--model`` to the CLI when set.
    claude_bin : str
        Executable to run (``"claude"`` on PATH by default).
    tracking_level : int
        Kept for interface parity; not used.
    counterfactual_type : str
        Kept for interface parity; not used.
    batch_size : int
        Collages per CLI call.
    timeout_s : int
        Seconds before a CLI call is aborted (counts as a failed batch).
    stage_dir : str or None
        Directory the temporary staging folders are created in; ``None`` uses
        the system temp dir.
    """

    def __init__(
        self,
        dataset=None,
        confounder_name=None,
        target_name=None,
        model=None,
        claude_bin="claude",
        tracking_level=0,
        counterfactual_type="1sided",
        batch_size=16,
        timeout_s=300,
        stage_dir=None,
    ):
        """Store the prompt names, CLI options and batching parameters."""
        self.dataset = dataset
        self.confounder_name = confounder_name or "the spurious / confounding feature"
        self.target_name = target_name or "the class-defining (target) feature"
        self.model = model
        self.claude_bin = claude_bin
        self.tracking_level = tracking_level
        self.counterfactual_type = counterfactual_type
        self.batch_size = batch_size
        self.timeout_s = timeout_s
        # Where to stage collages before handing them to `claude -p`. None = the
        # system temp dir (TMPDIR / /tmp). Never cwd, which may be read-only.
        self.stage_dir = stage_dir

    # ------------------------------------------------------------------ helpers
    def _build_prompt(self, paths):
        """Compose the labelling instructions listing the staged collage paths."""
        lines = [
            "You are labelling counterfactual image edits for a debiasing experiment.",
            "Each image is a side-by-side collage: the LEFT half is the ORIGINAL image, "
            "the RIGHT half is the EDITED (counterfactual) version of it.",
            f"The classifier's TRUE target feature is: {self.target_name}.",
            f"The SPURIOUS / confounding feature is: {self.confounder_name}.",
            "For each collage, compare the right half to the left half and decide:",
            '  "true"  - the edit changed the TRUE target feature (a valid counterfactual)',
            '  "false" - the edit changed the SPURIOUS / confounding feature instead',
            '  "ood"   - the edited image is not a plausible, in-distribution image',
            "Read every image file listed below (use the Read tool), then reply with "
            f"ONLY a JSON array of exactly {len(paths)} strings (one verdict per image, "
            "in the listed order) and no other text:",
        ]
        for i, p in enumerate(paths):
            lines.append(f"  [{i}] {os.path.abspath(p)}")
        return "\n".join(lines)

    def _ask_claude(self, paths):
        # `claude -p` reads files inside its launch dir OR any --add-dir, regardless
        # of OS permissions, so stage the collages into a fresh temp dir and allow it
        # with --add-dir below. Stage in the system temp dir (self.stage_dir / TMPDIR
        # / /tmp), NEVER os.getcwd(): under a --no-home sandbox cwd can sit on a
        # read-only $HOME, where creating the dir raises "Read-only file system".
        """Stage the collages, run ``claude -p`` once and parse its JSON verdicts."""
        stage = tempfile.mkdtemp(prefix="llm_teacher_", dir=self.stage_dir)
        try:
            staged = []
            for i, p in enumerate(paths):
                dst = os.path.join(stage, f"{i:03d}_{os.path.basename(p)}")
                shutil.copy(p, dst)
                staged.append(dst)
            cmd = [
                self.claude_bin,
                "-p",
                self._build_prompt(staged),
                "--output-format",
                "json",
                "--add-dir",
                stage,
            ]
            if self.model:
                cmd += ["--model", self.model]
            proc = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        finally:
            shutil.rmtree(stage, ignore_errors=True)
        raw = (proc.stdout or "").strip()
        # `--output-format json` wraps the model's answer in an envelope; the text
        # we want is the "result" field. Fall back to the raw stdout otherwise.
        text = raw
        try:
            env = json.loads(raw)
            if isinstance(env, dict) and "result" in env:
                text = env["result"]
        except (json.JSONDecodeError, ValueError):
            pass
        start, end = text.find("["), text.rfind("]")
        if start < 0 or end <= start:
            raise ValueError(f"no JSON array in claude output: {text[:200]!r}")
        verdicts = json.loads(text[start : end + 1])
        out = []
        for v in verdicts:
            v = str(v).strip().lower()
            out.append(v if v in ("true", "false", "ood") else "false")
        return out

    # -------------------------------------------------------------- main method
    def get_feedback(
        self,
        collage_path_list,
        y_target_end_confidence_list,
        y_source_list,
        y_list,
        **kwargs,
    ):
        """Mirror of Human2ModelTeacher.get_feedback: one verdict per collage.

        Only collages whose counterfactual actually swapped the class
        (confidence >= 0.5) AND where the student was originally correct are shown
        to the LLM; the rest get the same sentinel labels the human GUI produces.
        """
        n = len(collage_path_list)
        student_correct = [y_source_list[i] == y_list[i] for i in range(len(y_list))]

        feedback = [None] * n
        pending = []
        for i in range(n):
            if i < len(student_correct) and not student_correct[i]:
                feedback[i] = "student incorrect!"
            elif y_target_end_confidence_list[i] < 0.5:
                feedback[i] = "student not swapped!"
            else:
                pending.append(i)

        n_batches = 0
        n_failed = 0
        first_exc = None
        for s in range(0, len(pending), self.batch_size):
            batch = pending[s : s + self.batch_size]
            paths = [collage_path_list[i] for i in batch]
            n_batches += 1
            try:
                verdicts = self._ask_claude(paths)
            except Exception as exc:  # a transient teacher failure must not abort a run
                n_failed += 1
                if first_exc is None:
                    first_exc = exc
                _log.info(
                    "%s",
                    f"[LLM2ModelTeacher] claude call failed ({exc}); batch -> 'false'",
                )
                verdicts = ["false"] * len(batch)
            for j, i in enumerate(batch):
                feedback[i] = verdicts[j] if j < len(verdicts) else "false"

        # Falling back to "false" is right for a flaky call, but if EVERY batch failed
        # the teacher never actually ran, and "everything is spurious" is indistinguishable
        # downstream from a real labelling: CFKD would finetune on all of it and report a
        # perfectly plausible `gain` computed from a dead teacher. Fail loudly instead.
        if n_batches > 0 and n_failed == n_batches:
            raise RuntimeError(
                f"LLM2ModelTeacher: all {n_batches} claude call(s) failed, so every "
                f"counterfactual would be labelled 'false' and the resulting gain would be "
                f"meaningless. Refusing to continue. First failure: {first_exc!r}. "
                f"Check that '{self.claude_bin}' runs headlessly in this environment "
                f"(e.g. `{self.claude_bin} -p --output-format json 'ok'`)."
            )

        return [f if f is not None else "false" for f in feedback]

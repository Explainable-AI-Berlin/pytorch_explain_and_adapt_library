"""
Job store and the single-GPU worker of the web demo.

A job is a directory under the jobs root:

    <root>/<job_id>/job.json        status, form fields, timings
    <root>/<job_id>/model.onnx      the (output-selected) uploaded classifier
    <root>/<job_id>/dataset/        imgs/{0,1} + data.csv
    <root>/<job_id>/config.yaml     DiDAE config (+ data.yaml, generator.yaml)
    <root>/<job_id>/feedback/       WebFeedbackTeacher request/response files
    <root>/<job_id>/log.txt         run_didae.py stdout/stderr
    <root>/<job_id>/run/            DiDAE base_dir
    <root>/<job_id>/results.json    written by peal.web.summarize

One worker thread runs jobs strictly one after another (one GPU, one job), so
the wait estimate is queue position x the median duration of recent jobs.
"""

import json
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import threading
import time
import uuid

from peal.web.cache import cache_path
from peal.web.paths import get_project_resource_dir

STATUSES = ("queued", "running", "awaiting_feedback", "finished", "failed", "cancelled")
ACTIVE = ("queued", "running", "awaiting_feedback")
DEFAULT_JOB_SECONDS = float(
    os.environ.get("PEAL_WEB_DEFAULT_JOB_SECONDS", str(45 * 60))
)
#: The adaptor prints a step marker per phase. It emits exactly five of them --
#: ``1``, ``2-6``, ``7``, ``8`` and ``9`` -- and ``2-6`` is one combined phase,
#: not six. The pattern therefore has to accept a range: it used to stop at the
#: first digit group, so ``Step 2-6:`` never matched and the progress line stayed
#: on "distilling" for the sweep, which is most of the job.
STEP_RE = re.compile(r"\[DiDAE\]\s*Step\s*(\d+(?:\.\d+)?(?:-\d+)?)[:\s]")
STAGES = {
    "1": "distilling the classifier into the CLIP / SAE space",
    "2-6": "sweeping dictionary directions, rendering counterfactuals and "
    "verifying the flips",
    "7": "building collages for the top directions",
    "8": "waiting for your verdicts",
    "9": "finetuning on the false directions",
    # Kept so a marker printed as a bare number still resolves if the adaptor
    # ever splits the combined phase again.
    "2": "sweeping dictionary directions in latent space",
    "3": "ranking directions by latent flips",
    "4": "rendering counterfactuals",
    "5": "verifying rendered flips on the classifier",
    "6": "ranking directions by verified flips",
}
#: Order of the ``step`` keys a job passes through, for the progress timeline.
STEP_ORDER = ("queued", "check", "starting", "1", "2-6", "7", "8", "9", "done")


def _now():
    return time.time()


class JobStore:
    """Directory-backed store of web-demo jobs, one ``job.json`` per job.

    Every read and write of a ``job.json`` goes through a re-entrant lock and
    writes are atomic (``.tmp`` then ``os.replace``), so the FastAPI handlers and
    the :class:`Worker` thread can share one store.

    Parameters
    ----------
    root : str
        Jobs root directory; created if missing. Job ids are 12 hex chars and
        each job lives in ``<root>/<job_id>/``.
    """

    def __init__(self, root):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self._lock = threading.RLock()

    # ------------------------------------------------------------ basics
    def job_dir(self, job_id):
        """Directory of a job, validating the id format.

        Parameters
        ----------
        job_id : str
            Twelve lowercase hex characters.

        Returns
        -------
        str
            ``<root>/<job_id>``.

        Raises
        ------
        KeyError
            If ``job_id`` is not a well-formed id (also for ``None``).
        """
        if not re.fullmatch(r"[a-f0-9]{12}", job_id or ""):
            raise KeyError(job_id)
        return os.path.join(self.root, job_id)

    def _path(self, job_id):
        return os.path.join(self.job_dir(job_id), "job.json")

    def create(self, fields):
        """Create a new job directory and its initial ``job.json``.

        Parameters
        ----------
        fields : dict
            The upload form fields, stored verbatim under ``"fields"``.

        Returns
        -------
        dict
            The job record with status ``"created"``, a fresh 12-hex-char
            ``id``, ``created`` timestamp and ``None`` for ``started``,
            ``finished``, ``duration``, ``stage`` and ``error``.
        """
        job_id = uuid.uuid4().hex[:12]
        d = os.path.join(self.root, job_id)
        os.makedirs(d)
        job = {
            "id": job_id,
            "status": "created",
            "created": _now(),
            "started": None,
            "finished": None,
            "duration": None,
            "stage": None,
            "step": None,
            "steps": {},
            "error": None,
            "fields": fields,
        }
        self._write(job)
        return job

    def _write(self, job):
        path = self._path(job["id"])
        tmp = path + ".tmp"
        with self._lock:
            with open(tmp, "w") as f:
                json.dump(job, f, indent=2)
            os.replace(tmp, path)

    def get(self, job_id):
        """Load a job record.

        Parameters
        ----------
        job_id : str
            Job id.

        Returns
        -------
        dict
            Contents of ``job.json``.

        Raises
        ------
        KeyError
            If the id is malformed or the job has no ``job.json``.
        """
        path = self._path(job_id)
        if not os.path.isfile(path):
            raise KeyError(job_id)
        with self._lock:
            with open(path) as f:
                return json.load(f)

    def update(self, job_id, **changes):
        """Merge ``changes`` into a job record and write it back atomically.

        Parameters
        ----------
        job_id : str
            Job id.
        **changes
            Keys to set on the record (``status``, ``stage``, ``error``, ...).

        Returns
        -------
        dict
            The updated record.
        """
        with self._lock:
            job = self.get(job_id)
            job.update(changes)
            self._write(job)
            return job

    def list(self):
        """All readable job records under the root, oldest first.

        Directories whose name is not a job id, or whose ``job.json`` is
        missing or corrupt, are skipped.

        Returns
        -------
        list of dict
            Records sorted by their ``created`` timestamp.
        """
        jobs = []
        for name in os.listdir(self.root):
            if re.fullmatch(r"[a-f0-9]{12}", name):
                try:
                    jobs.append(self.get(name))
                except (KeyError, json.JSONDecodeError):
                    continue
        return sorted(jobs, key=lambda j: j["created"])

    def delete(self, job_id):
        """Remove a job's whole directory (uploads, run dir, results)."""
        d = self.job_dir(job_id)
        shutil.rmtree(d, ignore_errors=True)

    # ------------------------------------------------------------- queue
    def queue(self):
        """Active jobs (queued, running or awaiting feedback), oldest first.

        Returns
        -------
        list of dict
        """
        return [j for j in self.list() if j["status"] in ACTIVE]

    def next_queued(self):
        """The oldest job with status ``"queued"``, or ``None``.

        Returns
        -------
        dict or None
        """
        for j in self.list():
            if j["status"] == "queued":
                return j
        return None

    def median_duration(self, n=10):
        """Median wall-clock duration of the last ``n`` finished jobs.

        Parameters
        ----------
        n : int
            How many of the most recent finished jobs to consider.

        Returns
        -------
        float
            Seconds; ``DEFAULT_JOB_SECONDS`` (env ``PEAL_WEB_DEFAULT_JOB_SECONDS``,
            45 minutes by default) when no job has finished yet.
        """
        done = [
            j["duration"]
            for j in self.list()
            if j["status"] == "finished" and j["duration"]
        ]
        done = done[-n:]
        return statistics.median(done) if done else DEFAULT_JOB_SECONDS

    def queue_position(self, job_id):
        """1 = running now, 2 = next, ...; None when not active."""
        for pos, j in enumerate(self.queue(), start=1):
            if j["id"] == job_id:
                return pos
        return None

    def estimate_wait_seconds(self, job_id=None):
        """Seconds until the job starts (or, for a new upload, until it would
        start): the running job's remaining time plus one median per job ahead.
        A job waiting for verdicts counts as running: it holds the GPU."""
        median = self.median_duration()
        queue = self.queue()
        ahead = []
        for j in queue:
            if job_id is not None and j["id"] == job_id:
                break
            ahead.append(j)
        total = 0.0
        for j in ahead:
            if j["status"] in ("running", "awaiting_feedback") and j["started"]:
                total += max(60.0, median - (_now() - j["started"]))
            else:
                total += median
        return total

    def stats(self):
        """Queue summary shown on the landing page.

        Returns
        -------
        dict
            ``queued`` and ``running`` counts (a job awaiting feedback counts
            as running), ``median_job_seconds`` and
            ``estimated_wait_seconds_for_new_job``.
        """
        q = self.queue()
        return {
            "queued": sum(1 for j in q if j["status"] == "queued"),
            "running": sum(
                1 for j in q if j["status"] in ("running", "awaiting_feedback")
            ),
            "median_job_seconds": self.median_duration(),
            "estimated_wait_seconds_for_new_job": self.estimate_wait_seconds(None),
            # No ids here: a job id is the only key to its page and results.
            "jobs": [
                {
                    "position": pos,
                    "status": j["status"],
                    "stage": j.get("stage"),
                    "created": j.get("created"),
                    "started": j.get("started"),
                }
                for pos, j in enumerate(q, start=1)
            ],
        }

    def mark_step(self, job_id, step, **changes):
        """Update a job and record when it first reached ``step``.

        Parameters
        ----------
        job_id : str
            Job id.
        step : str
            One of :data:`STEP_ORDER`.
        **changes
            Further keys to set, as in :meth:`update`.

        Returns
        -------
        dict
            The updated record.
        """
        with self._lock:
            job = self.get(job_id)
            steps = dict(job.get("steps") or {})
            steps.setdefault(step, _now())
            job.update(changes, step=step, steps=steps)
            self._write(job)
            return job


class Worker(threading.Thread):
    """Runs queued jobs one at a time as `python run_didae.py --config ...`.

    A daemon thread that polls the store, launches the next queued job as a
    subprocess with stdout/stderr appended to ``<job_dir>/log.txt``, mirrors
    the ``[DiDAE] Step N`` markers of that log into the job's ``stage``,
    honours ``cancel_requested`` and finally runs ``peal.web.summarize``.

    Parameters
    ----------
    store : JobStore
        The store to poll and update.
    python : str, optional
        Interpreter for the subprocess; defaults to ``$PEAL_WEB_PYTHON`` or
        ``sys.executable``.
    poll_interval : float
        Seconds between store polls and between log checks of a running job.
    env : dict, optional
        Extra environment variables for the subprocess, applied after
        ``PEAL_RUNS`` (default ``<root>/_peal_runs``) and ``PYTHONUNBUFFERED``.

    Attributes
    ----------
    current : subprocess.Popen or None
        The running job's process, for cancellation from outside.
    """

    def __init__(self, store, python=None, poll_interval=3.0, env=None):
        super().__init__(daemon=True, name="peal-web-worker")
        self.store = store
        self.python = python or os.environ.get("PEAL_WEB_PYTHON") or sys.executable
        self.poll_interval = poll_interval
        self.env = env
        self._stop = threading.Event()
        self.current = None

    def stop(self):
        """Ask the polling loop to exit after the current job."""
        self._stop.set()

    def run(self):
        """Thread body: run queued jobs until :meth:`stop` is called.

        Any exception escaping :meth:`run_job` marks that job ``failed`` with
        the exception text and the loop continues.
        """
        while not self._stop.is_set():
            job = self.store.next_queued()
            if job is None:
                self._stop.wait(self.poll_interval)
                continue
            try:
                self.run_job(job["id"])
            except Exception as exc:  # the worker must survive any job
                self.store.update(
                    job["id"],
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                    finished=_now(),
                )

    def check_accuracy(self, job_dir, log_path, env=None, timeout_s=1800):
        """Run ``python -m peal.web.analysis --accuracy-only`` and wait for it.

        Writes ``accuracy.json``; failures are appended to ``log_path`` and do
        not stop the job.
        """
        if os.path.isfile(os.path.join(job_dir, "accuracy.json")):
            return
        with open(log_path, "ab") as log:
            try:
                subprocess.run(
                    [
                        self.python,
                        "-W",
                        "ignore",
                        "-m",
                        "peal.web.analysis",
                        "--accuracy-only",
                        job_dir,
                    ],
                    cwd=get_project_resource_dir(),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    timeout=timeout_s,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                log.write(f"[web] accuracy check failed: {exc}\n".encode())

    def run_dfr(self, job_dir, solver, log_path, env=None, timeout_s=3600):
        """Run ``python -m peal.web.dfr --solver=<solver> <job_dir>`` and wait.

        Writes ``dfr.json`` and the corrected ``run/model.onnx``; a failure is
        logged and leaves the uncorrected model in place.
        """
        with open(log_path, "ab") as log:
            log.write(f"[web] DFR ({solver}) on the judged directions\n".encode())
            log.flush()
            try:
                subprocess.run(
                    [
                        self.python,
                        "-W",
                        "ignore",
                        "-m",
                        "peal.web.dfr",
                        f"--solver={solver}",
                        job_dir,
                    ],
                    cwd=get_project_resource_dir(),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    timeout=timeout_s,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                log.write(f"[web] DFR failed: {exc}\n".encode())

    def start_analysis(self, job_dir, env=None):
        """Launch ``python -m peal.web.analysis <job_dir>`` in the background.

        Writes ``analysis.log``; the page polls for ``analysis.json``. Does
        nothing when the analysis already ran or is running, or when the sweep
        has not written its results yet.

        Parameters
        ----------
        job_dir : str
            Job directory.
        env : dict, optional
            Environment of the subprocess.

        Returns
        -------
        subprocess.Popen or None
            The started process.
        """
        marker = os.path.join(job_dir, "analysis.log")
        sweep = os.path.join(job_dir, "run", "sweep_results.pt")
        if os.path.exists(marker) or not os.path.isfile(sweep):
            return None
        log = open(marker, "ab")
        if os.path.isfile(cache_path(job_dir)):
            # With the image cache the analysis is CPU work (dictionary codes of
            # stored CLIP embeddings). Keep it off the GPU: during step 8 the
            # waiting DiDAE process still holds most of it, and a CUDA context
            # alone then fails with out-of-memory.
            env = dict(env if env is not None else os.environ)
            env["CUDA_VISIBLE_DEVICES"] = ""
        return subprocess.Popen(
            [self.python, "-W", "ignore", "-m", "peal.web.analysis", job_dir],
            cwd=get_project_resource_dir(),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )

    def command(self, job_dir):
        """The argv used to launch a job.

        Parameters
        ----------
        job_dir : str
            Job directory holding ``config.yaml``.

        Returns
        -------
        list of str
            ``[python, <project>/run_didae.py, "--config", <job_dir>/config.yaml]``.
        """
        return [
            self.python,
            os.path.join(get_project_resource_dir(), "run_didae.py"),
            "--config",
            os.path.join(job_dir, "config.yaml"),
        ]

    def run_job(self, job_id):
        """Run one job to completion and record its outcome.

        Marks the job ``running``, launches :meth:`command` from the project
        directory with the log appended to ``log.txt``, monitors it until it
        exits or is cancelled, then calls ``peal.web.summarize.summarize`` on
        the job directory (failures are logged, not raised) and sets the final
        status: ``finished`` on exit code 0, ``failed`` otherwise, or just the
        ``finished`` timestamp when the job was cancelled.

        Parameters
        ----------
        job_id : str
            Job id.
        """
        job_dir = self.store.job_dir(job_id)
        started = _now()
        env = dict(os.environ)
        env.setdefault("PEAL_RUNS", os.path.join(self.store.root, "_peal_runs"))
        env.setdefault("PYTHONUNBUFFERED", "1")
        if self.env:
            env.update(self.env)
        log_path = os.path.join(job_dir, "log.txt")
        # First the classifier's plain accuracy on the uploaded images, so a
        # wrong normalization / input size / output order is visible at once.
        self.store.mark_step(
            job_id,
            "check",
            status="running",
            started=started,
            stage="checking your classifier on the uploaded images",
        )
        self.check_accuracy(job_dir, log_path, env)
        self.store.mark_step(job_id, "starting", stage="starting")
        with open(log_path, "ab") as log:
            log.write(
                (
                    " ".join(shlex.quote(c) for c in self.command(job_dir)) + "\n"
                ).encode()
            )
            proc = subprocess.Popen(
                self.command(job_dir),
                cwd=get_project_resource_dir(),
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
            )
            self.current = proc
            try:
                self._monitor(job_id, job_dir, proc, log_path, env)
            finally:
                self.current = None
        code = proc.returncode
        job = self.store.get(job_id)
        if job["status"] == "cancelled":
            self.store.update(job_id, finished=_now())
            return
        fields = job.get("fields") or {}
        if code == 0 and fields.get("correction") == "dfr":
            self.store.mark_step(
                job_id, "9", stage="step 9: refitting the last layer (DFR)"
            )
            self.run_dfr(job_dir, fields.get("dfr_solver") or "logistic", log_path, env)
            # redo the analysis so it includes the corrected model
            for name in ("analysis.json", "analysis.log"):
                path = os.path.join(job_dir, name)
                if os.path.exists(path):
                    os.remove(path)
        analysis = self.start_analysis(job_dir, env)
        if analysis is not None:
            self.store.mark_step(
                job_id,
                "analysis",
                stage="evaluating the classifier per direction (group accuracies)",
            )
            analysis.wait()
        try:
            from peal.web.summarize import summarize

            summarize(job_dir)
        except Exception as exc:
            with open(log_path, "a") as f:
                f.write(f"[web] summarize failed: {type(exc).__name__}: {exc}\n")
        finished = _now()
        if code == 0:
            self.store.mark_step(
                job_id,
                "done",
                status="finished",
                finished=finished,
                duration=finished - started,
                stage="done",
            )
        else:
            self.store.update(
                job_id,
                status="failed",
                finished=finished,
                duration=finished - started,
                error=f"run_didae.py exited with code {code}; see log",
            )

    def _monitor(self, job_id, job_dir, proc, log_path, env=None):
        """Poll a running job: handle cancel requests, update status/stage."""
        feedback_dir = os.path.join(job_dir, "feedback")
        last_stage = None
        while proc.poll() is None:
            job = self.store.get(job_id)
            if job.get("cancel_requested") and job["status"] != "cancelled":
                proc.terminate()
                try:
                    proc.wait(15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                self.store.update(job_id, status="cancelled", stage="cancelled")
                break
            stage = self._stage_from_log(log_path) or last_stage
            pending = pending_feedback_round(feedback_dir)
            if pending is not None:
                status, step, stage_text = "awaiting_feedback", "8", STAGES["8"]
                # The feedback page shows the per-direction group accuracies.
                self.start_analysis(job_dir, env)
            else:
                status, step = "running", stage or "starting"
                stage_text = STAGES.get(stage, f"step {stage}") if stage else "starting"
            if job["status"] != status or job.get("stage") != stage_text:
                self.store.mark_step(job_id, step, status=status, stage=stage_text)
            last_stage = stage
            time.sleep(self.poll_interval)

    @staticmethod
    def _stage_from_log(log_path, tail_bytes=200_000):
        """Last ``[DiDAE] Step N`` number in the log tail, or ``None``."""
        try:
            with open(log_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - tail_bytes))
                text = f.read().decode(errors="replace")
        except OSError:
            return None
        steps = STEP_RE.findall(text)
        return steps[-1] if steps else None


def latest_activity(log_path, tail_bytes=16_000, max_chars=240):
    """The last meaningful line of a job log, for the "currently" display.

    tqdm redraws its bar with carriage returns, so only the last segment of a
    line is kept. The adaptor's long ``Loading config from ...`` dumps and the
    worker's own command line are skipped.

    Parameters
    ----------
    log_path : str
        Path of the job's ``log.txt``.
    tail_bytes : int
        How much of the end of the log to look at.
    max_chars : int
        Longer lines are cut to this length.

    Returns
    -------
    str or None
        The line, or ``None`` when the log is missing or has nothing to show.
    """
    try:
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - tail_bytes))
            text = f.read().decode(errors="replace")
    except OSError:
        return None
    for line in reversed(text.split("\n")):
        segments = [seg.strip() for seg in line.split("\r") if seg.strip()]
        if not segments:
            continue
        last = segments[-1]
        if last.startswith(("Loading config from", "No config model", "Config model")):
            continue
        if "run_didae.py --config" in last:
            continue
        return last if len(last) <= max_chars else last[: max_chars - 1] + "…"
    return None


def pending_feedback_round(feedback_dir):
    """The oldest request round without a response, or None."""
    if not os.path.isdir(feedback_dir):
        return None
    rounds = []
    for name in os.listdir(feedback_dir):
        m = re.fullmatch(r"feedback_request_(\d+)\.json", name)
        if m and not os.path.isfile(
            os.path.join(feedback_dir, f"feedback_response_{m.group(1)}.json")
        ):
            rounds.append(int(m.group(1)))
    return min(rounds) if rounds else None

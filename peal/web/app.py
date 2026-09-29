"""
HTTP API of the PEAL web demo (FastAPI).

    uvicorn peal.web.app:app --host 0.0.0.0 --port 8080

Flow:  POST /api/uploads/model   (ONNX)   -> interface + upload token
       POST /api/uploads/dataset (zip)    -> class folders + upload token
       POST /api/jobs                     -> job id (validates the mandatory fields)
       GET  /api/jobs/{id}                -> status, queue position, wait estimate
       GET  /api/jobs/{id}/feedback       -> collages waiting for a verdict
       POST /api/jobs/{id}/feedback       -> the verdicts
       GET  /api/jobs/{id}/results        -> results.json
       GET  /api/jobs/{id}/files/{path}   -> collages, corrected model
       POST /api/jobs/{id}/cancel

Environment:
    PEAL_WEB_JOBS      jobs root (default $PEAL_RUNS/web_jobs or ./web_jobs)
    PEAL_RAE_WEIGHTS   hf://<org>/<repo> or a local weights folder (required to run)
    PEAL_WEB_NO_WORKER=1  do not start the worker thread (tests, separate worker host)
    PEAL_WEB_MAX_ZIP_MB / PEAL_WEB_MAX_FILES / PEAL_WEB_MAX_ONNX_MB  upload limits
"""

import json
import os
import re
import shutil
import time
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from peal.architectures.onnx_predictor import inspect_onnx, select_onnx_outputs
from peal.web import ingest
from peal.web.config_builder import NORMALIZATION_PRESETS, build_job_configs
from peal.web.jobs import (
    STAGES,
    JobStore,
    Worker,
    latest_activity,
    pending_feedback_round,
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
MAX_ONNX_BYTES = int(
    float(os.environ.get("PEAL_WEB_MAX_ONNX_MB", "2048")) * 1024 * 1024
)
MAX_ZIP_BYTES = ingest.DEFAULT_MAX_BYTES
#: Upper bound of the form's "directions to render": each direction renders
#: every one of its latent flips (up to the whole sample pool).
MAX_RENDER_DIRECTIONS = int(os.environ.get("PEAL_WEB_MAX_RENDER_DIRECTIONS", "30"))
UPLOAD_TTL_S = 6 * 3600


def _jobs_root():
    """Jobs root: PEAL_WEB_JOBS, else $PEAL_RUNS/web_jobs, else ./web_jobs."""
    root = os.environ.get("PEAL_WEB_JOBS")
    if root:
        return root
    runs = os.environ.get("PEAL_RUNS")
    return os.path.join(runs, "web_jobs") if runs else os.path.abspath("web_jobs")


#: DiDAE's collage names: ``rank<r>_dir<d>_pair<p>_conf<c>.png`` for the pair,
#: and a folder ``rank<r>_dir<d>_<concept name>`` next to it holding the
#: detailed collage ``<p:07d>_collage.png`` (factual, counterfactual, SSIM
#: difference); see DiDAE._generate_direction_collages.
PAIR_COLLAGE_RE = re.compile(r"^rank(\d+)_dir(\d+)_pair(\d+)_")


def pretty_direction_name(folder_suffix):
    """``OFF_downing_SAE_3303_TO_ON_dogs_SAE_853`` -> ``OFF downing #3303 → ON dogs #853``."""
    name = re.sub(r"_?SAE_(\d+)", r" #\1", folder_suffix)
    name = name.replace("_TO_", " → ").replace("_", " ")
    return re.sub(r"\s+", " ", name).strip()


def group_by_direction(items):
    """Group feedback items by the direction their collage belongs to.

    Parameters
    ----------
    items : list of dict
        Items of a feedback request, each with its ``collage`` path.

    Returns
    -------
    list of dict
        One entry per direction in rank order, with ``rank``,
        ``direction_idx``, ``name`` (the concept edit, when the folder is
        found) and ``items``; each item gains ``detail_collage`` when its
        detailed collage exists. Items whose file name does not follow the
        pattern form one group each.
    """
    groups = {}
    for it in items:
        base = os.path.basename(it["collage"])
        m = PAIR_COLLAGE_RE.match(base)
        if not m:
            groups[("item", it["index"])] = {
                "rank": None,
                "direction_idx": None,
                "name": base,
                "items": [it],
            }
            continue
        rank, d_idx, pair = (int(v) for v in m.groups())
        key = (rank, d_idx)
        if key not in groups:
            prefix = f"rank{rank:03d}_dir{d_idx}_"
            parent = os.path.dirname(it["collage"])
            folder = next(
                (
                    f
                    for f in sorted(os.listdir(parent))
                    if f.startswith(prefix) and os.path.isdir(os.path.join(parent, f))
                ),
                None,
            )
            groups[key] = {
                "rank": rank,
                "direction_idx": d_idx,
                "name": (
                    pretty_direction_name(folder[len(prefix) :])
                    if folder
                    else f"direction {d_idx}"
                ),
                "folder": folder and os.path.join(parent, folder),
                "items": [],
            }
        g = groups[key]
        if g.get("folder"):
            detail = os.path.join(g["folder"], f"{pair:07d}_collage.png")
            if os.path.isfile(detail):
                it = {**it, "detail_collage": detail}
        g["items"].append(it)
    out = sorted(
        groups.values(),
        key=lambda g: (g["rank"] is None, g["rank"] or 0, g["items"][0]["index"]),
    )
    for g in out:
        g.pop("folder", None)
    return out


def create_app(jobs_root=None, start_worker=None):
    """
    Build the FastAPI application of the web demo.

    Creates the ``JobStore`` under ``jobs_root`` (plus an ``_uploads``
    subfolder for not-yet-submitted uploads), optionally starts the
    background ``Worker`` thread that executes queued jobs, registers all
    routes listed in the module docstring and mounts ``/static``.

    Parameters
    ----------
    jobs_root : str, optional
        Directory holding job directories. Defaults to ``_jobs_root()``.
    start_worker : bool, optional
        Whether to start the worker thread. Defaults to ``True`` unless
        ``PEAL_WEB_NO_WORKER=1`` is set.

    Returns
    -------
    fastapi.FastAPI
        The application; ``app.state`` carries ``store``, ``uploads_root``
        and ``worker`` (``None`` when no worker was started).
    """
    app = FastAPI(title="PEAL Clever Hans analysis", version="0.1")
    store = JobStore(jobs_root or _jobs_root())
    uploads_root = os.path.join(store.root, "_uploads")
    os.makedirs(uploads_root, exist_ok=True)
    app.state.store = store
    app.state.uploads_root = uploads_root
    if start_worker is None:
        start_worker = os.environ.get("PEAL_WEB_NO_WORKER", "0") != "1"
    app.state.worker = None
    if start_worker:
        app.state.worker = Worker(store)
        app.state.worker.start()

    # ------------------------------------------------------------ helpers
    def upload_dir(token, must_exist=True):
        """Map a 32-hex upload token to its directory (400 bad token, 404 missing)."""
        if (
            not token
            or not all(c in "0123456789abcdef" for c in token)
            or len(token) != 32
        ):
            raise HTTPException(400, "invalid upload token")
        d = os.path.join(uploads_root, token)
        if must_exist and not os.path.isdir(d):
            raise HTTPException(404, "upload expired or unknown")
        return d

    async def save_upload(file, target, limit):
        """Stream an upload to ``target`` in 1 MB chunks; 413 above ``limit``."""
        size = 0
        with open(target, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > limit:
                    out.close()
                    os.remove(target)
                    raise HTTPException(
                        413, f"file larger than {limit // (1024 * 1024)} MB"
                    )
                out.write(chunk)
        return size

    def sweep_uploads():
        """Delete upload directories older than ``UPLOAD_TTL_S`` (6 hours)."""
        now = time.time()
        for name in os.listdir(uploads_root):
            d = os.path.join(uploads_root, name)
            try:
                if now - os.path.getmtime(d) > UPLOAD_TTL_S:
                    shutil.rmtree(d, ignore_errors=True)
            except OSError:
                pass

    def get_job(job_id):
        """Return the stored job dict or raise a 404."""
        try:
            return store.get(job_id)
        except KeyError:
            raise HTTPException(404, "unknown job") from None

    def public_job(job):
        """
        Reduce a stored job to the fields exposed by the API.

        Strips custom ``mean``/``std`` from ``fields`` and adds
        ``queue_position``, ``estimated_wait_seconds``, ``median_job_seconds``,
        ``feedback_pending``, ``has_results`` and, while running, ``activity``
        (the last line of the log).
        """
        job_id = job["id"]
        out = {
            k: job.get(k)
            for k in (
                "id",
                "status",
                "created",
                "started",
                "finished",
                "duration",
                "stage",
                "step",
                "steps",
                "error",
            )
        }
        if not out["step"] and job.get("stage"):
            # Jobs recorded before steps were tracked: recover it from the text.
            by_text = {text: key for key, text in STAGES.items()}
            out["step"] = by_text.get(job["stage"], job["stage"])
        out["fields"] = {
            k: v for k, v in job.get("fields", {}).items() if k not in ("mean", "std")
        }
        out["queue_position"] = store.queue_position(job_id)
        out["estimated_wait_seconds"] = (
            store.estimate_wait_seconds(job_id) if job["status"] == "queued" else 0.0
        )
        out["median_job_seconds"] = store.median_duration()
        out["feedback_pending"] = (
            pending_feedback_round(os.path.join(store.job_dir(job_id), "feedback"))
            is not None
        )
        out["has_results"] = os.path.isfile(
            os.path.join(store.job_dir(job_id), "results.json")
        )
        acc_path = os.path.join(store.job_dir(job_id), "accuracy.json")
        out["accuracy_check"] = None
        if os.path.isfile(acc_path):
            with open(acc_path) as f:
                out["accuracy_check"] = json.load(f)
        out["activity"] = (
            latest_activity(os.path.join(store.job_dir(job_id), "log.txt"))
            if job["status"] == "running"
            else None
        )
        return out

    # -------------------------------------------------------------- pages
    @app.get("/", response_class=HTMLResponse)
    def index():
        """Serve the single-page frontend ``static/index.html``."""
        with open(os.path.join(STATIC_DIR, "index.html")) as f:
            return f.read()

    @app.get("/api/queue")
    def queue():
        """Return queue statistics of the job store."""
        return store.stats()

    @app.get("/api/options")
    def options():
        """Return form options: normalization presets, upload limits, RAE weights."""
        return {
            "normalizations": sorted(NORMALIZATION_PRESETS) + ["custom"],
            "max_zip_mb": MAX_ZIP_BYTES // (1024 * 1024),
            "max_onnx_mb": MAX_ONNX_BYTES // (1024 * 1024),
            "max_render_directions": MAX_RENDER_DIRECTIONS,
            "rae_weights": os.environ.get("PEAL_RAE_WEIGHTS"),
        }

    # ------------------------------------------------------------ uploads
    @app.post("/api/uploads/model")
    async def upload_model(file: UploadFile = File(...)):
        """
        Store an ONNX classifier and return its interface plus an upload token.

        The model must accept ``[batch, 3, H, W]``; the ``inspect_onnx``
        result is written to ``info.json`` next to ``model.onnx``.
        """
        sweep_uploads()
        token = uuid.uuid4().hex
        d = upload_dir(token, must_exist=False)
        os.makedirs(d)
        path = os.path.join(d, "model.onnx")
        await save_upload(file, path, MAX_ONNX_BYTES)
        try:
            info = inspect_onnx(path)
        except Exception as exc:
            shutil.rmtree(d, ignore_errors=True)
            raise HTTPException(400, f"not a usable ONNX classifier: {exc}") from exc
        shape = info["input_shape"]
        if len(shape) != 4 or (shape[1] not in (3, None)):
            shutil.rmtree(d, ignore_errors=True)
            raise HTTPException(
                400,
                f"expected an image classifier with input [batch, 3, H, W], got {shape}",
            )
        # Probe convertibility now rather than at the end of the job. An
        # unconvertible graph still works for explanation and ranking, but it cannot
        # be finetuned, and the template asks for finetuning; discovering that after
        # a full sweep and all the human judging wastes the whole run.
        finetunable, convert_error = True, None
        try:
            from peal.architectures.onnx_predictor import load_onnx_as_torch

            load_onnx_as_torch(path, device="cpu", check=False)
        except Exception as exc:  # ImportError, RuntimeError, NotImplementedError
            finetunable = False
            convert_error = f"{type(exc).__name__}: {exc}"
        info["filename"] = os.path.basename(file.filename or "model.onnx")
        info["finetunable"] = finetunable
        # DFR (refitting only the final linear layer) needs that layer located in
        # the graph; the form offers it only then.
        try:
            from peal.architectures.onnx_predictor import find_final_linear

            layer = find_final_linear(path)
        except Exception:
            layer = None
        info["dfr_available"] = layer is not None
        if layer is not None:
            info["final_layer"] = {
                k: layer[k] for k in ("node", "op", "n_features", "n_classes")
            }
        if convert_error:
            info["convert_error"] = convert_error
        with open(os.path.join(d, "info.json"), "w") as f:
            json.dump(info, f)
        return {"token": token, "filename": file.filename, **info}

    @app.post("/api/uploads/dataset")
    async def upload_dataset(file: UploadFile = File(...)):
        """
        Unpack a zip of class folders and return the class counts plus a token.

        The zip is removed after ingestion; ``classes.json`` records the
        class root and the file list per class for ``create_job``.
        """
        sweep_uploads()
        token = uuid.uuid4().hex
        d = upload_dir(token, must_exist=False)
        os.makedirs(d)
        zip_path = os.path.join(d, "dataset.zip")
        await save_upload(file, zip_path, MAX_ZIP_BYTES)
        try:
            class_root, classes = ingest.ingest_zip(zip_path, d)
        except ingest.IngestError as exc:
            shutil.rmtree(d, ignore_errors=True)
            raise HTTPException(400, str(exc)) from exc
        os.remove(zip_path)
        # The run splits the data 80/10/10 and fits a covariance of DINO
        # features on the validation part; with a handful of images that part
        # holds one image and the job dies deep inside DiDAE (NaN covariance).
        min_images = int(os.environ.get("PEAL_WEB_MIN_IMAGES", "50"))
        n_images = sum(len(v) for v in classes.values())
        if n_images < min_images:
            shutil.rmtree(d, ignore_errors=True)
            raise HTTPException(
                400,
                f"the dataset holds {n_images} images; at least {min_images} "
                "are needed (10% of them form the validation split)",
            )
        with open(os.path.join(d, "classes.json"), "w") as f:
            json.dump({"class_root": class_root, "classes": classes}, f)
        return {
            "token": token,
            "filename": file.filename,
            "classes": {k: len(v) for k, v in classes.items()},
        }

    # --------------------------------------------------------------- jobs
    @app.post("/api/jobs")
    def create_job(
        model_token: str = Form(...),
        dataset_token: str = Form(...),
        class_a: str = Form(...),
        class_b: str = Form(...),
        class_a_name: str = Form(...),
        class_b_name: str = Form(...),
        output_index_a: int = Form(...),
        output_index_b: int = Form(...),
        normalization: str = Form(...),
        input_height: int = Form(...),
        input_width: int = Form(...),
        mean: str = Form(""),
        std: str = Form(""),
        accept_terms: bool = Form(False),
        max_per_class: int = Form(0),
        render_directions: int = Form(10),
        render_per_direction: int = Form(30),
        direction_type: str = Form("pairs"),
        correction: str = Form("none"),
        dfr_solver: str = Form("logistic"),
    ):
        """
        Validate the form, materialize the job directory and enqueue the job.

        Checks class choice, class names, output indices against the ONNX
        output count, input size against the model's declared shape and the
        normalization (preset or custom ``mean``/``std``). On success the two
        selected ONNX outputs are exported to ``<job>/model.onnx``, the pair
        dataset is copied to ``<job>/dataset``, the PEAL configs are built
        and the upload directories are deleted. Failures during preparation
        mark the job ``failed`` and raise a 400.
        """
        if not accept_terms:
            raise HTTPException(
                400, "you must accept the terms (non-commercial demo, data retention)"
            )
        md = upload_dir(model_token)
        dd = upload_dir(dataset_token)
        with open(os.path.join(md, "info.json")) as f:
            info = json.load(f)
        with open(os.path.join(dd, "classes.json")) as f:
            ds = json.load(f)
        classes = ds["classes"]
        # mandatory field validation
        errors = []
        if class_a not in classes or class_b not in classes:
            errors.append("class folders must be chosen from the uploaded dataset")
        if class_a == class_b:
            errors.append("the two classes must differ")
        if not class_a_name.strip() or not class_b_name.strip():
            errors.append("both class names are required")
        n_out = info.get("num_outputs")
        for idx in (output_index_a, output_index_b):
            if idx < 0 or (n_out is not None and idx >= n_out):
                errors.append(
                    f"output index {idx} is outside the model's {n_out} outputs"
                )
        if output_index_a == output_index_b:
            errors.append("the two output indices must differ")
        shape = info["input_shape"]
        if shape[2] is not None and shape[2] != input_height:
            errors.append(
                f"the model declares input height {shape[2]}, not {input_height}"
            )
        if shape[3] is not None and shape[3] != input_width:
            errors.append(
                f"the model declares input width {shape[3]}, not {input_width}"
            )
        if (
            input_height < 32
            or input_width < 32
            or input_height > 1024
            or input_width > 1024
        ):
            errors.append("input size must be between 32 and 1024 pixels")
        fields = {
            "class_a": class_a,
            "class_b": class_b,
            "class_a_name": class_a_name.strip(),
            "class_b_name": class_b_name.strip(),
            "output_index_a": output_index_a,
            "output_index_b": output_index_b,
            "normalization": normalization,
            "input_height": input_height,
            "input_width": input_width,
            "model_filename": os.path.basename(info.get("filename", "model.onnx")),
            "render_directions": render_directions,
            "render_per_direction": render_per_direction,
            "direction_type": direction_type,
            "correction": correction,
            "dfr_solver": dfr_solver,
        }
        if normalization == "custom":
            try:
                fields["mean"] = [float(v) for v in mean.split(",")]
                fields["std"] = [float(v) for v in std.split(",")]
            except ValueError:
                errors.append(
                    "custom mean/std must be three comma-separated numbers each"
                )
        elif normalization not in NORMALIZATION_PRESETS:
            errors.append(f"unknown normalization {normalization!r}")
        # The form's min="0" is client-side only. A negative cap reached
        # build_pair_dataset as files[:negative], which silently dropped images from
        # the end of each class instead of failing.
        if max_per_class < 0:
            errors.append(f"max_per_class must be >= 0, got {max_per_class}")
        if not 1 <= render_directions <= MAX_RENDER_DIRECTIONS:
            errors.append(
                f"directions to render must be between 1 and {MAX_RENDER_DIRECTIONS}, "
                f"got {render_directions}"
            )
        if correction not in ("none", "dfr", "finetune"):
            errors.append(
                f"correction must be none, dfr or finetune, got {correction!r}"
            )
        if correction == "dfr" and not info.get("dfr_available"):
            errors.append(
                "DFR needs the model's final linear layer, which was not found in "
                "this ONNX graph; choose no correction or full finetuning"
            )
        if correction == "finetune" and not info.get("finetunable", True):
            errors.append("this ONNX graph cannot be converted for finetuning")
        if dfr_solver not in ("logistic", "svm", "ridge"):
            errors.append(
                f"DFR solver must be logistic, svm or ridge, got {dfr_solver!r}"
            )
        if direction_type not in ("single", "pairs"):
            errors.append(
                f"direction type must be single or pairs, got {direction_type!r}"
            )
        if render_per_direction < 0:
            errors.append(
                f"renders per direction must be >= 0 (0 = all), got {render_per_direction}"
            )
        if errors:
            raise HTTPException(400, "; ".join(errors))

        job = store.create(fields)
        job_dir = store.job_dir(job["id"])
        try:
            select_onnx_outputs(
                os.path.join(md, "model.onnx"),
                os.path.join(job_dir, "model.onnx"),
                [output_index_a, output_index_b],
            )
            counts = ingest.build_pair_dataset(
                ds["class_root"],
                classes,
                class_a,
                class_b,
                os.path.join(job_dir, "dataset"),
                max_per_class=max_per_class or None,
                link=False,
            )
            n_samples = sum(counts.values())
            build_job_configs(job_dir, fields, n_samples)
        except Exception as exc:
            store.update(
                job["id"],
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                finished=time.time(),
            )
            raise HTTPException(400, f"could not prepare the job: {exc}") from exc
        shutil.rmtree(md, ignore_errors=True)
        shutil.rmtree(dd, ignore_errors=True)
        job = store.mark_step(
            job["id"],
            "queued",
            status="queued",
            n_samples=n_samples,
            class_counts=counts,
        )
        return public_job(job)

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        """Return the public view of a job."""
        return public_job(get_job(job_id))

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        """Cancel a queued job or flag a running one with ``cancel_requested``."""
        job = get_job(job_id)
        if job["status"] == "queued":
            job = store.update(job_id, status="cancelled", finished=time.time())
        elif job["status"] in ("running", "awaiting_feedback"):
            job = store.update(job_id, cancel_requested=True)
        return public_job(job)

    @app.get("/api/jobs/{job_id}/log")
    def job_log(job_id: str, tail: int = 20000):
        """Return the last ``tail`` bytes of the job's ``log.txt``."""
        get_job(job_id)
        path = os.path.join(store.job_dir(job_id), "log.txt")
        if not os.path.isfile(path):
            return {"log": ""}
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - tail))
            return {"log": f.read().decode(errors="replace")}

    # ------------------------------------------------------------ feedback
    @app.get("/api/jobs/{job_id}/feedback")
    def feedback_request(job_id: str):
        """
        Return the collages of the pending feedback round, if any.

        Reads ``feedback/feedback_request_<round>.json`` written by the
        worker's ``WebFeedbackTeacher`` and adds a ``collage_url`` per item,
        plus ``directions``: the items grouped by the direction they belong to
        (see :func:`group_by_direction`), so the page can ask for one verdict
        per direction.
        """
        get_job(job_id)
        fdir = os.path.join(store.job_dir(job_id), "feedback")
        rnd = pending_feedback_round(fdir)
        if rnd is None:
            return {"round": None, "items": []}
        with open(os.path.join(fdir, f"feedback_request_{rnd}.json")) as f:
            req = json.load(f)
        job_dir = store.job_dir(job_id)

        def url(path):
            return f"/api/jobs/{job_id}/files/{os.path.relpath(path, job_dir)}"

        directions = group_by_direction(req["items"])
        detail_urls = {}
        for d in directions:
            for it in d["items"]:
                if it.get("detail_collage"):
                    detail_urls[it["index"]] = url(it["detail_collage"])
            d["items"] = [it["index"] for it in d["items"]]
        items = []
        for it in req["items"]:
            extra = {"collage_url": url(it["collage"])}
            if it["index"] in detail_urls:
                extra["detail_collage_url"] = detail_urls[it["index"]]
            items.append({**it, **extra})
        return {
            "round": rnd,
            "items": items,
            "n_items": len(items),
            "directions": directions,
        }

    @app.post("/api/jobs/{job_id}/feedback")
    async def feedback_response(job_id: str, request: Request):
        """
        Accept the verdicts of a feedback round.

        The JSON body holds ``round`` and ``verdicts`` (item index to
        ``"true"``, ``"false"`` or ``"ood"``); every requested item needs a
        verdict. The result is written atomically to
        ``feedback/feedback_response_<round>.json`` for the worker.
        """
        get_job(job_id)
        payload = await request.json()
        fdir = os.path.join(store.job_dir(job_id), "feedback")
        rnd = payload.get("round")
        verdicts = payload.get("verdicts") or {}
        if rnd is None or not os.path.isfile(
            os.path.join(fdir, f"feedback_request_{rnd}.json")
        ):
            raise HTTPException(400, "unknown feedback round")
        with open(os.path.join(fdir, f"feedback_request_{rnd}.json")) as f:
            req = json.load(f)
        expected = {int(it["index"]) for it in req["items"]}
        clean = {}
        for k, v in verdicts.items():
            if v not in ("true", "false", "ood"):
                raise HTTPException(400, f"verdict for {k} must be true, false or ood")
            clean[int(k)] = v
        missing = expected - set(clean)
        if missing:
            raise HTTPException(400, f"missing verdicts for {sorted(missing)}")
        tmp = os.path.join(fdir, f".feedback_response_{rnd}.tmp")
        with open(tmp, "w") as f:
            json.dump(
                {
                    "round": rnd,
                    "verdicts": {str(k): v for k, v in clean.items()},
                    "submitted": time.time(),
                },
                f,
            )
        os.replace(tmp, os.path.join(fdir, f"feedback_response_{rnd}.json"))
        return {"ok": True, "round": rnd, "n": len(clean)}

    @app.get("/api/jobs/{job_id}/progress")
    def progress(job_id: str):
        """Return ``run/sweep_progress.json``: the latent ranking of the
        directions being rendered and their ambient / verified counts so far,
        or ``{"stage": null}`` before the sweep reaches rendering."""
        get_job(job_id)
        path = os.path.join(store.job_dir(job_id), "run", "sweep_progress.json")
        if not os.path.isfile(path):
            return {"stage": None}
        with open(path) as f:
            return json.load(f)

    @app.get("/api/jobs/{job_id}/analysis")
    def analysis(job_id: str):
        """
        Return ``analysis.json`` (see :mod:`peal.web.analysis`) with image URLs.

        ``status`` is ``"ready"``, ``"running"`` (``analysis.log`` exists but
        no result yet), ``"failed"`` (the log ends in a traceback) or
        ``"pending"``.
        """
        get_job(job_id)
        job_dir = store.job_dir(job_id)
        path = os.path.join(job_dir, "analysis.json")
        log = os.path.join(job_dir, "analysis.log")
        if not os.path.isfile(path):
            if not os.path.isfile(log):
                return {"status": "pending"}
            with open(log, errors="replace") as f:
                failed = "Traceback" in f.read()
            return {"status": "failed" if failed else "running"}
        with open(path) as f:
            out = json.load(f)
        for d in out.get("directions", []):
            for p in d.get("pairs", []):
                p["image_url"] = f"/api/jobs/{job_id}/files/{p['image']}"
        out["status"] = "ready"
        return out

    # ------------------------------------------------------------- results
    @app.get("/api/jobs/{job_id}/results")
    def results(job_id: str):
        """Return ``results.json`` with collage and download URLs filled in."""
        job = get_job(job_id)
        path = os.path.join(store.job_dir(job_id), "results.json")
        if not os.path.isfile(path):
            raise HTTPException(404, f"no results yet (status {job['status']})")
        with open(path) as f:
            res = json.load(f)
        for d in res.get("directions", []):
            d["collage_urls"] = [
                f"/api/jobs/{job_id}/files/{p}" for p in d.get("collages", [])
            ]
        res["download_urls"] = {
            k: f"/api/jobs/{job_id}/files/{v}"
            for k, v in res.get("outputs", {}).items()
        }
        return res

    @app.get("/api/jobs/{job_id}/files/{path:path}")
    def job_file(job_id: str, path: str):
        """Serve a whitelisted file (collages, exported models, results) of a job."""
        get_job(job_id)
        job_dir = os.path.abspath(store.job_dir(job_id))
        target = os.path.abspath(os.path.join(job_dir, path))
        if not target.startswith(job_dir + os.sep) or not os.path.isfile(target):
            raise HTTPException(404, "no such file")
        allowed = path.startswith(
            (
                "run/direction_collages/",
                "run/model.onnx",
                "run/model.cpl",
                "results.json",
            )
        ) or (path.startswith("run/successful_flips/") and path.endswith(".png"))
        if not allowed:
            raise HTTPException(403, "not a downloadable file")
        return FileResponse(target)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def __getattr__(name):
    """Lazily build the module-level ``app`` on first attribute access."""
    # `uvicorn peal.web.app:app` builds the app (and starts the worker) on first
    # access; importing create_app for tests or a custom jobs root does not.
    if name == "app":
        global app
        app = create_app()
        return app
    raise AttributeError(name)

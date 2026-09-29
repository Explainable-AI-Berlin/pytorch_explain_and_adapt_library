"""
After a job's run_didae.py subprocess exits, condense its run directory into
<job>/results.json for the results page:

    python -m peal.web.summarize <job_dir>

Reads run/sweep_results.pt (the direction ranking), run/direction_feedback.txt
(the uploader's verdicts), run/direction_collages/** (the rendered
counterfactual collages) and log.txt (accuracies), and lists the deliverables
(run/model.onnx, run/model.cpl).
"""

import glob
import json
import os
import re
import sys


def _feedback(run_dir):
    """Parse ``direction_feedback.txt`` lines ``direction=<i>, feedback=<v>, ...``
    into ``{direction_idx: {"feedback", "n_true", "n_false", "n_pairs"}}``."""
    path = os.path.join(run_dir, "direction_feedback.txt")
    out = {}
    if not os.path.isfile(path):
        return out
    for line in open(path):
        m = re.match(r"direction=(\d+), feedback=(\w+)", line.strip())
        if m:
            fields = dict(
                kv.split("=", 1) for kv in line.strip().split(", ") if "=" in kv
            )
            out[int(m.group(1))] = {
                "feedback": m.group(2),
                "n_true": fields.get("n_true"),
                "n_false": fields.get("n_false"),
                "n_pairs": fields.get("n_pairs"),
            }
    return out


def _collages(run_dir, job_dir):
    """Map direction index -> PNG paths (relative to ``job_dir``) found under
    ``direction_collages/rank<r>_dir<d>_*/``."""
    out = {}
    root = os.path.join(run_dir, "direction_collages")
    for folder in sorted(glob.glob(os.path.join(root, "rank*_dir*_*"))):
        m = re.match(r"rank(\d+)_dir(\d+)_", os.path.basename(folder))
        if not m:
            continue
        pngs = sorted(glob.glob(os.path.join(folder, "*.png")))
        out[int(m.group(2))] = [os.path.relpath(p, job_dir) for p in pngs]
    return out


def _accuracies(log_path):
    """Grep ``log.txt`` for ``[DiDAE] <name> test accuracy: <v>`` lines (stored
    as ``<name>_test_accuracy``) and the student's pre-CFKD accuracy."""
    out = {}
    if not os.path.isfile(log_path):
        return out
    for line in open(log_path, errors="replace"):
        m = re.search(
            r"\[DiDAE\]\s*(.*?)\s*test accuracy:\s*([0-9.]+)", line, re.IGNORECASE
        )
        if m:
            key = (m.group(1) or "test").strip().lower().replace(" ", "_").replace(
                "-", "_"
            ) or "test"
            out[key + "_test_accuracy"] = float(m.group(2))
        m = re.search(r"Student (?:test )?accuracy(?: before CFKD)?:\s*([0-9.]+)", line)
        if m:
            out.setdefault("pre_cfkd_test_accuracy", float(m.group(1)))
    return out


def summarize(job_dir):
    """Condense a finished web-demo job into ``<job_dir>/results.json``.

    Parameters
    ----------
    job_dir : str
        Job directory containing ``run/`` (the DiDAE run directory) and
        ``log.txt`` (captured stdout of ``run_didae.py``).

    Returns
    -------
    dict
        The written summary with keys ``directions`` (one entry per ranked
        direction from ``run/sweep_results.pt`` with its flip counts, verdict
        and collage paths), ``feedback`` (verdicts keyed by direction index as
        strings), ``false_directions`` (indices judged ``"false"``),
        ``accuracies`` (parsed from ``log.txt``), ``outputs`` (relative paths
        of ``run/model.onnx`` / ``run/model.cpl`` when present) and
        ``corrected`` (True when an ONNX model exists and at least one
        direction was marked false).

    Notes
    -----
    ``torch`` is imported lazily and only when ``run/sweep_results.pt`` exists,
    so the summary of a failed job can be built without a GPU environment.
    All paths in the result are relative to ``job_dir`` so the results page
    can serve them directly.
    """
    job_dir = os.path.abspath(job_dir)
    run_dir = os.path.join(job_dir, "run")
    result = {"directions": [], "feedback": {}, "accuracies": {}, "outputs": {}}
    sweep_path = os.path.join(run_dir, "sweep_results.pt")
    feedback = _feedback(run_dir)
    collages = _collages(run_dir, job_dir)
    if os.path.isfile(sweep_path):
        import torch

        meta = torch.load(sweep_path, map_location="cpu", weights_only=False)
        for rank, r in enumerate(meta):
            idx = int(r["direction_idx"])
            fb = feedback.get(idx, {})
            result["directions"].append(
                {
                    "rank": rank + 1,
                    "direction_idx": idx,
                    "name": str(r.get("dimension_name", f"dim_{idx}")),
                    "latent_flips": r.get("latent_flip_count"),
                    "verified_flips": r.get("success_count"),
                    "ambient_flips": r.get("ambient_flip_count"),
                    "total_attempted": r.get("total_attempted"),
                    "feedback": fb.get("feedback"),
                    "collages": collages.get(idx, []),
                }
            )
    result["feedback"] = {str(k): v for k, v in feedback.items()}
    result["false_directions"] = [
        int(k) for k, v in feedback.items() if v.get("feedback") == "false"
    ]
    result["accuracies"] = _accuracies(os.path.join(job_dir, "log.txt"))
    for name in ("model.onnx", "model.cpl"):
        p = os.path.join(run_dir, name)
        if os.path.isfile(p):
            result["outputs"][name] = os.path.relpath(p, job_dir)
    dfr_path = os.path.join(job_dir, "dfr.json")
    if os.path.isfile(dfr_path):
        with open(dfr_path) as f:
            result["dfr"] = json.load(f)
    result["corrected"] = "model.onnx" in result["outputs"] and (
        "dfr" in result or len(result["false_directions"]) > 0
    )
    with open(os.path.join(job_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    r = summarize(sys.argv[1])
    print(
        json.dumps(
            {
                k: (len(v) if isinstance(v, list) else v)
                for k, v in r.items()
                if k != "feedback"
            },
            indent=2,
        )
    )

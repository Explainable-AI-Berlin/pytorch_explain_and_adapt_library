#!/usr/bin/env python
"""Collect the paper/thesis tables mechanically from finished PEAL run trees.

This reads only artefacts that a completed run already wrote -- it trains
nothing and changes nothing under $PEAL_RUNS.

For every ``<tree>/<method>`` it reads

* ``<tree>/<method>/0/validation_stats.npz``  -> the desiderata metrics, and
* ``<tree>/<method>/logs/events*``            -> the final ``gain`` scalar,

and writes a machine-readable JSON index plus, optionally, the body of the
LaTeX table that the thesis ``\\input``s. Runs that share a label (e.g. the
same dataset at different seeds) are aggregated as mean +- population std.

Examples
--------
    # everything below $PEAL_RUNS that looks like a finished explainer run
    python reproduction_scripts/collect_results.py --runs "$PEAL_RUNS" --discover \
        --out results/didae_results.json --latex results/didae_main_table.tex

    # a hand-picked set of trees, with explicit labels
    python reproduction_scripts/collect_results.py --runs "$PEAL_RUNS" \
        --tree "Square=square1k/colora_confounding_colorb/torchvision/classifier_poisoned098/repro_ddpm_s0" \
        --tree "CelebA-Blond=celeba1k/Blond_Hair/classifier_poisoned098/repro_ddpm_s0" \
        --out results/main.json
"""
import argparse
import glob
import json
import os
import re

import numpy as np

# (key in validation_stats.npz, column name, as percentage?)
# Column names deliberately mirror the stored keys: what a table calls "NAFR"
# differs between chapters, so the mapping is made in the table, not here.
METRICS = [
    ("flip_rate", "flip_rate", True),
    ("latent_diversity", "Diversity", True),
    ("latent_sparsity", "Sparsity", True),
    ("non_adversarial_rate", "NA", True),
    ("unbiasedness", "Unbiasedness", True),
    ("counterfactuals_per_second", "CF/s", False),
]
SKIP = re.compile(r"_old_\d|^checkpoints$|^logs$|^wandb$")


def final_gain(method_dir):
    """Last value of the ``gain`` scalar in this run's TensorBoard log."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        return None
    for f in sorted(glob.glob(os.path.join(method_dir, "logs", "events*"))):
        try:
            ea = EventAccumulator(f)
            ea.Reload()
            if "gain" in ea.Tags()["scalars"]:
                return float(ea.Scalars("gain")[-1].value)
        except Exception:
            continue
    return None


def read_method(method_dir):
    stats = os.path.join(method_dir, "0", "validation_stats.npz")
    if not os.path.exists(stats):
        return None
    row = {}
    z = np.load(stats, allow_pickle=True)
    for key, col, pct in METRICS:
        v = float(z[key]) if key in z.files else float("nan")
        row[col] = 100.0 * v if (pct and v == v) else v
    g = final_gain(method_dir)
    row["Gain"] = 100.0 * g if g is not None else float("nan")
    return row


def discover(runs_root, max_depth):
    """Every directory that has at least one finished method below it."""
    trees = []
    root_depth = runs_root.rstrip("/").count(os.sep)
    for dirpath, dirnames, _ in os.walk(runs_root):
        if dirpath.count(os.sep) - root_depth > max_depth:
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if not SKIP.search(d)]
        if any(
            os.path.exists(os.path.join(dirpath, d, "0", "validation_stats.npz"))
            for d in dirnames
        ):
            trees.append(os.path.relpath(dirpath, runs_root))
    return sorted(trees)


def label_of(rel_tree):
    """Dataset label for a discovered tree: its first path component."""
    return rel_tree.split(os.sep)[0]


def aggregate(rows):
    """mean +- population std over the seeds of one (label, method)."""
    out = {}
    for col in [c for _, c, _ in METRICS] + ["Gain"]:
        vals = [r[col] for r in rows if r.get(col) == r.get(col)]
        if not vals:
            out[col] = None
        else:
            out[col] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "n": len(vals),
            }
    return out


def fmt(cell, decimals=1):
    if cell is None:
        return "--"
    if cell["n"] == 1:
        return f"{cell['mean']:.{decimals}f}"
    return f"{cell['mean']:.{decimals}f}$\\pm${cell['std']:.{decimals}f}"


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--runs", default=os.environ.get("PEAL_RUNS", "./peal_runs"))
    p.add_argument(
        "--tree",
        action="append",
        default=[],
        help="LABEL=relative/path/under/PEAL_RUNS (repeatable)",
    )
    p.add_argument(
        "--discover", action="store_true", help="find finished runs automatically"
    )
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--out", default="results/results.json")
    p.add_argument("--latex", default=None, help="also write the LaTeX table body here")
    args = p.parse_args()

    trees = [(t.split("=", 1)[0], t.split("=", 1)[1]) for t in args.tree if "=" in t]
    if args.discover:
        trees += [(label_of(t), t) for t in discover(args.runs, args.max_depth)]
    if not trees:
        p.error("nothing to collect: pass --tree LABEL=PATH or --discover")

    index, missing = {}, []
    for label, rel in trees:
        path = os.path.join(args.runs, rel)
        if not os.path.isdir(path):
            missing.append(rel)
            continue
        for method in sorted(os.listdir(path)):
            if SKIP.search(method) or not os.path.isdir(os.path.join(path, method)):
                continue
            row = read_method(os.path.join(path, method))
            if row is None:
                continue
            index.setdefault(label, {}).setdefault(method, []).append(
                {"tree": rel, **row}
            )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(
            {"runs_root": args.runs, "missing": missing, "results": index}, fh, indent=1
        )
    print(
        f"wrote {args.out}: {len(index)} labels, "
        f"{sum(len(m) for m in index.values())} methods, {len(missing)} missing trees"
    )

    if args.latex:
        lines = []
        for label in sorted(index):
            for method in sorted(index[label]):
                agg = aggregate(index[label][method])
                n = max([c["n"] for c in agg.values() if c] or [0])
                cells = [
                    fmt(agg[c], 2 if c == "CF/s" else 1) for _, c, _ in METRICS
                ] + [fmt(agg["Gain"])]
                lines.append(
                    f"{label} & {method} & {n} & " + " & ".join(cells) + r" \\"
                )
        os.makedirs(os.path.dirname(os.path.abspath(args.latex)), exist_ok=True)
        with open(args.latex, "w") as fh:
            fh.write(
                "% generated by reproduction_scripts/collect_results.py -- do not edit by hand\n"
            )
            fh.write("\n".join(lines) + "\n")
        print(f"wrote {args.latex}: {len(lines)} rows")


if __name__ == "__main__":
    main()

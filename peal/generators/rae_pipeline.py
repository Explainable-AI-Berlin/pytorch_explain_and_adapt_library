"""Two-stage RAEv2 training pipeline behind the RAEDiffusionAutoencoder generator.

    python -m peal.generators.rae_pipeline --config <generator yaml> --stage {stage1,stats,cache,stage2,all}

Every stage is idempotent: it looks for its own final artifact and returns
immediately when that exists, so a supervisor can simply re-run "all" after a
crash. RAEv2's own scripts resume from the highest-epoch checkpoint in the
experiment's checkpoints/ directory, so a retry costs at most one epoch.

Layout under the generator's base_path (default $PEAL_RUNS/imagenet/rae_clip):

    config.yaml                       copy of the generator config (for --continue_training)
    stage1/<stage1_experiment>/       RAEv2 stage-1 run (checkpoints/ep-*.pt, log, src/)
    stage1_assets/decoder.pt          EMA decoder extracted from the final stage-1 checkpoint
    stage1_assets/stats.pt            per-channel latent mean/var over the train set
    stage2/<stage2_experiment>/       RAEv2 stage-2 run (checkpoints/ep-*.pt, log, src/)
    <cache_dir> (node-local /tmp)    premixed stage-2 latent + CLS cache, metadata.json marks completion

Nothing here imports torch, so it can run as the driver process for torchrun.
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import time

import yaml


def _peal_base():
    """Repository root: ``$PEAL_BASE`` or two directories above this file."""
    return os.environ.get("PEAL_BASE") or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )


#: RAEv2 is CC BY-NC 4.0, which cannot be relicensed under PEAL's LGPL-3.0. The
#: modified fork the RAE generators need therefore lives in its own
#: non-commercial distribution, peal-xai-rae (packages/peal-xai-rae in the
#: repository, `pip install "peal-xai[rae]"`), and never in the peal-xai wheel.
#: Upstream RAEv2 alone cannot load the published weights: it lacks the fork's
#: clipproj-vit-L encoder (see packages/peal-xai-rae/peal_rae/RAEv2/MODIFICATIONS.md).
RAEV2_REPO = "https://github.com/nanovisionx/RAEv2"
RAEV2_COMMIT = "8a0d238f8dc3b261aba98b217f6c79c0182e8e94"
RAEV2_LICENCE = "CC BY-NC 4.0 (non-commercial use only)"


def _packaged_raev2_dir():
    """RAEv2 root of an installed peal-xai-rae, or ``None`` when it is not installed."""
    try:
        import peal_rae  # only imports os; cheap
    except ImportError:
        return None
    return peal_rae.raev2_dir()


def default_raev2_dir():
    """Where RAEv2 is expected, first match wins:

    1. ``$PEAL_RAEV2_DIR``;
    2. an installed ``peal-xai-rae`` (``peal_rae.raev2_dir()``);
    3. the in-repository copy ``packages/peal-xai-rae/peal_rae/RAEv2`` of a clone;
    4. ``external/RAEv2``, where ``tools/install_rae.py`` used to put an upstream
       clone (kept so existing setups keep working).

    The path of the last candidate is returned even when nothing exists, so
    :func:`require_raev2` can name it in its error.
    """
    env = os.environ.get("PEAL_RAEV2_DIR")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    candidates = [
        _packaged_raev2_dir(),
        os.path.join(_peal_base(), "packages", "peal-xai-rae", "peal_rae", "RAEv2"),
    ]
    for path in candidates:
        if path and os.path.isdir(os.path.join(path, "src")):
            return path
    return os.path.join(_peal_base(), "external", "RAEv2")


def require_raev2(path):
    """Return ``path`` if RAEv2 is installed there, else raise an actionable error.

    Only the RAE generators need this. Every other PEAL generator, including the
    ImageNet and CelebA diffusion autoencoders and their DDPM inversion, runs
    without RAEv2.
    """
    if os.path.isdir(os.path.join(path, "src")):
        return path
    raise FileNotFoundError(
        f"RAEv2 was not found at {path}.\n"
        "The RAE generators (RAEDiffusionAutoencoder) need it; nothing else in PEAL does.\n"
        f"It is licensed {RAEV2_LICENCE}, so it ships separately from PEAL's LGPL code:\n"
        '    pip install "peal-xai[rae]"            # from PyPI\n'
        "    pip install ./packages/peal-xai-rae    # from a clone of the repository\n"
        "or point $PEAL_RAEV2_DIR at a copy of the fork."
    )


def expand_path(p, raev2_dir=None):
    """Expand ``<PEAL_RAEV2>``, ``<PEAL_BASE>``, ``$PEAL_RUNS``, ``$PEAL_DATA``
    and ``~`` in a path.

    ``<PEAL_RAEV2>`` points inside the RAEv2 checkout (see
    :func:`default_raev2_dir`), which holds the decoder architecture configs and
    the DINO discriminator checkpoint. It replaced
    ``<PEAL_BASE>/peal/dependencies/ADA/third_party/RAEv2`` on 2026-09-25.

    Parameters
    ----------
    p : str or None
        Path possibly containing PEAL placeholders.
    raev2_dir : str, optional
        Where RAEv2 lives; defaults to :func:`default_raev2_dir`.

    Returns
    -------
    str or None
        The expanded path, or ``None`` when ``p`` is ``None``.
    """
    if p is None:
        return None
    p = str(p)
    p = p.replace("<PEAL_RAEV2>", raev2_dir or default_raev2_dir())
    p = p.replace("<PEAL_BASE>", _peal_base())
    p = p.replace("$PEAL_RUNS", os.environ.get("PEAL_RUNS", "peal_runs"))
    p = p.replace("$PEAL_DATA", os.environ.get("PEAL_DATA", "datasets"))
    return os.path.expanduser(p)


def load_generator_config(path):
    """Load an RAE generator yaml and fill in the pipeline defaults.

    Path-valued keys (``base_path``, ``stage1_config``, ``stage2_config``,
    ``raev2_dir``, ``data``) are expanded, and defaults are set for
    ``raev2_dir``, the experiment names, precisions, stats sample counts,
    ``keep_checkpoints`` and the stage-2 cache settings.

    Parameters
    ----------
    path : str
        Path of the generator config yaml.

    Returns
    -------
    dict
        The config dict consumed by :class:`RAEPipeline`.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in ("base_path", "stage1_config", "stage2_config", "raev2_dir", "data"):
        if key in cfg and isinstance(cfg[key], str):
            cfg[key] = expand_path(cfg[key])
    cfg.setdefault("raev2_dir", default_raev2_dir())
    cfg.setdefault("stage1_experiment", "stage1")
    cfg.setdefault("stage2_experiment", "stage2")
    cfg.setdefault("stage1_precision", "bf16")
    cfg.setdefault("stage2_precision", "bf16")
    cfg.setdefault("stats_num_samples", 100000)
    cfg.setdefault("stats_batch_size", 128)
    cfg.setdefault("keep_checkpoints", 2)
    cfg.setdefault("cache_dir", "/tmp/rae_clip_stage2_cache")
    cfg.setdefault("cache_views", "original,hflip")
    cfg.setdefault("cache_batch_size", 128)
    cfg.setdefault("cache_permutation_seed", 20260911)
    return cfg


class RAEPipeline:
    """Drive RAEv2's stage-1, stats, cache and stage-2 scripts as subprocesses.

    Each ``run_*`` method checks its own completion marker first, so the
    pipeline can be restarted after a crash. Training scripts run under
    ``torch.distributed.run`` with one process per visible GPU (or
    ``$RAE_NPROC``), with ``cwd`` set to the RAEv2 checkout and its ``src``
    and ``.deps`` directories on ``PYTHONPATH``. Output goes to
    ``<base_path>/{stage1,stats,cache,stage2}.log``.

    Parameters
    ----------
    cfg : dict
        Config from :func:`load_generator_config`; uses ``base_path``,
        ``raev2_dir``, ``stage1_config``, ``stage2_config``, the experiment
        names, precisions, ``keep_checkpoints`` and the ``cache_*`` keys.
    logger : callable, optional
        Receives progress strings. Default ``print``.

    Attributes
    ----------
    stage1_exp, stage2_exp : str
        The RAEv2 experiment directories.
    decoder_path, stats_path : str
        Stage-1 assets consumed by stage 2.
    stage1_yaml, stage2_yaml : dict
        The parsed RAEv2 stage configs.
    """

    def __init__(self, cfg, logger=print):
        """Resolve the run layout from ``cfg`` and parse both RAEv2 stage configs."""
        self.cfg = cfg
        self.log = logger
        self.base = cfg["base_path"]
        self.raev2 = cfg["raev2_dir"]
        self.stage1_dir = os.path.join(self.base, "stage1")
        self.stage2_dir = os.path.join(self.base, "stage2")
        self.assets = os.path.join(self.base, "stage1_assets")
        self.stage1_exp = os.path.join(self.stage1_dir, cfg["stage1_experiment"])
        self.stage2_exp = os.path.join(self.stage2_dir, cfg["stage2_experiment"])
        self.decoder_path = os.path.join(self.assets, "decoder.pt")
        self.stats_path = os.path.join(self.assets, "stats.pt")
        with open(cfg["stage1_config"]) as f:
            self.stage1_yaml = yaml.safe_load(f)
        with open(cfg["stage2_config"]) as f:
            self.stage2_yaml = yaml.safe_load(f)
        self._resolved = {}

    # ------------------------------------------------------------------ utils
    def resolved_stage_config(self, stage):
        """Path of a stage config with PEAL placeholders already expanded.

        RAEv2's train scripts receive ``--config <path>`` and parse it with
        OmegaConf, which knows nothing about ``<PEAL_RAEV2>`` or ``<PEAL_BASE>``;
        a placeholder would reach it verbatim and surface as
        ``Incorrect path_or_model_id: '<PEAL_RAEV2>/configs/decoder/ViTXL'``.
        The expanded copy is written once per process under ``<base>/resolved``
        and reused.

        Parameters
        ----------
        stage : str
            ``"stage1"`` or ``"stage2"``.

        Returns
        -------
        str
            Path of the expanded copy, or of the original when it held no
            placeholder.
        """
        if stage in self._resolved:
            return self._resolved[stage]
        source = self.cfg[f"{stage}_config"]
        loaded = self.stage1_yaml if stage == "stage1" else self.stage2_yaml

        def expand(node):
            if isinstance(node, dict):
                return {k: expand(v) for k, v in node.items()}
            if isinstance(node, list):
                return [expand(v) for v in node]
            if isinstance(node, str) and (
                "<PEAL_RAEV2>" in node or "<PEAL_BASE>" in node
            ):
                return expand_path(node, raev2_dir=self.raev2)
            return node

        expanded = expand(loaded)
        if expanded == loaded:
            self._resolved[stage] = source
            return source

        out_dir = os.path.join(self.base, "resolved")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, f"{stage}_{os.path.basename(source)}")
        with open(out, "w") as f:
            yaml.safe_dump(expanded, f, sort_keys=False)
        self.log(f"[rae_pipeline] wrote resolved {stage} config to {out}")
        self._resolved[stage] = out
        return out

    def _ngpu(self):
        """Number of processes for torchrun: ``$RAE_NPROC`` or the GPU count."""
        n = os.environ.get("RAE_NPROC")
        if n:
            return int(n)
        try:
            out = subprocess.run(
                ["nvidia-smi", "--list-gpus"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            return max(1, len([l for l in out.splitlines() if l.strip()]))
        except Exception:
            return 1

    def _env(self, experiment_name):
        """Subprocess environment with RAEv2 on ``PYTHONPATH`` and W&B offline."""
        env = dict(os.environ)
        env["EXPERIMENT_NAME"] = experiment_name
        env["RAE_STAGE2_CACHE"] = self.cfg["cache_dir"]
        env.setdefault("WANDB_MODE", "offline")
        env.setdefault("XFORMERS_DISABLED", "1")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        deps = os.path.join(self.raev2, ".deps")
        src = os.path.join(self.raev2, "src")
        env["PYTHONPATH"] = ":".join(
            [p for p in [deps, src, env.get("PYTHONPATH", "")] if p]
        )
        return env

    def _run(self, cmd, experiment_name, log_name):
        """Run ``cmd`` in the RAEv2 checkout, appending output to a log file.

        Raises ``RuntimeError`` on a non-zero exit code.
        """
        os.makedirs(self.base, exist_ok=True)
        log_path = os.path.join(self.base, log_name)
        self.log(f"[rae_pipeline] cwd={self.raev2}")
        self.log(f"[rae_pipeline] $ {' '.join(cmd)}")
        self.log(f"[rae_pipeline] log -> {log_path}")
        t0 = time.time()
        with open(log_path, "a") as lf:
            lf.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
            lf.flush()
            proc = subprocess.run(
                cmd,
                cwd=self.raev2,
                env=self._env(experiment_name),
                stdout=lf,
                stderr=subprocess.STDOUT,
            )
        self.log(
            f"[rae_pipeline] exit {proc.returncode} after {(time.time() - t0) / 3600:.2f} h"
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"{cmd[1] if len(cmd) > 1 else cmd[0]} failed with rc={proc.returncode}; see {log_path}"
            )

    def _torchrun(self, script, args, experiment_name, log_name):
        """Run an RAEv2 script under ``torch.distributed.run --standalone``."""
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={self._ngpu()}",
            script,
        ] + args
        self._run(cmd, experiment_name, log_name)

    @staticmethod
    def _epoch_ckpts(exp_dir):
        """Sorted ``(epoch, path)`` pairs of ``checkpoints/ep-*.pt`` in ``exp_dir``."""
        found = []
        for f in glob.glob(os.path.join(exp_dir, "checkpoints", "ep-*.pt")):
            m = re.search(r"ep-(\d+)\.pt$", f)
            if m:
                found.append((int(m.group(1)), f))
        return sorted(found)

    def prune_checkpoints(self, exp_dir, keep=None):
        """Delete all but the newest ``keep`` epoch checkpoints of an experiment.

        ``ep-last.pt`` is a hard link and survives on its own. RAEv2 writes
        about 10 GB per epoch and never prunes.

        Parameters
        ----------
        exp_dir : str
            RAEv2 experiment directory containing ``checkpoints/``.
        keep : int, optional
            How many to keep; defaults to ``cfg["keep_checkpoints"]``. A value
            of 0 or less prunes nothing.
        """
        keep = self.cfg["keep_checkpoints"] if keep is None else keep
        ckpts = self._epoch_ckpts(exp_dir)
        for _, f in ckpts[:-keep] if keep > 0 else []:
            try:
                os.remove(f)
                self.log(f"[rae_pipeline] pruned {f}")
            except OSError as e:
                self.log(f"[rae_pipeline] could not prune {f}: {e}")

    def save_config_copy(self, src_path):
        """Copy the generator yaml to ``<base_path>/config.yaml`` (if different)."""
        os.makedirs(self.base, exist_ok=True)
        dst = os.path.join(self.base, "config.yaml")
        if os.path.abspath(src_path) != os.path.abspath(dst):
            shutil.copyfile(src_path, dst)

    # ----------------------------------------------------------------- stage 1
    def stage1_final_ckpt(self):
        """Path of the stage-1 checkpoint for the configured final epoch."""
        epochs = int(self.stage1_yaml["training"]["epochs"])
        return os.path.join(self.stage1_exp, "checkpoints", f"ep-{epochs:07d}.pt")

    def stage1_done(self):
        """Whether the final stage-1 checkpoint exists."""
        return os.path.isfile(self.stage1_final_ckpt())

    def run_stage1(self):
        """Train the RAEv2 stage-1 autoencoder unless already finished.

        Prunes old checkpoints first, then runs ``src/train_stage1.py`` with
        ``stage1_config``, ``stage1_dir`` and ``stage1_precision``.

        Raises
        ------
        RuntimeError
            When the script exits 0 without writing the final checkpoint.
        """
        if self.stage1_done():
            self.log(f"[rae_pipeline] stage 1 done: {self.stage1_final_ckpt()}")
            return
        self.prune_checkpoints(self.stage1_exp)
        os.makedirs(self.stage1_dir, exist_ok=True)
        self._torchrun(
            "src/train_stage1.py",
            [
                "--config",
                self.resolved_stage_config("stage1"),
                "--results-dir",
                self.stage1_dir,
                "--precision",
                self.cfg["stage1_precision"],
            ],
            self.cfg["stage1_experiment"],
            "stage1.log",
        )
        if not self.stage1_done():
            raise RuntimeError("stage 1 exited 0 but the final checkpoint is missing")

    # ------------------------------------------------------------------ stats
    def stats_done(self):
        """Whether both ``decoder.pt`` and ``stats.pt`` exist."""
        return os.path.isfile(self.decoder_path) and os.path.isfile(self.stats_path)

    def run_stats(self):
        """Extract the EMA decoder and per-channel latent statistics of stage 1.

        Runs ``scripts/stage1/extract_decoder.py`` and
        ``scripts/stage1/compute_encoder_stats.py`` (the latter under torchrun
        over ``stats_num_samples`` train images), skipping whichever asset is
        already present.

        Raises
        ------
        RuntimeError
            When stage 1 has not finished, or the assets are still missing
            after the scripts exit 0.
        """
        if not self.stage1_done():
            raise RuntimeError("stage 1 has not finished; cannot extract the decoder")
        os.makedirs(self.assets, exist_ok=True)
        if not os.path.isfile(self.decoder_path):
            self._run(
                [
                    sys.executable,
                    "scripts/stage1/extract_decoder.py",
                    "--config",
                    self.resolved_stage_config("stage1"),
                    "--ckpt",
                    self.stage1_final_ckpt(),
                    "--use-ema",
                    "--out",
                    self.decoder_path,
                ],
                self.cfg["stage1_experiment"],
                "stats.log",
            )
        else:
            self.log(f"[rae_pipeline] decoder present: {self.decoder_path}")
        if not os.path.isfile(self.stats_path):
            ds = self.stage1_yaml["dataset"]
            data_path = os.path.join(ds["data_dir"], str(ds.get("split", "train")))
            self._torchrun(
                "scripts/stage1/compute_encoder_stats.py",
                [
                    "--config",
                    self.resolved_stage_config("stage1"),
                    "--data-path",
                    data_path,
                    "--batch-size",
                    str(self.cfg["stats_batch_size"]),
                    "--num-samples",
                    str(self.cfg["stats_num_samples"]),
                    "--num-workers",
                    "8",
                    "--output-path",
                    self.stats_path,
                ],
                self.cfg["stage1_experiment"],
                "stats.log",
            )
        else:
            self.log(f"[rae_pipeline] stats present: {self.stats_path}")
        if not self.stats_done():
            raise RuntimeError(
                "stats stage exited 0 but decoder.pt/stats.pt are missing"
            )

    # ------------------------------------------------------------------ cache
    def cache_done(self):
        """Whether the stage-2 latent cache has its ``metadata.json`` marker."""
        return os.path.isfile(os.path.join(self.cfg["cache_dir"], "metadata.json"))

    def run_cache(self):
        """Encode the train set once (latent + CLIP embedding) into premixed shards.

        The shard-batch sampler serves every micro-batch from a single shard, so
        the source order is globally permuted first (cache_permutation_seed);
        without that each shard would hold one ImageFolder class. A cache
        directory that already holds ``rank*`` files but no ``metadata.json``
        is treated as incomplete and rebuilt with ``--overwrite``.

        Raises
        ------
        RuntimeError
            When the stage-1 assets are missing or the build exits 0 without
            writing ``metadata.json``.
        """
        if self.cache_done():
            self.log(f"[rae_pipeline] cache present: {self.cfg['cache_dir']}")
            return
        if not self.stats_done():
            raise RuntimeError("stage-1 assets missing; run the stats stage first")
        out = self.cfg["cache_dir"]
        partial = os.path.isdir(out) and any(
            f.startswith("rank") for f in os.listdir(out)
        )
        ds = self.stage1_yaml[
            "dataset"
        ]  # the stage-2 yaml describes the cache, not the images
        args = [
            "--config",
            self.resolved_stage_config("stage2"),
            "--out",
            out,
            "--data-dir",
            str(ds["data_dir"]),
            "--split",
            str(ds.get("split", "train")),
            "--views",
            str(self.cfg["cache_views"]),
            "--batch-size",
            str(self.cfg["cache_batch_size"]),
            "--num-workers",
            "10",
            "--shard-size",
            "512",
            "--output-dtype",
            "bf16",
            "--include-cls",
            "--global-permutation-seed",
            str(self.cfg["cache_permutation_seed"]),
        ]
        if partial:
            self.log(f"[rae_pipeline] incomplete cache at {out}; rebuilding")
            args.append("--overwrite")
        self._torchrun(
            "scripts/build_stage2_latent_cache_distributed.py",
            args,
            self.cfg["stage2_experiment"],
            "cache.log",
        )
        if not self.cache_done():
            raise RuntimeError("cache build exited 0 but metadata.json is missing")

    # ----------------------------------------------------------------- stage 2
    def stage2_final_ckpt(self):
        """Path of the stage-2 checkpoint for the configured final epoch."""
        epochs = int(self.stage2_yaml["training"]["epochs"])
        return os.path.join(self.stage2_exp, "checkpoints", f"ep-{epochs:07d}.pt")

    def stage2_done(self):
        """Whether the final stage-2 checkpoint exists."""
        return os.path.isfile(self.stage2_final_ckpt())

    def run_stage2(self):
        """Train the RAEv2 stage-2 diffusion model on the latent cache.

        Verifies that the stage-2 yaml's ``pretrained_decoder_path`` and
        ``normalization_stat_path`` point at this pipeline's assets, prunes old
        checkpoints and runs ``src/train.py`` under torchrun.

        Raises
        ------
        RuntimeError
            When prerequisites are missing, the stage-2 config points at other
            assets, or the final checkpoint is missing after a clean exit.
        """
        if self.stage2_done():
            self.log(f"[rae_pipeline] stage 2 done: {self.stage2_final_ckpt()}")
            return
        if not self.stats_done():
            raise RuntimeError("stage-1 assets missing; run the stats stage first")
        if not self.cache_done():
            raise RuntimeError(
                "stage-2 latent cache missing; run the cache stage first"
            )
        s1 = self.stage2_yaml["stage_1"]["params"]
        for key, path in (
            ("pretrained_decoder_path", self.decoder_path),
            ("normalization_stat_path", self.stats_path),
        ):
            if os.path.abspath(str(s1.get(key, ""))) != os.path.abspath(path):
                raise RuntimeError(
                    f"stage-2 config {key}={s1.get(key)} does not point at {path}"
                )
        self.prune_checkpoints(self.stage2_exp)
        os.makedirs(self.stage2_dir, exist_ok=True)
        self._torchrun(
            "src/train.py",
            [
                "--config",
                self.resolved_stage_config("stage2"),
                "--results-dir",
                self.stage2_dir,
                "--precision",
                self.cfg["stage2_precision"],
            ],
            self.cfg["stage2_experiment"],
            "stage2.log",
        )
        if not self.stage2_done():
            raise RuntimeError("stage 2 exited 0 but the final checkpoint is missing")

    def run(self, stage="all"):
        """Run one stage, or all four in order.

        Parameters
        ----------
        stage : {"stage1", "stats", "cache", "stage2", "all"}, optional
            Which stage to run. Default ``"all"``.
        """
        if stage in ("stage1", "all"):
            self.run_stage1()
        if stage in ("stats", "all"):
            self.run_stats()
        if stage in ("cache", "all"):
            self.run_cache()
        if stage in ("stage2", "all"):
            self.run_stage2()


def main():
    """CLI entry point: ``--config <generator yaml> --stage <stage>``."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--stage", default="all", choices=["stage1", "stats", "cache", "stage2", "all"]
    )
    args = ap.parse_args()
    cfg = load_generator_config(args.config)
    pipe = RAEPipeline(cfg)
    pipe.save_config_copy(args.config)
    pipe.run(args.stage)


if __name__ == "__main__":
    main()

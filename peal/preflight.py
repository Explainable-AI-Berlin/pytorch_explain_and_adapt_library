"""Check whether a run config *could* run, without starting it.

Starting a PEAL run costs anywhere from minutes to a day, and several failure
modes only surface late: a config that does not parse, a student or generator
that is not on this machine, an output directory that would be moved aside
because ``overwrite`` is on. This module answers those questions in under a
second so a reproduction script, or CI, can refuse to start instead.

Use it from the command line via ``preflight.py`` in the repository root, or
programmatically::

    from peal.preflight import check_config, check_script
    problems, notes = check_config("configs/.../square1000x098_sce_cfkd.yaml")
"""

import os
import re
import subprocess
from typing import List, Optional, Tuple

import yaml

#: config keys whose value is a path to something that must already exist
INPUT_KEYS = (
    "student",
    "teacher",
    "generator",
    "sparse_dictionary",
    "explainer",
    "data",
    "test_data",
    "training",
    "architecture",
    "distilled_predictor",
    "stage1_config",
    "stage2_config",
)
#: sub-keys carrying a path when one of the INPUT_KEYS holds an inline block
#: rather than a path, as in ``teacher: {type: symbolic, model: <checkpoint>}``
NESTED_PATH_KEYS = ("model", "config", "path", "checkpoint", "model_path")
#: keys naming a directory the run *writes*, which therefore need not exist yet
OUTPUT_KEYS = ("base_dir", "base_path")
#: `model_path` is both: an adaptor names the checkpoint it corrects (a file that
#: must exist), a predictor names the directory it will train into.
CHECKPOINT_SUFFIXES = (".cpl", ".pt", ".ckpt", ".pth")
#: file extensions that count as loadable weights for a generator or a sparse
#: dictionary. A generator directory keeps its `config.yaml` after the weights are
#: deleted, so the config alone is not evidence that it can generate. `.npz` matters
#: as much as the torch suffixes: a sparse dictionary stores itself as `weights.npz`,
#: and leaving it out made this check report every Procrustes and SVD dictionary in
#: the tree as weightless.
WEIGHT_SUFFIXES = (
    ".cpl",
    ".pt",
    ".ckpt",
    ".pth",
    ".safetensors",
    ".bin",
    ".npz",
    ".npy",
)
#: absolute-path prefixes that are personal by convention (another user's home
#: directory is detected separately in :func:`is_personal`)
PERSONAL_PREFIXES = ("/scratch/",)


def is_personal(value: str, resolved: str) -> bool:
    """True if a config path lives in another user's home directory or a scratch
    space, unless it sits under ``$PEAL_DATA`` or ``$PEAL_RUNS``."""
    import getpass

    roots = tuple(
        os.path.abspath(r)
        for r in (os.environ.get("PEAL_DATA"), os.environ.get("PEAL_RUNS"))
        if r
    )
    for path in (value, resolved):
        if roots and os.path.abspath(path).startswith(roots):
            return False
        if path.startswith(PERSONAL_PREFIXES):
            return True
        m = re.match(r"^/(?:home|Users)/([^/]+)/", path)
        if m and m.group(1) != getpass.getuser():
            return True
    return False


def project_root() -> str:
    """Absolute path of the repository root (the parent of the ``peal`` package)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def expand(path: str) -> str:
    """Resolve the placeholders PEAL configs use."""
    return (
        path.replace("<PEAL_BASE>", project_root())
        .replace("$PEAL_RUNS", os.environ.get("PEAL_RUNS", "peal_runs"))
        .replace("${PEAL_RUNS}", os.environ.get("PEAL_RUNS", "peal_runs"))
        .replace("$PEAL_DATA", os.environ.get("PEAL_DATA", "datasets"))
    )


def produced_by(cfg: dict) -> List[str]:
    """The artefact paths running this config would create.

    A generator config writes ``base_path/config.yaml``, a predictor writes
    ``model_path/model.cpl``, an adaptor writes under ``base_dir``. Consumers
    downstream name exactly those paths.
    """
    made: List[str] = []
    base_path = cfg.get("base_path")
    if isinstance(base_path, str) and "/" in base_path:
        made += [expand(base_path), os.path.join(expand(base_path), "config.yaml")]
    model_path = cfg.get("model_path")
    if (
        isinstance(model_path, str)
        and "/" in model_path
        and not model_path.endswith(CHECKPOINT_SUFFIXES)
    ):
        made += [expand(model_path), os.path.join(expand(model_path), "model.cpl")]
    base_dir = cfg.get("base_dir")
    if isinstance(base_dir, str) and "/" in base_dir:
        made += [expand(base_dir), os.path.join(expand(base_dir), "model.cpl")]
    return made


def check_config(
    path: str, check_existence: bool = True, will_exist: Optional[set] = None
) -> Tuple[List[str], List[str]]:
    """Return (problems, notes) for one config. Problems mean it cannot run.

    With ``check_existence=False`` only the machine-independent checks run
    (parses, no conflict markers, no foreign paths), which is what a commit
    hook on a laptop without ``$PEAL_RUNS`` can honestly assert.

    ``will_exist`` holds artefacts that an earlier line of the same reproduction
    script creates. The scripts are ordered: they train a generator and then run
    the explainers that consume it, so an input that is absent today but produced
    upstream is not a problem, it is simply a step that has not been run yet.
    Without this the checker reports a script as broken when it is merely unrun.
    """
    will_exist = will_exist or set()
    problems: List[str] = []
    notes: List[str] = []
    full = expand(path)
    rel = path.replace(project_root() + "/", "")
    try:
        text = open(full).read()
    except FileNotFoundError:
        return [f"config not found: {rel}"], []
    except IsADirectoryError:
        return [f"not a file: {rel}"], []
    if re.search(r"^<{7} ", text, re.M):
        return [f"unresolved merge-conflict markers in {rel}"], []
    try:
        cfg = yaml.safe_load(text)
    except yaml.YAMLError as error:
        return [f"not valid YAML: {str(error).splitlines()[0]}"], []
    if not isinstance(cfg, dict):
        return [f"config is not a mapping: {rel}"], []

    candidates = []
    for key in INPUT_KEYS:
        value = cfg.get(key)
        # Some of these keys carry an inline block instead of a path, and the path
        # then sits one level down: `teacher: {type: symbolic, model: ...}`. Those
        # were invisible here, so a missing teacher checkpoint only surfaced at
        # runtime.
        if isinstance(value, dict):
            for sub in NESTED_PATH_KEYS:
                nested = value.get(sub)
                if isinstance(nested, str) and "/" in nested:
                    candidates.append((f"{key}.{sub}", nested))
            continue
        if not isinstance(value, str) or "/" not in value:
            continue  # inline config, or a plain name like "Baseline:true"
        candidates.append((key, value))

    for key, value in candidates:
        resolved = expand(value)
        if is_personal(value, resolved):
            problems.append(f"{key} points at a personal path: {value}")
        elif resolved in will_exist:
            notes.append(f"{key} is produced by an earlier line of this script")
        elif check_existence and not os.path.exists(resolved):
            problems.append(f"{key} does not exist: {value}")
        elif not value.startswith(("$", "<", "/")):
            # A bare relative path resolves against whatever directory the run is
            # launched from, not against $PEAL_RUNS, so it works only by accident.
            notes.append(
                f"{key} is a relative path and will resolve against the cwd: {value}"
            )

    # A generator input points at the generator's config.yaml, and that file
    # survives even when the weights do not: the retention sweep of 2026-09-16
    # emptied `ema/`, `model/` and `opt/` under most DDPM directories while leaving
    # every config in place. A config-exists check therefore passes a generator that
    # cannot generate, and the run only fails once it tries to load.
    for key in ("generator", "sparse_dictionary", "stage1_config", "stage2_config"):
        ref = cfg.get(key)
        if not isinstance(ref, str) or not ref.endswith(("config.yaml", "config.yml")):
            continue
        ref_resolved = expand(ref)
        if not os.path.exists(ref_resolved):
            continue  # already reported above
        home = os.path.dirname(ref_resolved)
        weights = []
        for root, _dirs, files in os.walk(home):
            if root[len(home) :].count(os.sep) > 1:
                continue
            weights += [f for f in files if f.endswith(WEIGHT_SUFFIXES)]
            if weights:
                break
        if not weights:
            problems.append(
                f"{key} has a config but no weights: {ref} "
                "(its directory holds no .cpl or .pt file, so it cannot be loaded)"
            )

    # A data config names the dataset directory it reads. Checking that one level
    # down catches a whole class of failure that a config-exists check misses: the
    # tabular data configs named `datasets/<name>` relative to the working
    # directory rather than `$PEAL_DATA/<name>`, so every tabular run died at load.
    for key in ("data", "test_data", "unpoisoned_data"):
        ref = cfg.get(key)
        if not isinstance(ref, str) or not ref.endswith((".yaml", ".yml")):
            continue
        ref_resolved = expand(ref)
        if not os.path.exists(ref_resolved):
            continue  # already reported above as a missing input
        try:
            with open(ref_resolved) as handle:
                sub = yaml.safe_load(handle)
        except (OSError, yaml.YAMLError):
            continue
        dataset_path = (sub or {}).get("dataset_path")
        if not isinstance(dataset_path, str) or not dataset_path:
            continue
        # Neither of these stops a run: the tabular dataset classes fetch their
        # csv from Kaggle or a public mirror when the directory is absent. So both
        # are notes, not problems -- but both are worth saying, because a fetch
        # needs network access the compute nodes may not have, and a relative
        # dataset_path makes the download land inside the repository.
        if not dataset_path.startswith(("$", "<", "/")):
            notes.append(
                f"{key} names dataset_path {dataset_path}, which resolves against "
                "the working directory rather than $PEAL_DATA; a dataset class that "
                "downloads will write into the repository"
            )
        elif check_existence and not os.path.exists(expand(dataset_path)):
            notes.append(
                f"{key} dataset is not on this machine: {dataset_path} "
                "(the dataset class must fetch or build it, which needs network access)"
            )

    model_path = cfg.get("model_path")
    if isinstance(model_path, str) and "/" in model_path:
        resolved = expand(model_path)
        if is_personal(model_path, resolved):
            problems.append(f"model_path points at a personal path: {model_path}")
        elif model_path.endswith(CHECKPOINT_SUFFIXES):
            if resolved in will_exist:
                notes.append("model_path is produced by an earlier line of this script")
            elif check_existence and not os.path.exists(resolved):
                problems.append(f"model_path does not exist: {model_path}")
        else:
            short = model_path.replace(
                os.environ.get("PEAL_RUNS", "peal_runs"), "$PEAL_RUNS"
            )
            notes.append(f"would train into {short}")

    for key in OUTPUT_KEYS:
        target = cfg.get(key)
        if not isinstance(target, str) or "/" not in target:
            continue
        resolved = expand(target)
        short = target.replace(os.environ.get("PEAL_RUNS", "peal_runs"), "$PEAL_RUNS")
        if is_personal(target, resolved):
            problems.append(f"{key} points at a personal path: {target}")
        elif check_existence and os.path.isdir(resolved) and cfg.get("overwrite", True):
            notes.append(
                f"{short} already exists and overwrite is on, so the existing "
                "run would be moved aside and regenerated"
            )
        else:
            notes.append(f"would write to {short}")
    return problems, notes


def configs_in_script(path: str) -> List[str]:
    """Every --config argument a reproduction script passes, in order.

    The scripts use shell loops, ``for METHOD in ace dime sce; do ... done``, so a
    config path may contain ``${METHOD}``. Reading the literal text would report a
    config that does not exist when the loop in fact expands to several that do.
    Loop variables are therefore collected and substituted, one emitted config per
    value. A variable that is never bound by a ``for`` is left as written, so it
    still shows up as missing rather than being silently dropped.
    """
    text = open(path).read().replace("\\\n", " ")  # join line continuations
    bindings: dict = {}
    found, seen = [], set()
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            continue
        loop = re.match(r"for\s+([A-Za-z_][A-Za-z_0-9]*)\s+in\s+(.+?)(?:;|\s*$)", line)
        if loop:
            values = [v for v in loop.group(2).split() if v not in ("do",)]
            if values:
                bindings[loop.group(1)] = values
        for match in re.finditer(r'--config\s+"?([^"\s]+)"?', line):
            cfg = match.group(1)
            variables = re.findall(r"\$\{?([A-Za-z_][A-Za-z_0-9]*)\}?", cfg)
            expansions = [cfg]
            for var in variables:
                if var not in bindings:
                    continue
                expansions = [
                    re.sub(r"\$\{?" + var + r"\}?", value, e)
                    for e in expansions
                    for value in bindings[var]
                ]
            for e in expansions:
                if e not in seen:
                    seen.add(e)
                    found.append(e)
    return found


def scripts_invoked_by(path: str) -> List[str]:
    """Every ``python <file>`` target a reproduction script invokes, in order.

    Commented-out lines are skipped, and so is anything that looks like a flag
    rather than a path. Duplicates collapse.
    """
    text = open(path).read().replace("\\\n", " ")
    found, seen = [], set()
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            continue
        for match in re.finditer(
            r"(?:^|[|;&]\s*|\s)python[0-9.]*\s+((?:-[A-Za-z]+\s+)*)([A-Za-z0-9_][A-Za-z0-9_/.-]*\.py)",
            line,
        ):
            target = match.group(2)
            if target not in seen:
                seen.add(target)
                found.append(target)
    return found


def _tracked(rel: str) -> bool:
    """True when git has this file. False when git is unavailable or it is not tracked."""
    try:
        done = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", rel],
            cwd=project_root(),
            capture_output=True,
        )
        return done.returncode == 0
    except Exception:
        return True  # no git here: do not invent failures


def check_script_executables(path: str, require_tracked: bool = False):
    """Check the files a script *runs*, not the configs it passes.

    A config-only check cannot see this class of breakage. Three files the
    reproduction scripts invoke had never been committed, so the scripts ran
    here and would have died on a fresh clone, at a line reached only after days
    of training. ``require_tracked`` asks the stricter question, would this
    survive a clone, and is what a release check wants; the default asks only
    whether it can run on this machine.
    """
    problems, notes = [], []
    for target in scripts_invoked_by(path):
        full = os.path.join(project_root(), target)
        if not os.path.exists(full):
            problems.append(f"script invokes a file that does not exist: {target}")
        elif not _tracked(target):
            message = f"{target} exists but is not tracked by git, so a clone would not get it"
            (problems if require_tracked else notes).append(message)
    return problems, notes


def check_script(path: str, check_existence: bool = True):
    """Run check_config over every config a script references, in script order.

    Order matters: each config is checked knowing what the lines above it will
    have produced, which is how the scripts are meant to be run.

    Parameters
    ----------
    path : str
        Path of a shell reproduction script containing ``--config`` arguments.
    check_existence : bool, optional
        Forwarded to :func:`check_config`. Default ``True``.

    Returns
    -------
    list of tuple
        One ``(config_path, problems, notes)`` triple per config, in the order
        the script references them.
    """
    results, will_exist = [], set()
    for cfg in configs_in_script(path):
        problems, notes = check_config(cfg, check_existence, will_exist)
        results.append((cfg, problems, notes))
        try:
            loaded = yaml.safe_load(open(expand(cfg)))
            if isinstance(loaded, dict):
                will_exist.update(produced_by(loaded))
        except Exception:
            pass
    return results

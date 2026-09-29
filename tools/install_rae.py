"""Install the RAEv2 fork behind PEAL's RAE generators (CC BY-NC 4.0, non-commercial).

The fork lives in this repository as its own distribution, ``peal-xai-rae``
(``packages/peal-xai-rae``). It is licensed CC BY-NC 4.0 like the upstream RAEv2
it modifies, so it stays out of the LGPL ``peal-xai`` wheel and is installed
only on request; this script shows you the licence first.

Only the RAE generators need it (``RAEDiffusionAutoencoder``, the ImageNet and
CelebA representation-autoencoder pipelines). Every other PEAL generator,
including the ImageNet and CelebA diffusion autoencoders and PEAL's own DDPM
inversion, runs without it.

    python tools/install_rae.py                  # pip install ./packages/peal-xai-rae
    python tools/install_rae.py --dir ~/RAEv2    # copy the fork there instead
    python tools/install_rae.py --check          # only report what PEAL will use

After ``--dir``, point PEAL at the copy:

    export PEAL_RAEV2_DIR=~/RAEv2

From a clone nothing has to be installed at all: PEAL also finds the in-repository
copy directly. The pip route is what an installed ``peal-xai`` needs
(``pip install "peal-xai[rae]"`` does the same from PyPI).

Plain upstream RAEv2 is NOT enough: it lacks the fork's ``clipproj-vit-L``
encoder and fails on the published weights with ``Unknown encoder type:
clipproj``. ``--check`` flags such a clone.
"""

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from peal.generators.rae_pipeline import (  # noqa: E402
    RAEV2_COMMIT,
    RAEV2_LICENCE,
    RAEV2_REPO,
    default_raev2_dir,
)

PACKAGE_DIR = os.path.join(REPO, "packages", "peal-xai-rae")
FORK_DIR = os.path.join(PACKAGE_DIR, "peal_rae", "RAEv2")

#: What the RAE generator's inference path imports from RAEv2's src/ (the
#: `dependencies` of packages/peal-xai-rae/pyproject.toml, plus OpenAI CLIP).
INFERENCE_IMPORTS = {
    "omegaconf": "omegaconf",
    "einops": "einops",
    "safetensors": "safetensors",
    "timm": "timm",
    "diffusers": "diffusers",
    "transformers": "transformers",
    "clip": "clip @ git+https://github.com/openai/CLIP.git",
}

LICENCE_NOTICE = f"""
The RAE generators run on a modified RAEv2, a separate work from PEAL.

  Upstream   : {RAEV2_REPO} (commit {RAEV2_COMMIT})
  Fork       : packages/peal-xai-rae (changes listed in peal_rae/RAEv2/MODIFICATIONS.md)
  Licence    : {RAEV2_LICENCE}

CC BY-NC 4.0 permits non-commercial use with attribution. It is not an
open-source licence and it does not permit commercial use. By installing the
fork you accept that licence for it, separately from PEAL's own LGPL-3.0 terms.
The pretrained ImageNet weights (hf://sidney1505/peal-rae-clip-imagenet) are
CC BY-NC 4.0 as well.
""".strip()


def installed(path):
    """Tell whether a RAEv2 checkout exists at ``path``.

    Parameters
    ----------
    path : str
        Candidate RAEv2 root.

    Returns
    -------
    bool
        ``True`` when ``<path>/src`` is a directory.
    """
    return os.path.isdir(os.path.join(path, "src"))


def is_fork(path):
    """Tell whether the RAEv2 at ``path`` is the fork (has the clipproj encoder).

    Parameters
    ----------
    path : str
        RAEv2 root.

    Returns
    -------
    bool
        ``True`` when ``src/encoders/vision_encoder.py`` knows ``clipproj``.
    """
    encoder = os.path.join(path, "src", "encoders", "vision_encoder.py")
    try:
        with open(encoder, encoding="utf-8") as handle:
            return "clipproj" in handle.read()
    except OSError:
        return False


def missing_imports():
    """The inference dependencies that cannot be imported here.

    Returns
    -------
    list of str
        pip requirement strings for each missing module.
    """
    return [
        req
        for module, req in INFERENCE_IMPORTS.items()
        if importlib.util.find_spec(module) is None
    ]


def report(path):
    """Print where PEAL finds RAEv2 and whether it can run the published generator.

    Parameters
    ----------
    path : str
        The RAEv2 root PEAL resolves (see ``default_raev2_dir``).

    Returns
    -------
    int
        0 when a fork copy is found and its imports are satisfied, else 1.
    """
    if not installed(path):
        print(f"RAEv2 is NOT available (PEAL looked last in {path}).")
        print("The RAE generators are unavailable; every other PEAL generator works.")
        return 1
    print(f"PEAL uses RAEv2 at {path}")
    status = 0
    if not is_fork(path):
        print(
            "  WARNING: this is plain upstream RAEv2 without the clipproj-vit-L encoder;\n"
            "  the published RAE weights will not load. Remove it, unset\n"
            "  $PEAL_RAEV2_DIR, or rerun this script with --dir to replace it."
        )
        status = 1
    missing = missing_imports()
    if missing:
        print("  missing Python packages for inference:")
        print("    pip install " + " ".join(f'"{m}"' for m in missing))
        status = 1
    if status == 0:
        print("  fork with the clipproj encoder; inference dependencies present.")
    return status


def main():
    """Report, pip-install, or copy the RAEv2 fork.

    Returns
    -------
    int
        Exit status: 0 on success, 1 when RAEv2 is missing or incomplete
        (``--check``), the licence was declined, or installation failed.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--dir",
        default=None,
        help="copy the fork's RAEv2 folder here instead of pip-installing the package",
    )
    ap.add_argument("--check", action="store_true", help="report status and exit")
    ap.add_argument(
        "--yes", action="store_true", help="accept the licence without prompting"
    )
    args = ap.parse_args()

    if args.check:
        return report(default_raev2_dir())

    if not installed(FORK_DIR):
        print(f"The fork is missing from this checkout ({FORK_DIR}).", file=sys.stderr)
        return 1

    print(LICENCE_NOTICE)
    what = (
        f"Copy the fork into {os.path.abspath(os.path.expanduser(args.dir))}"
        if args.dir
        else f"pip install {os.path.relpath(PACKAGE_DIR)}"
    )
    if not args.yes:
        try:
            reply = input(f"\n{what}? [y/N] ").strip().lower()
        except EOFError:
            reply = ""
        if reply not in ("y", "yes"):
            print("Aborted. Nothing was installed.")
            return 1

    if args.dir:
        target = os.path.abspath(os.path.expanduser(args.dir))
        if os.path.exists(target):
            if not installed(target):
                print(
                    f"{target} exists and is not a RAEv2 checkout; refusing to overwrite it.",
                    file=sys.stderr,
                )
                return 1
            print(f"Replacing the RAEv2 at {target}")
            shutil.rmtree(target)
        shutil.copytree(
            FORK_DIR, target, ignore=shutil.ignore_patterns("__pycache__", ".caches")
        )
        print(f"\nRAEv2 fork copied to {target}")
        if os.path.abspath(default_raev2_dir()) != target:
            print(f"Add this to your environment:\n    export PEAL_RAEV2_DIR={target}")
    else:
        # --no-deps: from a clone, peal-xai itself is usually not pip-installed,
        # and letting pip resolve it would fetch PEAL from PyPI over the clone.
        cmd = [sys.executable, "-m", "pip", "install", "--no-deps", PACKAGE_DIR]
        print("\n" + " ".join(cmd))
        if subprocess.call(cmd) != 0:
            print("pip install failed.", file=sys.stderr)
            return 1

    missing = missing_imports()
    if missing:
        print("\nStill needed for inference:")
        print("    pip install " + " ".join(f'"{m}"' for m in missing))
    print("RAEv2 keeps its own licence (CC BY-NC 4.0); see its LICENSE file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

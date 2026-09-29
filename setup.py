import pathlib
import re

import setuptools

HERE = pathlib.Path(__file__).parent


def _version():
    """The version string, read textually from ``peal/__init__.py``.

    The file is parsed rather than imported: importing ``peal`` at build time
    would require torch and every other runtime dependency to be installed
    first, which is exactly what this script is meant to declare.

    Returns
    -------
    str
        The value of ``peal.__version__``.

    Raises
    ------
    RuntimeError
        If the assignment is missing, so a botched release fails loudly here
        instead of publishing a wheel with the wrong number.
    """
    text = (HERE / "peal" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__ = "([^"]+)"', text, re.M)
    if match is None:
        raise RuntimeError("peal/__init__.py defines no __version__")
    return match.group(1)


# What `pip install peal-xai` pulls in: the packages that first-party peal
# modules import at module level, with lower bounds rather than the exact pins
# in requirements.txt. Those pins are a frozen conda environment; feeding them
# to install_requires made PEAL demand numpy==1.22.3 and torch==2.1.2 of every
# user, drag Sphinx in as a runtime dependency, and conflict with essentially
# any existing environment. requirements.txt stays as the reproducibility
# record for the papers and is installed explicitly, not by depending on PEAL.
#
# Bounds are the versions PEAL is developed against, except where an upper
# bound is real: numpy 2 breaks the pinned scipy/scikit-learn stack, and
# pydantic 3 would break every config model.
CORE_REQUIREMENTS = [
    "torch>=2.1",
    "torchvision>=0.16",
    "numpy>=1.22,<2",
    "scipy>=1.8",
    "pandas>=1.5",
    "matplotlib>=3.5",
    "scikit-learn>=1.0",
    "Pillow>=9.0",
    "pydantic>=2.0,<3",
    "PyYAML>=6.0",
    "tqdm>=4.62",
    "psutil>=5.9",
    "requests>=2.28",
    "huggingface-hub>=0.23",
    "transformers>=4.36",
    "diffusers>=0.28",
    "timm>=0.9",
    "torchmetrics>=1.0",
    "tensorboard>=2.10",
    "zennit>=0.4.6",
    "blobfile>=2.0",
    "wget>=3.2",
    "seaborn>=0.11",
    "torch-kmeans>=0.2",
]


def _requirements(path="requirements.txt"):
    """Pinned runtime dependencies, comments and blank lines stripped.

    Used only for the ``web`` extra, which has its own small pinned file.
    requirements.txt itself is no longer fed to ``install_requires``; see
    ``CORE_REQUIREMENTS``.
    """
    try:
        with open(path) as handle:
            lines = handle.read().splitlines()
    except FileNotFoundError:
        print(
            f"[peal setup] {path} not found; installing without pinned "
            "dependencies. See environment.yaml for the full environment."
        )
        return []
    requirements = []
    for line in lines:
        # Strip trailing comments too: requirements-web.txt annotates its pins
        # inline ("onnxruntime>=1.17  # CPU wheel"), and a comment left on the
        # line would ride into install_requires as part of the specifier.
        line = line.split("#", 1)[0].strip()
        if line:
            requirements.append(line)
    return requirements


# What the distribution deliberately leaves out. See LICENSING.md for the full
# map; the short version is that the wheel stays LGPL-only.
#
#   matryoshka_sae  removed from the tree entirely on 2026-09-25: upstream
#                   declared no licence. Reimplemented as
#                   peal/sparse_dictionaries/batch_topk_network.py.
#   RAEv2           CC BY-NC 4.0, cannot be relicensed as LGPL. The modified
#                   fork lives in packages/peal-xai-rae and ships as its own
#                   non-commercial wheel, peal-xai-rae (the `rae` extra below).
#                   find_namespace_packages only looks under peal/, so it can
#                   never end up in this wheel.
#   ADA             in the git repository under CC BY-NC 4.0 (David Drexlin), but
#                   not in the distribution (see the entry below).
EXCLUDED_PACKAGES = [
    # ADA (David Drexlin) is in the git repository under CC BY-NC 4.0 (see
    # LICENSING.md), but it is kept out of the wheel: nothing under peal/
    # imports it, and its non-commercial terms must not reach everyone who
    # pip-installs PEAL. Keeping it out is what lets the package stay LGPL-only.
    "peal.dependencies.ADA",
    "peal.dependencies.ADA.*",
    # build and IDE droppings that find_namespace_packages would otherwise list
    "*__pycache__*",
    "*.outputs",
    "*.outputs.*",
]

with open("README.md") as f:
    long_description = f.read()

# Optional stacks, keyed by the name peal._optional.require prints in its error
# message. Keep the two in step: an extra renamed here without updating the
# require() call sends users to an install command that does not exist.
EXTRAS = {
    # PathLDM generators (Camelyon17, Follicles).
    "pathldm": ["omegaconf"],
    # The Stable Diffusion 3 generator.
    "stablediffusion": ["captum"],
    # ONNX classifiers as predictors (peal/architectures/onnx_predictor.py).
    # Plain onnxruntime, not onnxruntime-gpu: the uploaded classifier is
    # converted to a torch module anyway, and the GPU wheel has no
    # linux/aarch64 build (the DGX Spark the web demo targets).
    "onnx": ["onnx>=1.15", "onnxruntime>=1.17", "onnx2torch>=1.5"],
    # The FastAPI demo in peal/web, plus the Flask pages the human and cluster
    # teachers serve to collect feedback.
    "web": _requirements("requirements-web.txt") + ["flask>=2.2", "flask-cors>=3.0"],
    # SpRAy and ViRelAy teachers and the LRP explainer.
    "xai": ["corelay>=0.2.1", "virelay>=0.4.0", "zennit-crp>=0.6.0", "h5py>=3.7"],
    # WILDS-backed datasets (Camelyon17, RxRx1) and the Kaggle downloads.
    "datasets": ["wilds>=2.0", "kaggle>=1.5", "kagglehub>=0.2"],
    # OpenAI CLIP for the Stable Diffusion autoencoder. Not on PyPI under this
    # name, so it installs from source:
    #   pip install "clip @ git+https://github.com/openai/CLIP.git"
    "clip": ["open-clip-torch>=2.20"],
    # Distributed sampling in the DDPM generators; needs a system MPI.
    "mpi": ["mpi4py>=3.1"],
    # Image transforms that render through pygame.
    "pygame": ["pygame>=2.1"],
}
# Everything installable from PyPI in one go. "clip" is included for its
# open-clip half; the OpenAI package still needs the git command above.
EXTRAS["full"] = sorted({r for name, reqs in EXTRAS.items() for r in reqs})
# The RAE generators. NOT part of "full": peal-xai-rae is CC BY-NC 4.0
# (non-commercial only), so it has to be an explicit opt-in, never a side effect
# of asking for "everything". Built from packages/peal-xai-rae; the ImageNet
# weights come from hf://sidney1505/peal-rae-clip-imagenet at first use.
EXTRAS["rae"] = ["peal-xai-rae"]


setuptools.setup(
    # The import package stays `peal`; the distribution name does not,
    # because `peal` on PyPI is an unrelated active-learning project.
    name="peal-xai",
    version=_version(),
    description=(
        "PEAL: counterfactual explanation and repair of image classifiers "
        "and foundation-model probes."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Sidney Bender and the PEAL contributors",
    url="https://github.com/Explainable-AI-Berlin/pytorch_explain_and_adapt_library",
    license="LGPL-3.0-or-later",
    license_files=["LICENSE.txt", "COPYING.LESSER", "COPYING"],
    project_urls={
        "Source": "https://github.com/Explainable-AI-Berlin/pytorch_explain_and_adapt_library",
        "Documentation": "https://explainable-ai-berlin.github.io/pytorch_explain_and_adapt_library/",
        "Issues": "https://github.com/Explainable-AI-Berlin/pytorch_explain_and_adapt_library/issues",
    },
    keywords=[
        "counterfactual explanations",
        "explainable ai",
        "clever hans",
        "diffusion autoencoder",
        "sparse autoencoder",
        "model repair",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: POSIX :: Linux",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    # Upper bound is real, not defensive: numpy==1.22.3, scipy==1.8.0,
    # matplotlib==3.5.1 and scikit-learn==1.0.2 publish no cp312 wheels.
    python_requires=">=3.9,<3.12",
    # `packages=["peal"]` shipped only peal/__init__.py, so an installed wheel
    # had none of the subpackages. Excluding peal/dependencies/ wholesale was
    # the next problem: 18 first-party modules import it, so the wheel could not
    # load most generators, the editors, both GroupDRO adaptors, the trainers or
    # two of the sparse dictionaries. The vendored code is therefore shipped,
    # minus what PEAL is not allowed to redistribute (see EXCLUDED_PACKAGES and
    # THIRD_PARTY_NOTICES.md).
    #
    # find_namespace_packages, not find_packages: most vendored folders carry no
    # __init__.py and are imported as implicit namespace packages
    # (peal.dependencies.ddpm_inversion has 19 import sites and no __init__.py).
    packages=setuptools.find_namespace_packages(
        include=["peal", "peal.*"],
        exclude=EXCLUDED_PACKAGES,
    ),
    # Only .py files ship by default, which is what keeps the 12 MB glow gif,
    # the 11 MB DiCE pickle and the vendored notebooks out of the wheel. These
    # are the non-Python files that are actually read at runtime.
    package_data={
        "peal.web": ["static/*"],
        "peal.dependencies.SpLiCE": ["data/vocab/*.txt"],
        # LICENSE/UPSTREAM files travel with the code they cover. MIT and
        # Apache-2.0 both require the licence and copyright notice to accompany
        # copies, and until 2026-09-25 the wheel shipped eleven vendored
        # components' source without any of their licence texts.
        "": [
            "*.yaml",
            "*.yml",
            "*.json",
            "LICENSE",
            "LICENSE.*",
            "LICENCE",
            "COPYING*",
            "UPSTREAM.md",
            "ORIGIN.md",
        ],
    },
    install_requires=CORE_REQUIREMENTS,
    # The command-line entry points. Each module under peal/entrypoints/ keeps
    # a main() that parses sys.argv itself; the identically named scripts in the
    # repository root are thin wrappers, so `python run_cfkd.py ...` from a
    # clone and `peal-cfkd ...` from an install run the same code.
    entry_points={
        "console_scripts": [
            "peal-cfkd = peal.entrypoints.run_cfkd:main",
            "peal-didae = peal.entrypoints.run_didae:main",
            "peal-explain = peal.entrypoints.run_explainer:main",
            "peal-adapt = peal.entrypoints.run_adaptor:main",
            "peal-train-generator = peal.entrypoints.train_generator:main",
            "peal-train-predictor = peal.entrypoints.train_predictor:main",
            "peal-distill-predictor = peal.entrypoints.train_distilled_predictor:main",
            "peal-evaluate-predictor = peal.entrypoints.evaluate_predictor:main",
            "peal-sae-analysis = peal.entrypoints.run_sae_analysis:main",
            "peal-component-analysis = peal.entrypoints.run_component_analysis:main",
            "peal-generate-dataset = peal.entrypoints.generate_dataset:main",
            "peal-preflight = peal.entrypoints.preflight:main",
        ],
    },
    # Optional stacks. Each one backs a subset of the library that imports its
    # packages lazily through peal._optional.require, so a missing extra
    # surfaces as an ImportError naming the install command instead of
    # breaking `import peal`.
    extras_require=EXTRAS,
)

# PEAL tutorial notebooks

Working examples of the PEAL library (`pip install peal-xai`). Each notebook builds its pipeline from
PEAL's own building blocks - config classes (`DataConfig`, `PredictorConfig`, `DiDAEConfig`, ...) and
factories (`get_datasets`, `get_predictor`, `get_generator`, `get_adaptor`, ...) - so that you can see
which object does what and swap any of them for your own. Every step has a default example and an
"Option B" cell for your own data, model or generator.

| notebook | what it shows | time on one GPU (measured on an H100) |
|---|---|---|
| [`00_tour_of_peal_with_trains.ipynb`](00_tour_of_peal_with_trains.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Explainable-AI-Berlin/pytorch_explain_and_adapt_library/blob/master/notebooks/00_tour_of_peal_with_trains.ipynb) | **Start here.** A DINOv3 probe that tells freight cars from passenger cars, and its graffiti shortcut: data, probe, the RAE generator, the MSAE concept dictionary, DiDAE's ranking, your verdicts, and a last-layer correction (DFR) that stops held-out graffiti counterfactuals from fooling it. | 10-15 min + downloads; peak 19.5 GB, 14 GB in its small-GPU mode |
| [`01_explain_and_correct_with_sce.ipynb`](01_explain_and_correct_with_sce.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Explainable-AI-Berlin/pytorch_explain_and_adapt_library/blob/master/notebooks/01_explain_and_correct_with_sce.ipynb) | The other explainer family: SCE counterfactuals from a pretrained ImageNet diffusion model, run through the CFKD adaptor, and how CFKD corrects a model with a teacher's verdicts. Uses the model and data of notebook 02. | ~20+ min + 2.1 GB download; needs a large GPU (A100); see the notebook |
| [`02_bring_your_own_model_and_data.ipynb`](02_bring_your_own_model_and_data.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Explainable-AI-Berlin/pytorch_explain_and_adapt_library/blob/master/notebooks/02_bring_your_own_model_and_data.ipynb) | The two contracts: a folder of images per class (ingested like a web-demo upload) and a model as ONNX (export, check with onnxruntime, load with `get_predictor`, match the normalization). Example: a pretrained ResNet-18 cut down to the two train classes. | < 1 min |
| [`03_explain_an_onnx_model_with_didae.ipynb`](03_explain_an_onnx_model_with_didae.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Explainable-AI-Berlin/pytorch_explain_and_adapt_library/blob/master/notebooks/03_explain_an_onnx_model_with_didae.ipynb) | DiDAE on the ONNX model of notebook 02, without training anything: which concepts change its decision, how well the distilled probe matches it, the evidence per concept. | 4-11 min + downloads; peak 19 GB (smaller pool on small GPUs) |

Suggested order: 00, then 02 -> 03 (or 02 -> 01).

## Running them

**On Google Colab:** open a notebook with its badge, choose *Runtime -> Change runtime type -> GPU*
(a free T4 is enough) and run the first cell. It installs PEAL from PyPI and restarts the runtime
once (PEAL needs `numpy < 2`); run it again and continue. Two things the first cell takes care of:

* PEAL 0.1.0 declares Python < 3.12 while Colab runs 3.12, so it installs with
  `--ignore-requires-python` (PEAL runs on 3.12).
* The pip package carries the library, not the config files, so the cell makes a small sparse clone
  of this repository (`configs/` and the dictionary's vocabulary) and points `$PEAL_BASE` at it.

**The images** of notebooks 00 and 02 are two ImageNet classes. ImageNet may not be redistributed,
so you fetch them from Hugging Face yourself: accept the terms of
[`ILSVRC/imagenet-1k`](https://huggingface.co/datasets/ILSVRC/imagenet-1k) with your account and log
in (a Colab secret `HF_TOKEN`, or `huggingface_hub.notebook_login()`). Only the parts of the train
split that hold the two classes are downloaded. Or point the notebooks at your own images.

**Locally:** `pip install "peal-xai[rae,onnx]" git+https://github.com/openai/CLIP.git` (Python 3.9-3.11)
and run the notebooks from this folder; the first cell then uses the repository's `configs/`.
`$PEAL_DATA` and `$PEAL_RUNS` default to `~/peal_data/datasets` and `~/peal_data/runs`.

Every notebook saves its results under `$PEAL_RUNS` and skips finished steps when run again
(`REUSE = True`).

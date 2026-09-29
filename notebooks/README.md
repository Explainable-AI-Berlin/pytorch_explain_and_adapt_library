# PEAL tutorial notebooks

Notebook versions of the walkthroughs in the top-level `README.md`. Each one derives its config
files from the shipped ones rather than asking you to copy and edit by hand, and ends by loading the
run's outputs so you can look at the counterfactuals in place.

Run them from a kernel with the `peal` environment active. They `chdir` to the repository root in
their first cell, because every PEAL entry point expects that as the working directory.

| notebook | what it does | trains anything? |
|---|---|---|
| `00_tour_of_peal_with_waterbirds.ipynb` | **Start here.** The whole workflow with the library's own config classes and factories: Waterbirds (downloads itself), a DINOv3 linear probe, the RAE generator and MSAE dictionary, DiDAE, your verdicts and a last-layer correction (DFR), with worst-group accuracy before and after. Every step has a default and a "your own" cell. | the probe (~9 min); ~45 min in total on one GPU |
| `01_explain_your_classifier_with_sce.ipynb` | Your images and labels in, SCE counterfactuals out. PEAL trains both the DDPM generator and the classifier. | yes, both (GPU-days for the generator) |
| `02_bring_your_own_onnx_predictor_sce.ipynb` | You supply a trained model as ONNX; SCE explains it. Uses the shipped ImageNet DDPM unless you opt into your own. | only the generator, and only if you opt in |
| `03_explain_an_onnx_probe_with_didae.ipynb` | DiDAE ranks the dictionary directions your ONNX probe actually reads, with before/after pairs as evidence. | no |

Notebook 3 is the cheapest place to start if you already have a model: it reuses a pretrained
generator and a pretrained sparse dictionary and only ever calls your model forward.

Two things that trip people up, both covered in the notebooks:

* PEAL hands an ONNX graph exactly the tensor the data config produces, with no normalization of its
  own, so `input_size` and `normalization` have to match what your model expects.
* An ONNX student can usually be fine-tuned too: PEAL converts the graph into a trainable torch
  module with `onnx2torch`. When a graph uses an operator the converter does not implement, PEAL
  warns and falls back to an inference-only `onnxruntime` closure, which can be explained and ranked
  but not repaired. The DINOv3 probes from the ImageNet experiments fall back this way, on a `Size`
  node. Notebook 3 shows how to keep the ONNX file for the sweep and point at a PyTorch checkpoint
  for the repair, which is what you need in that case.
* Notebook 3 trains nothing, but it is not dependency-free: its reference generator is a
  `RAEDiffusionAutoencoder`, which needs the separate, non-commercial RAEv2 fork (`pip install "peal-xai[rae]"` or `python tools/install_rae.py`)
  and a published weights folder in `$PEAL_RAE_WEIGHTS`.
* Notebooks 2 and 3 need `onnx` and `onnxruntime`, which the base `peal` environment does not carry.
  Install the web extras (`pip install -r requirements-web.txt`) or use the `.venv-web` interpreter.

<!-- Provenance record for a vendored third-party component of PEAL.
     Generated 2026-09-25; keep it updated when the copy is refreshed. -->

# Smoothed differentiation (SmoothDiff)

**This directory is not part of PEAL.** It is a vendored copy of third-party
research code, kept in `peal/dependencies/` and separate from PEAL's own
implementation. All credit for the method and the original implementation
belongs to the authors below; please cite their work, not PEAL, when you use it.

| | |
|---|---|
| **Original authors** | Adrian Hill, Neal McKee, Johannes Maeß |
| **Reference** | Hill, McKee, Maeß, *Smoothed Differentiation Efficiently Mitigates Shattered Gradients in Explanations*, NeurIPS 2025 |
| **Upstream repository** | <https://github.com/adrhill/smoothdiff-experiments> |
| **Compared against** | `960cfc6390167a60d294886147a9b35e6a31eb1f` on branch `main` (2026-02-16), file `experiments/cluster/quantus-eval/smoothdiff.py` |
| **Upstream license** | MIT (`LICENSE`, copied verbatim into this folder) |

Authorship confirmed by the PEAL maintainer, who knows the authors, 2026-09-25.

## Why PEAL vendors it

Gradient smoothing for counterfactual generation. `smoothdiff.py` swaps a
network's ReLU and MaxPool layers for custom autograd functions that collect
activation statistics on the forward pass and return a gradient averaged over
noisy samples on the backward pass — SmoothGrad-style smoothing applied inside
the network rather than at the input.

It is imported by `peal/generators/stable_diffusion_3.py`, which offers it as one
of the `gradient_smoothing` options (`distilled`, `vanilla`, `lrp`,
`smoothdiff`), with defaults of 20 samples at noise 0.5.

Upstream publishes research code, not an installable package, which is why it is
vendored rather than declared as a dependency.

## How this copy differs from upstream

The vendored file corresponds to upstream's
`experiments/cluster/quantus-eval/smoothdiff.py` (239 lines) rather than to the
`python/smoothdiff_torch/` package, which is a larger, later variant.

Compared by syntax tree, so that reformatting is separated from behaviour:
**seven of the eight shared top-level definitions are identical** —
`SmoothReLUFunction`, `SmoothMaxPool2dFunction`, `SmoothDiffLayer`,
`SmoothReLU`, `SmoothMaxPool2d`, `set_smoothdiff_layer_mode` and `smooth_layer`.

Differences, all PEAL's:

- `replace_nonlinear_layers` was modified.
- Two helpers were added: `check_supported_layers`, which validates a model
  against the `SUPPORTED_LAYERS` set before smoothing, and `reset_statisticcs`
  (the misspelling is ours), which clears the collected statistics between runs.

Nothing upstream was removed.

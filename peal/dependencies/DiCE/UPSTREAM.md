<!-- Provenance record for a vendored third-party component of PEAL.
     Generated 2026-09-22; keep it updated when the copy is refreshed. -->

# DiCE — Diverse Counterfactual Explanations

**This directory is not part of PEAL.** It is a vendored copy of third-party
research code, kept in `peal/dependencies/` and separate from PEAL's own
implementation. All credit for the method and the original implementation
belongs to the authors below; please cite their work, not PEAL, when you use it.

| | |
|---|---|
| **Original authors** | Ramaravind K. Mothilal, Amit Sharma, Chenhao Tan (Microsoft Research / InterpretML) |
| **Reference** | Mothilal, Sharma, Tan, *Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations*, FAT* 2020 |
| **Upstream repository** | <https://github.com/interpretml/DiCE> |
| **Compared against** | `8a3aea404f857fa599bfa7da663d94823b674688` on branch `main` (2025-07-13) |
| **Upstream license** | MIT |

## Why PEAL vendors it

Tabular counterfactual baseline. Imported as the top-level package `dice_ml` from `peal/explainers/no_generator_counterfactual_explainers.py`; configured by `configs/tabular_experiments/explainers/dice_*.yaml`.

Upstream publishes research code, not an installable package, and the copy here
needed changes to run inside PEAL's pipeline and to keep baseline comparisons
fair. That is why it is forked rather than declared as a dependency.

## How this copy differs from upstream

Measured against upstream `8a3aea404f` (2025-07-13): **57 of 60 Python files are
byte-identical**, 2 differ only in formatting (Black; identical syntax trees),
and 1 carry functional changes.

- `dice_ml/explainer_interfaces/dice_genetic.py` carries a functional change; two further files differ only in formatting.

Any differences listed here are ours, assuming upstream has not changed since
the copy was taken.

## Note

DiCE is also published on PyPI as `dice-ml`. If the local change can be upstreamed or dropped, this fork could be replaced by an ordinary dependency.


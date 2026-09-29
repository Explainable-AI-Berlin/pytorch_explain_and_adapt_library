<!-- Provenance record for a vendored third-party component of PEAL.
     Generated 2026-09-22; keep it updated when the copy is refreshed. -->

# TIME — Text-to-Image Models for Counterfactual Explanations

**This directory is not part of PEAL.** It is a vendored copy of third-party
research code, kept in `peal/dependencies/` and separate from PEAL's own
implementation. All credit for the method and the original implementation
belongs to the authors below; please cite their work, not PEAL, when you use it.

| | |
|---|---|
| **Original authors** | Guillaume Jeanneret, Loïc Simon, Frédéric Jurie (Université de Caen Normandie) |
| **Reference** | Jeanneret, Simon, Jurie, *Text-to-Image Models for Counterfactual Explanations: a Black-Box Approach*, WACV 2024 |
| **Upstream repository** | <https://github.com/guillaumejs2403/TIME> |
| **Compared against** | `3ee4f96f021643576af03d7b2ef3c9096e304bfd` on branch `main` (2023-11-15) |
| **Upstream license** | MIT |

## Why PEAL vendors it

TIME baseline explainer; its textual-inversion training and counterfactual generation are called from PEAL's Stable Diffusion generators.

Upstream publishes research code, not an installable package, and the copy here
needed changes to run inside PEAL's pipeline and to keep baseline comparisons
fair. That is why it is forked rather than declared as a dependency.

## How this copy differs from upstream

Measured against upstream `3ee4f96f02` (2023-11-15): **0 of 12 Python files are
byte-identical**, 6 differ only in formatting (Black; identical syntax trees),
and 5 carry functional changes.

- `training.py` and `get_predictions.py` refactored from scripts into callable functions (`textual_inversion_training`, `get_predictions`, `run_epoch`, `data_loader_val`) driven by PEAL configs.
- `core/utils.py`, `core/phrases.py`, `models/__init__.py` adapted to PEAL's model and checkpoint handling; 6 further files differ only in formatting.
- 2026-09-28: personal absolute paths (colleagues' home directories, cluster scratch) in `training.py`, `generate_ce.py` replaced by defaults derived from `$PEAL_DATA` / `$PEAL_RUNS`; no functional change otherwise.


Any differences listed here are ours, assuming upstream has not changed since
the copy was taken.


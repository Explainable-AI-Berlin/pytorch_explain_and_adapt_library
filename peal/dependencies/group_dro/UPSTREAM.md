<!-- Provenance record for a vendored third-party component of PEAL.
     Generated 2026-09-22; keep it updated when the copy is refreshed. -->

# Group DRO

**This directory is not part of PEAL.** It is a vendored copy of third-party
research code, kept in `peal/dependencies/` and separate from PEAL's own
implementation. All credit for the method and the original implementation
belongs to the authors below; please cite their work, not PEAL, when you use it.

| | |
|---|---|
| **Original authors** | Shiori Sagawa, Pang Wei Koh, Tatsunori B. Hashimoto, Percy Liang (Stanford) |
| **Reference** | Sagawa et al., *Distributionally Robust Neural Networks for Group Shifts*, ICLR 2020 |
| **Upstream repository** | <https://github.com/kohpangwei/group_DRO> |
| **Compared against** | `cbbc1c5b06844e46b87e264326b56056d2a437d1` on branch `master` (2023-01-03) |
| **Upstream license** | MIT |

## Why PEAL vendors it

GroupDRO baseline for model correction, used by `peal/adaptors/group_distributionally_robust_optimization.py` and `peal/training/trainers.py`.

Upstream publishes research code, not an installable package, and the copy here
needed changes to run inside PEAL's pipeline and to keep baseline comparisons
fair. That is why it is forked rather than declared as a dependency.

## How this copy differs from upstream

Measured against upstream `cbbc1c5b06` (2023-01-03): **0 of 9 Python files are
byte-identical**, none differ only in formatting (Black; identical syntax trees),
and 8 carry functional changes.

- `loss.py` gained `get_group_stats` and a `replication` argument so PEAL can log per-group statistics through its own trainer.
- The dataset loaders (`data/*.py`) take explicit arguments (`root_dir`, `dataset`, `target_name`, `confounder_names`, ...) instead of an argparse namespace.
- 12 upstream files (their standalone training scripts and unused datasets) are not vendored.
- `loss.py` already carried an upstream-attribution docstring before this file existed.

Any differences listed here are ours, assuming upstream has not changed since
the copy was taken.


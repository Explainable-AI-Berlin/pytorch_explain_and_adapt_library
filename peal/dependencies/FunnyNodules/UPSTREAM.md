<!-- Provenance record for a vendored third-party component of PEAL.
     Generated 2026-09-22; keep it updated when the copy is refreshed. -->

# FunnyNodules — synthetic medical dataset

**This directory is not part of PEAL.** It is a vendored copy of third-party
research code, kept in `peal/dependencies/` and separate from PEAL's own
implementation. All credit for the method and the original implementation
belongs to the authors below; please cite their work, not PEAL, when you use it.

| | |
|---|---|
| **Original authors** | Luisa Gallée, Yiheng Xiong, Meinrad Beer, Michael Götz (Ulm University, XRad-Ulm) |
| **Reference** | Gallée et al., *FunnyNodules: A Customizable Medical Dataset Tailored for Evaluating Explainable AI*, 2025 |
| **Upstream repository** | <https://github.com/XRad-Ulm/FunnyNodules> |
| **Compared against** | `ac25d4135780e6349d753a6ba56372da1eeea2b4` on branch `main` (2026-01-19) |
| **Upstream license** | MIT |

## Why PEAL vendors it

Synthetic nodule dataset generator, called from `peal/data/dataset_generators.py` and configured by ~68 dataset configs.

Upstream publishes research code, not an installable package, and the copy here
needed changes to run inside PEAL's pipeline and to keep baseline comparisons
fair. That is why it is forked rather than declared as a dependency.

## How this copy differs from upstream

Measured against upstream `ac25d41357` (2026-01-19): **45 of 45 Python files are
byte-identical**, none differ only in formatting (Black; identical syntax trees),
and 0 carry functional changes.

- None. All 45 Python files are byte-identical to upstream.

Any differences listed here are ours, assuming upstream has not changed since
the copy was taken.


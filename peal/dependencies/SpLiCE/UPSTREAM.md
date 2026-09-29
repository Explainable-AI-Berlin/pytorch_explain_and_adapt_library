<!-- Provenance record for a vendored third-party component of PEAL.
     Generated 2026-09-22; keep it updated when the copy is refreshed. -->

# SpLiCE — Sparse Linear Concept Embeddings

**This directory is not part of PEAL.** It is a vendored copy of third-party
research code, kept in `peal/dependencies/` and separate from PEAL's own
implementation. All credit for the method and the original implementation
belongs to the authors below; please cite their work, not PEAL, when you use it.

| | |
|---|---|
| **Original authors** | Usha Bhalla, Alex Oesterling, Suraj Srinivas, Flavio P. Calmon, Himabindu Lakkaraju (AI4LIFE Group, Harvard) |
| **Reference** | Bhalla et al., *Interpreting CLIP with Sparse Linear Concept Embeddings*, NeurIPS 2024 |
| **Upstream repository** | <https://github.com/AI4LIFE-GROUP/SpLiCE> |
| **Compared against** | `9a498102ce7c6701f4afe361ffaa86c39b47fa5f` on branch `main` (2025-03-27) |
| **Upstream license** | Apache-2.0 |

## Why PEAL vendors it

Alternative concept dictionary over CLIP, wrapped by `peal/sparse_dictionaries/splice_decomposition.py` (imported as the top-level package `splice`).

Upstream publishes research code, not an installable package, and the copy here
needed changes to run inside PEAL's pipeline and to keep baseline comparisons
fair. That is why it is forked rather than declared as a dependency.

## How this copy differs from upstream

Measured against upstream `9a498102ce` (2025-03-27): **13 of 14 Python files are
byte-identical**, none differ only in formatting (Black; identical syntax trees),
and 1 carry functional changes.

- `splice/splice.py` carries a functional change (loading the model through `transformers`); the other 13 files are byte-identical.

Any differences listed here are ours, assuming upstream has not changed since
the copy was taken.


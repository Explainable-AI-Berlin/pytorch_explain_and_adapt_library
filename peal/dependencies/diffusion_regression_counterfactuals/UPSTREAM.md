<!-- Provenance record for a vendored third-party component of PEAL.
     Generated 2026-09-22; keep it updated when the copy is refreshed. -->

# Diffusion Counterfactuals for Image Regressors

**This directory is not part of PEAL.** It is a vendored copy of third-party
research code, kept in `peal/dependencies/` and separate from PEAL's own
implementation. All credit for the method and the original implementation
belongs to the authors below; please cite their work, not PEAL, when you use it.

| | |
|---|---|
| **Original authors** | Trung Duc Ha, Sidney Bender (TU Berlin) |
| **Reference** | Ha & Bender, *Diffusion Counterfactuals for Image Regressors*, xAI 2025 |
| **Upstream repository** | <https://github.com/DevinTDHa/Diffusion-Counterfactuals-for-Image-Regressors> |
| **Compared against** | `c7f5b32a67921c296b89342db49516bf9793cdc6` on branch `main` (2025-10-13) |
| **Upstream license** | MIT |

## Why PEAL vendors it

Supplies the DiffAE implementation PEAL's own diffusion autoencoder builds on: `peal/generators/diffusion_autoencoder.py` imports from `src/related_work/diffae`.

Upstream publishes research code, not an installable package, and the copy here
needed changes to run inside PEAL's pipeline and to keep baseline comparisons
fair. That is why it is forked rather than declared as a dependency.

## How this copy differs from upstream

Measured against upstream `c7f5b32a67` (2025-10-13): **10 of 141 Python files are
byte-identical**, none differ only in formatting (Black; identical syntax trees),
and 17 carry functional changes.

- Our copy is a subset of the upstream tree plus PEAL-side additions (114 files not in upstream, 125 upstream files not vendored); 17 shared files carry small functional changes, mostly config and path handling in the `scripts/` entry points.
- 2026-09-28: personal absolute paths (colleagues' home directories, cluster scratch) in the `scripts/` entry points, `src/related_work/ACE/utils/create-mini-val.py` and the ACE regression submit scripts replaced by defaults derived from `$PEAL_DATA` / `$PEAL_RUNS`; no functional change otherwise.


Any differences listed here are ours, assuming upstream has not changed since
the copy was taken.

## Note

Co-authored by the present author; vendored rather than depended upon because PEAL builds directly on its DiffAE implementation.


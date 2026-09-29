<!-- Provenance record for a vendored third-party component of PEAL.
     Generated 2026-09-22; keep it updated when the copy is refreshed. -->

# FastDiME (CelebA benchmark code)

**This directory is not part of PEAL.** It is a vendored copy of third-party
research code, kept in `peal/dependencies/` and separate from PEAL's own
implementation. All credit for the method and the original implementation
belongs to the authors below; please cite their work, not PEAL, when you use it.

| | |
|---|---|
| **Original authors** | Paraskevas Pegios, Nina Weng, Eike Petersen, Aasa Feragen, Siavash Bigdeli |
| **Reference** | Weng et al., *Fast Diffusion-Based Counterfactuals for Shortcut Removal and Generation*, ECCV 2024 |
| **Upstream repository** | <https://github.com/ppegiosk/FastDiME_CelebA> |
| **Compared against** | `6d4a4eba939d3e3678be6ead2347d5351efbc7e4` on branch `main` (2024-09-03) |
| **Upstream license** | MIT |

## Why PEAL vendors it

FastDiME baseline explainer. `peal/generators/ddpm_generator.py` calls `main` and `core.sample_utils.PerceptualLoss`; the Stable Diffusion and FLUX generators reuse its sampling utilities.

Upstream publishes research code, not an installable package, and the copy here
needed changes to run inside PEAL's pipeline and to keep baseline comparisons
fair. That is why it is forked rather than declared as a dependency.

## How this copy differs from upstream

Measured against upstream `6d4a4eba93` (2024-09-03): **4 of 35 Python files are
byte-identical**, 15 differ only in formatting (Black; identical syntax trees),
and 15 carry functional changes.

- Imports rewritten so the code runs as a subpackage of PEAL rather than as a standalone script tree (`peal.*` in, the original standalone eval/plotting imports out).
- `main()` changed from a no-argument script entry point to `main(args)` so PEAL can drive it from a config instead of the command line.
- 15 further files differ only in formatting (Black; identical ASTs).

Any differences listed here are ours, assuming upstream has not changed since
the copy was taken.


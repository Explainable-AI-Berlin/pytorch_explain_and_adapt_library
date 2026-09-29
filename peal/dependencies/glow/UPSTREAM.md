<!-- Provenance record for a vendored third-party component of PEAL.
     Generated 2026-09-22; keep it updated when the copy is refreshed. -->

# glow-pytorch

**This directory is not part of PEAL.** It is a vendored copy of third-party
research code, kept in `peal/dependencies/` and separate from PEAL's own
implementation. All credit for the method and the original implementation
belongs to the authors below; please cite their work, not PEAL, when you use it.

| | |
|---|---|
| **Original authors** | Kim Seonghyeon (rosinality) |
| **Reference** | Implementation of Kingma & Dhariwal, *Glow: Generative Flow with Invertible 1x1 Convolutions*, NeurIPS 2018 |
| **Upstream repository** | <https://github.com/rosinality/glow-pytorch> |
| **Compared against** | `97081ff115a694cf04aeedbd58447f33d242c879` on branch `master` (2020-10-07) |
| **Upstream license** | MIT |

## Why PEAL vendors it

Normalizing-flow generator, wrapped by `peal/generators/glow_generator.py` (`Glow`, `gaussian_log_p`, `training`).

Upstream publishes research code, not an installable package, and the copy here
needed changes to run inside PEAL's pipeline and to keep baseline comparisons
fair. That is why it is forked rather than declared as a dependency.

## How this copy differs from upstream

Measured against upstream `97081ff115` (2020-10-07): **0 of 2 Python files are
byte-identical**, 1 differs only in formatting (Black; identical syntax trees),
and 1 carry functional changes.

- `train.py`: the script body was refactored into a reusable `training(...)` function and `train(...)` gained a `model_single` argument, so PEAL can call it in-process instead of as a script.
- `model.py` differs only in formatting.

Any differences listed here are ours, assuming upstream has not changed since
the copy was taken.


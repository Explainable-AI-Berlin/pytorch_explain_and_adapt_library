<!-- Provenance record for a vendored third-party component of PEAL.
     Generated 2026-09-22; keep it updated when the copy is refreshed. -->

# ACE — Adversarial Counterfactual Visual Explanations

**This directory is not part of PEAL.** It is a vendored copy of third-party
research code, kept in `peal/dependencies/` and separate from PEAL's own
implementation. All credit for the method and the original implementation
belongs to the authors below; please cite their work, not PEAL, when you use it.

| | |
|---|---|
| **Original authors** | Guillaume Jeanneret, Loïc Simon, Frédéric Jurie (Université de Caen Normandie) |
| **Reference** | Jeanneret, Simon, Jurie, *Adversarial Counterfactual Visual Explanations*, CVPR 2023 |
| **Upstream repository** | <https://github.com/guillaumejs2403/ACE> |
| **Compared against** | `6684b93ef2d2d6a594fdbac87db25e3f640340e7` on branch `main` (2025-03-12) |
| **Upstream license** | MIT |

## Why PEAL vendors it

ACE baseline explainer, and the `guided_diffusion` training stack it ships. `peal/generators/ddpm_generator.py` calls `run_ace.main` and reuses `guided_diffusion`'s samplers, schedulers and training loop.

Upstream publishes research code, not an installable package, and the copy here
needed changes to run inside PEAL's pipeline and to keep baseline comparisons
fair. That is why it is forked rather than declared as a dependency.

## How this copy differs from upstream

Measured against upstream `6684b93ef2` (2025-03-12): **3 of 51 Python files are
byte-identical**, 13 differ only in formatting (Black; identical syntax trees),
and 27 carry functional changes.

- Imports rewritten for subpackage use; the standalone CLI/eval entry points are replaced by PEAL's config-driven ones.
- `guided_diffusion/train_util.py`: `run_loop`, `run_step` and `forward_backward` take PEAL's config, TensorBoard writer and progress bar, so training reports into PEAL's run directory.
- 8 files added by us (PEAL-side glue), 1 upstream file dropped, 13 further files differ only in formatting.
- 2026-09-28: personal absolute paths (colleagues' home directories, cluster scratch) in `utils/create-mini-val.py` replaced by defaults derived from `$PEAL_DATA` / `$PEAL_RUNS`; no functional change otherwise.


Any differences listed here are ours, assuming upstream has not changed since
the copy was taken.


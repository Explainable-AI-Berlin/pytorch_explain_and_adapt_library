<!-- Provenance record for a vendored third-party component of PEAL.
     Generated 2026-09-25; keep it updated when the copy is refreshed. -->

# Adversarially trained smoothed classifiers — PGD-L2 and DDN attacks

**This directory is not part of PEAL.** It is a vendored copy of third-party
research code, kept in `peal/dependencies/` and separate from PEAL's own
implementation. All credit for the method and the original implementation
belongs to the authors below; please cite their work, not PEAL, when you use it.

| | |
|---|---|
| **Original authors** | Hadi Salman, Greg Yang, Jerry Li, Huan Zhang, Pengchuan Zhang, Ilya Razenshteyn, Sébastien Bubeck (Microsoft Research) |
| **Reference** | Salman, Li, Razenshteyn, Zhang, Zhang, Bubeck, Yang, *Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers*, NeurIPS 2019 |
| **Upstream repository** | <https://github.com/Hadisalman/smoothing-adversarial> (file `code/attacks.py`) |
| **Compared against** | `cff781d53ed5fa7df6ba567d3b51df202b6ec14e` on branch `master` (2019-11-09); `code/attacks.py` itself last changed in `3911bd8a499e0d2668910d9dba124389af8db405` (2019-06-13) |
| **Upstream license** | MIT (`LICENSE`, copied verbatim into this folder) |

The `DDN` attack it contains implements Rony, Hajri, Granger, Ben Ayed and
Pedersoli, *Decoupling Direction and Norm for Efficient Gradient-Based L2
Adversarial Attacks and Defenses*, CVPR 2019. Cite that paper when you use
`DDN`.

## Why PEAL vendors it

`PGD_L2` is the adversarial attack used for adversarial training. It is imported
by `peal/training/trainers.py` (the `adv_training` option of `ModelTrainer`) and
by both GroupDRO adaptors,
`peal/adaptors/group_distributionally_robust_optimization.py` and
`peal/adaptors/peal_group_distributionally_robust_optimization.py`. Those imports
are at module level, so this folder is required to import the trainer at all,
not only when adversarial training is switched on.

Upstream publishes research code, not an installable package, which is why it is
vendored rather than declared as a dependency.

## How this copy differs from upstream

Compared by syntax tree against upstream `cff781d53e`, so that reformatting is
separated from behaviour: the file defines the same three classes
(`Attacker`, `PGD_L2`, `DDN`) with the same ten methods in the same order, and
**nine of the ten are identical in their syntax tree** — the textual differences
there are Black formatting only (wrapped signatures, double quotes, whitespace).

`PGD_L2._attack` is the one method with a functional change:

- the "input values should be in the [0, 1] range" `ValueError` now reports the
  observed range;
- adding the smoothing noise (`adv = adv + noise`) is wrapped in a
  `try` / `except Exception` that calls `pdb.set_trace()`.

**The debugger call is a landmine.** In an unattended run it turns any failure
there into `BdbQuit`, killing the process, and because it sits in an exception
handler it also hides the original error. PEAL's own pre-commit hook
(`tools/no_debugger_calls.py`) exists to prevent exactly this but excludes
`peal/dependencies/`, so it is not caught. It should be replaced by letting the
exception propagate, or by a logged error.

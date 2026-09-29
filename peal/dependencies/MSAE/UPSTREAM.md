<!-- Provenance record for a vendored third-party component of PEAL.
     Generated 2026-09-22; keep it updated when the copy is refreshed. -->

# MSAE — Hierarchical (Matryoshka) Sparse Autoencoders for CLIP

**This directory is not part of PEAL.** It is a vendored copy of third-party
research code, kept in `peal/dependencies/` and separate from PEAL's own
implementation. All credit for the method and the original implementation
belongs to the authors below; please cite their work, not PEAL, when you use it.

| | |
|---|---|
| **Original authors** | Vladimir Zaigrajew, Hubert Baniecki, Przemyslaw Biecek |
| **Reference** | Zaigrajew, Baniecki, Biecek, *Interpreting CLIP with Hierarchical Sparse Autoencoders*, 2025 |
| **Upstream repository** | <https://github.com/WolodjaZ/MSAE> |
| **Compared against** | `51a5f03b3cb629176aba3e825840c72e3c27bd48` on branch `main` (2026-01-17) |
| **Upstream license** | MIT |

## Why PEAL vendors it

Supplies the public 6,144-atom MSAE dictionary over OpenAI CLIP ViT-L/14 used for the ImageNet and NICO++ experiments. Loaded by `peal/sparse_dictionaries/msae_decomposition.py`.

Upstream publishes research code, not an installable package, and the copy here
needed changes to run inside PEAL's pipeline and to keep baseline comparisons
fair. That is why it is forked rather than declared as a dependency.

## How this copy differs from upstream

Measured against upstream `51a5f03b3c` (2026-01-17): **10 of 12 Python files are
byte-identical**, none differ only in formatting (Black; identical syntax trees),
and 2 carry functional changes.

- `utils.py` and `sae.py` carry small functional changes; the remaining 10 files are byte-identical.
- The loader in PEAL, not this folder, works around MSAE's bare top-level imports (`from utils import ...`) by importing its modules under private `sys.modules` keys.

Any differences listed here are ours, assuming upstream has not changed since
the copy was taken.


<!-- Provenance record for a vendored third-party component of PEAL.
     Generated 2026-09-22; keep it updated when the copy is refreshed. -->

# Edit-friendly DDPM inversion

**This directory is not part of PEAL.** It is a vendored copy of third-party
research code, kept in `peal/dependencies/` and separate from PEAL's own
implementation. All credit for the method and the original implementation
belongs to the authors below; please cite their work, not PEAL, when you use it.

| | |
|---|---|
| **Original authors** | Inbar Huberman-Spiegelglas, Vladimir Kulikov, Tomer Michaeli (Technion) |
| **Reference** | Huberman-Spiegelglas, Kulikov, Michaeli, *An Edit Friendly DDPM Noise Space: Inversion and Manipulations*, CVPR 2024 |
| **Upstream repository** | <https://github.com/inbarhub/DDPM_inversion> |
| **Compared against** | `58fc881772d34f0c24be9e34725d213731aa009d` on branch `main` (2024-07-11) |
| **Upstream license** | MIT |

## Why PEAL vendors it

The inversion behind every DDPM-inverted result in the thesis. Imported from `peal/generators/diffusion_autoencoder.py` and `peal/generators/pathldm_autoencoder.py`.

Upstream publishes research code, not an installable package, and the copy here
needed changes to run inside PEAL's pipeline and to keep baseline comparisons
fair. That is why it is forked rather than declared as a dependency.

## How this copy differs from upstream

Measured against upstream `58fc881772` (2024-07-11): **1 of 9 Python files are
byte-identical**, 1 differs only in formatting (Black; identical syntax trees),
and 6 carry functional changes.

- `inversion_forward_process` and `inversion_reverse_process` gained an `encoder_hidden_states` argument, plus classifier/guidance arguments (`f`, `classifier`, `classifier_scale`), so the stored noise maps can be replayed under an *edited* semantic conditioning and under classifier guidance.
- Added PathLDM variants of the forward/reverse/variance steps (`inversion_forward_process_pathldm`, `reverse_step_pathldm`, ...) for the latent-diffusion decoder.
- `prompt_to_prompt` utilities adapted accordingly; 1 file byte-identical, 1 formatting-only.

Any differences listed here are ours, assuming upstream has not changed since
the copy was taken.


# Repository layout

The project is split into three ownership boundaries.

## ADA core

`src/ada`, `configs/ada`, `scripts/ada`, and `tests/ada` implement semantic
atlases, support metrics, controlled deletion/restoration, cross-representation
evaluation, and language-space interpretation.

## Generator integration

`third_party/RAEv2` is a clean pinned upstream submodule. The files under
`integrations/raev2` describe and apply the exact development changes used for
the current CLS-conditioned generator experiments. Keeping this boundary
explicit prevents unrelated RAE experiments from entering the ADA history.

## Data and compute state

Datasets, embeddings, patch-latent caches, checkpoints, Slurm logs, and generated
samples are external state. They are referenced through environment variables
and provenance metadata, never copied into Git.

The original RAE working tree remains the active laboratory. This
repository is the curated collaboration and release surface; code should flow
from the laboratory into this repository through reviewed commits rather than
by moving the laboratory itself.

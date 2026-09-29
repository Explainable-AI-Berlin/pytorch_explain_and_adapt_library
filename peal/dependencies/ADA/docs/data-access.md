# Data and artifact access

No ImageNet-derived data are stored in Git.

On Hydra, the current collaborator handoff is:

```text
/home/space/pathomics/ADA
```

It is group-readable and group-writable by members of `pathomics`, including
`daviddrexlin` and `lciernik`, and is not world-accessible.

Download it from a local machine with:

```bash
rsync -ah --partial --info=progress2 \
  hydra:/home/space/pathomics/ADA/ \
  ./ADA-data/
```

The ImageNet-1K CLS statistics are under:

```text
artifacts/generator/in1k_dinov2l_cls_statistics_20260817/
```

The bundle contains 1,281,167 raw DINOv2-L CLS vectors with 1,024 dimensions,
plus class labels, stable source indices, extraction views, norms, and train
coordinate statistics. The exported array is float32 but originates from a
bf16 cache. It is not standardized, PCA-projected, or L2-normalized.

The full Stage-2 training cache also contains spatial patch latents with shape
`[1024, 16, 16]` per image. Those patch latents are intentionally not included
in the collaborator bundle because of their size.

The three compact generator snapshots are under:

```text
models/latest/
```

They contain epoch-33 EMA weights for the CLS-only, class+CLS, and class-only
models. The directory-level manifest records SHA256 hashes and the exact
checkpoint step. Full optimizer-bearing checkpoints remain outside the shared tree.

The validated CA8 cross-attention EMA bundle is separate:

```text
models/ca8_latest/
```

It contains the epoch-58-complete `ep-0000059.pt` boundary EMA, exact CA8
configuration, manifest hashes, and hard-linked copies of the pinned decoder
and latent statistics.

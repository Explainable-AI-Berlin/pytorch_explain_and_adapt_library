# RAEv2 generator models

All current generator, FID, edit, and synthetic-repair results documented in
this repository use the native RAEv2 training and sampling pipeline. The
configuration target `stage1.RAE` names a model class within RAEv2; it does not
mean that those experiments used the older RAE implementation.

## What is versioned

Git contains the three exact model configurations, the ADA changes to pinned
RAEv2, the parallel training launcher, the compact model exporter, and a unified
sampler for all three matched conditioning variants plus the CA8 cross-attention
ablation.

The multi-gigabyte weights remain external artifacts. Putting them in ordinary
Git history would make every clone permanently carry every checkpoint version.

## Prepare the RAEv2 tree

```bash
git submodule update --init
bash integrations/raev2/apply_overlay.sh
```

Download the pinned DINOv2-L K=1 decoder and normalization statistics:

```bash
python scripts/generator/download_raev2_dinov2l_stage1.py \
  --output-root third_party/RAEv2/pretrained_models
```

The downloader verifies both files against checksums recorded in the model
bundle.

## Environment

Copy `.env.example` to `.env` and set:

```bash
ADA_CONTAINER=/path/to/rae-container.sif
ADA_STAGE2_CACHE=/path/to/premixed-stage2-cache
ADA_CHECKPOINT_ROOT=/path/to/raev2-stage2-checkpoints
ADA_MODEL_BUNDLE_ROOT=/home/space/pathomics/ADA/models/latest
ADA_CLS_STATS_ROOT=/home/space/pathomics/ADA/artifacts/generator/in1k_dinov2l_cls_statistics_20260817
```

## Sample a model

Submit one GPU job:

```bash
bash scripts/generator/submit_raev2_sample.sh cls_only
bash scripts/generator/submit_raev2_sample.sh cls_class
CLASS_IDS=0,1,2,3 bash scripts/generator/submit_raev2_sample.sh class_only
ADA_MODEL_BUNDLE_ROOT=/home/space/pathomics/ADA/models/ca8_latest bash scripts/generator/submit_raev2_sample.sh cls_ca8
```

By default the CLS variants use rows `0,1,2,3` from the shared raw CLS array.
Override them without editing code:

```bash
INDICES=100,200,300,400 SAMPLES_PER_CONDITION=4 \
  bash scripts/generator/submit_raev2_sample.sh cls_class
```

Each sampling job writes individual PNGs, a grid, and `metadata.json`. The
metadata records the checkpoint epoch and step, seeds, source CLS row indices,
class IDs, and re-encoded CLS adherence when applicable.

### Checkpoint provenance for reported metrics

Do not compare a conditioning label without its checkpoint:

| Reported result | Frozen checkpoint |
| --- | --- |
| Class-only FID50k `4.3116` | additive epoch 31 |
| CLS-only FID50k `3.3163` | additive epoch 31 |
| Class+CLS FID50k `3.3485` | additive epoch 31 |
| Validation-CLS oracle FID50k `2.3243` | additive CLS-only epoch 31 |
| CA8 FID50k `3.3290` | CA8 epoch 31 |
| CA8 FID50k `3.1485` | CA8 epoch 58 |
| Canonical six-region synthetic repair | CA8 epoch 31 |
| Additive graph/edit mechanism panels | additive CLS-only epoch 31/33 boundary as recorded by each artifact |

The validation-CLS row supplies each validation image's own condition. It is an
oracle reconstruction-style diagnostic, not a deployable unconditional FID.

### Which model to use

- **CLS-only AdaLN, epoch 33:** primary model for Gaussian wiggles, direct
  SLERP, graph-geodesic paths, and fine-grained condition-fidelity analysis.
- **Class-only, epoch 33:** matched broad class-conditional baseline.
- **Class+CLS, epoch 33:** dual-condition ablation for class stabilization.
- **CLS-only CA8, epoch-58-complete boundary:** stability and conditioning-route
  ablation; eight learned CLS-derived tokens enter gated cross-attention every
  fourth block.

CA8 remained finite through the additive-AdaLN collapse window. Its completed
epoch-58 balanced ImageNet-1K evaluation reached FID50k `3.1485`; its
matched epoch-31 value was `3.3290`, compared with `3.3163`
for epoch-31 additive CLS-only. The epoch-58 result makes CA8 the strongest
completed stable global-distribution checkpoint, not proof that cross-attention
is intrinsically better at matched compute.

The canonical six-region repair screen used CA8 epoch 31. Re-evaluation at
epoch 58 increased exact-CLS target-CE repair only from `0.2637` to
`0.2783`; direct SLERP was essentially unchanged, and graph repair
remained worse than direct SLERP. Additive CLS-only remains the source of the
reported graph-path and fine-grained edit panels. See
[the CA8 downstream report](ca8-downstream-result.md).

The CA1 one-token cross-attention run is not recommended. At epoch 39 it had
nearly identical re-encoded CLS adherence to CA8 (`0.8572` versus `0.8553`),
slightly lower fixed-condition diversity (`0.1454` versus `0.1559`), and no
robust paired fidelity advantage. It later became unstable and was cancelled.
With one context key, cross-attention weights are identically one, so CA1 acts
as a learned broadcast residual rather than content-selective attention. See
[the conditioner frontier](model-training-frontier-2026-09-07.md) for the
matched audit and the remaining model-training question.

### Sample from a wiggled CLS condition

For `cls_only` or `cls_class`, apply a reproducible isotropic Gaussian
perturbation directly to each raw 1024-dimensional DINOv2-L CLS row:

```bash
INDICES=100,200,300,400 \
CLS_NOISE_SIGMA=0.1 CLS_NOISE_SEED=7 \
SAMPLES_PER_CONDITION=4 \
  bash scripts/generator/submit_raev2_sample.sh cls_only
```

The perturbation direction is deterministic per source index and independent of
the diffusion `SEED`. By default the perturbed vector is rescaled to preserve
the original CLS norm. `metadata.json` records the input cosine and L2
displacement, both CLS norms, and the generated image re-encoded cosine to the
perturbed and original conditions. A nonzero CLS noise value is rejected for
`class_only` because that model has no CLS input.

This is a local sensitivity and generation diagnostic, not a guarantee that an
isotropic perturbation follows a semantic data-manifold direction. Compare it
with `CLS_NOISE_SIGMA=0` under the same diffusion seed, and prefer supported
real-real interpolation or local tangent directions for scientific claims.

### Sample graph-geodesic CLS conditions

Build a paired direct-SLERP and same-class graph-geodesic condition bank from
the shared raw CLS geometry:

```bash
python scripts/generator/build_cls_graph_geodesic_condition_bank.py \
  --source-manifest /path/to/frozen_source_manifest.json \
  --geometry-bundle "${ADA_CLS_STATS_ROOT}" \
  --output-dir outputs/graph_conditions \
  --use-all-sources
```

The graph is a symmetric union-kNN graph over real same-class CLS rows with
angular edge lengths. Dijkstra selects real atlas vertices; requested progress
positions are then resampled by piecewise SLERP along those local edges. Thus
the graph vertices are real conditions, while most reported intermediate
conditions are interpolated coordinates.

The builder writes `conditions.float32.npy` and an aligned `y.int16.npy`.
Point the existing sampler at those arrays without changing code:

```bash
CLS_ARRAY="$PWD/outputs/graph_conditions/conditions.float32.npy" \
LABEL_ARRAY="$PWD/outputs/graph_conditions/y.int16.npy" \
INDICES=0,1,2,3,4,5,6 \
SAMPLES_PER_CONDITION=2 \
  bash scripts/generator/submit_raev2_sample.sh cls_only
```

`records.json` identifies each row's path mode, progress, endpoints, graph
vertices, class, and support distance. Direct and graph paths use identical
endpoints, which is required for a fair comparison.

## Export the newest immutable snapshots

The training jobs write full checkpoints containing model, EMA, optimizer, and
scheduler state. Export one compact EMA snapshot per variant with:

```bash
sbatch scripts/generator/run_export_raev2_model_bundle.sh
```

The exporter selects the highest numbered immutable `ep-*.pt` checkpoint. It
never reads the moving `ep-last.pt` alias.

## Train all three variants

Once the premixed latent cache and Stage-1 assets are available:

```bash
bash scripts/generator/submit_raev2_three_variants.sh
```

The three seven-day, four-GPU jobs are submitted independently. They differ only
in conditioning:

| Variant | Timestep | Class | DINOv2-L CLS |
| --- | --- | --- | --- |
| `cls_only` | yes | no | yes |
| `cls_class` | yes | yes | yes |
| `class_only` | yes | yes | no |

The CLS is projected and added to the DiT AdaLN conditioning vector. It is not
concatenated to every patch token.

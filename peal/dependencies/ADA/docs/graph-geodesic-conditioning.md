# Graph-Geodesic CLS Conditioning

**Frozen:** 2026-08-27
**Resolved:** 2026-09-08
**Status:** all mechanism, CFG, memorization, regional-distribution, and
downstream-repair sidecars complete

## Construction

For each region and deletion seed:

- graph vertices are retained q=.25 target-region examples only;
- deleted examples are never vertices or generator conditions;
- direct SLERP and graph paths use identical endpoint pairs;
- a symmetric union-kNN graph starts at k=8 and expands deterministically only
  if required for connectivity;
- angular CLS distance defines edge length;
- Dijkstra chooses a sequence of real atlas vertices;
- requested path positions are resampled by piecewise SLERP at matched
  arc-length progress.

The path endpoints and Dijkstra vertices are real CLS tokens. Most displayed
intermediate coordinates are interpolated points on edges, not observed tokens.

## What Was Tested

1. Direct versus graph paths across CFG scales `{0, .5, 1, 2, 4}`.
2. Matched q=.25, 50%-of-missing synthetic repair over six sparse regions.
3. Generated-versus-deleted regional feature distributions.
4. Exact-source and endpoint retrieval in independent DINOv3 and SigLIP2
   feature spaces.

All submitted graph jobs completed, including `4778513_[0-4]`,
`4778514`, `4778522`, and `4779693`. Earlier
language describing CFG or memorization jobs as queued is superseded.

## Mechanism Result

At CLS CFG scale 1, graph minus direct SLERP gives:

| Metric | Graph minus direct | Direction |
| --- | ---: | --- |
| Generated same-class support distance | -1.5689 | graph is closer to empirical support |
| Re-encoded condition adherence | +0.0431 | graph follows its requested condition better |
| Endpoint/source lock fraction | -0.2792 | graph is less endpoint-locked |

The original 24-source mechanism pilot gave the same qualitative pattern:
support distance `-1.5121`, adherence `+0.0459`, and
endpoint locking `-0.2708`.

This is a real condition-geometry effect.

## Downstream Result

Graph paths do not improve ADA repair in either completed generator bank.

### Original additive-AdaLN epoch-31 graph experiment

| Added data | Target CE repair |
| --- | ---: |
| Class-only synthetic | 0.0841 |
| Graph-geodesic synthetic | 0.2152 |
| Retained-anchor repetition | 0.2570 |
| Exact retained CLS synthetic | 0.2791 |
| Direct SLERP synthetic | 0.2896 |
| Unique target real reference | 0.3201 |

Graph minus direct SLERP is `-0.0745` with CI
`[-0.1522, -0.0168]`. Graph is also worse by
`+0.0109` FD-DINO-RP64 and `+0.0077` RBF MMD against
deleted regional data.

### Later CA8 epoch-31 bank

| Added data | Target CE repair |
| --- | ---: |
| Class-only synthetic | 0.0572 |
| Graph-geodesic synthetic | 0.1991 |
| Retained-anchor image augmentation | 0.2312 |
| Direct SLERP synthetic | 0.2401 |
| Retained-anchor repetition | 0.2570 |
| Exact retained CLS synthetic | 0.2638 |
| Unique target real reference | 0.3201 |

Here graph minus direct SLERP is `-0.0410` with CI
`[-0.1080, -0.0035]`. The point estimates differ because these are
separately sampled additive-AdaLN and CA8 banks, not duplicate analyses of one
image set. Their conclusion agrees.

Positive repair contrasts would favor graph synthesis; both are negative.
Better support and lower endpoint locking therefore do not imply better
decision-relevant augmentation.

## Matched Repair Design

Every arm starts from the same q=.25 deleted state and adds 50% of the missing
regional count. The original additive-AdaLN experiment is self-contained below:

| Training state or addition | Target accuracy | Accuracy repair | Target CE | CE repair | Global balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| q=.25 deleted, no additions | 66.02% | 0.00 pp | 1.327 | 0.000 | 95.268% |
| Class-only synthetic | 69.52% | +3.50 pp | 1.243 | +0.084 | 95.287% |
| Direct retained-neighbor SLERP | 70.42% | +4.40 pp | 1.038 | +0.290 | 95.290% |
| Retained-anchor graph path | 71.13% | +5.11 pp | 1.112 | +0.215 | 95.290% |
| Exact retained-anchor CLS | 71.83% | +5.81 pp | 1.048 | +0.279 | 95.293% |
| Stochastic retained-anchor augmentation | 70.74% | +4.72 pp | 1.096 | +0.231 | 95.289% |
| Unique target real reference | 71.79% | +5.77 pp | 1.007 | +0.320 | 95.277% |
| Full q=1 reference | 75.29% | +9.27 pp | 0.869 | +0.458 | 95.269% |

Repair is relative to the q=.25 starting state. The full q=1 row restores all
deleted examples and is neither an equal-budget arm nor a mathematical upper
bound. Tiny global changes are expected because each separately trained probe
modifies one local region in one class.

The six regions are four sparse interiors and two sparse boundaries from the
frozen IN100 pilot. Spherical K=10 clustering used class-centered DINOv2-Base
CLS residuals. Selection used training-only same-class k=50 support, global
neighbor purity/entropy, robust class margin, and validation count only as an
evaluability floor. It did not use validation errors or intervention outcomes.

Across all six regions, 224 of 908 members were retained and 684 were deleted.
Each individual intervention deleted 81-137 images, or 0.064%-0.108% of
IN100. The six regions were never deleted simultaneously.

Exact retained-anchor CLS means an unmodified DINOv2-L vector sampled with
replacement from the retained 25% of that target region. No deleted vector is a
condition. The full-IN1K Stage-2 generator had nevertheless seen those
identities during its original training, which is why this remains a capability
screen.

## Completed Job Ledger

| Job | Role |
| --- | --- |
| `4778513_[0-4]` | direct/graph path by CFG scale |
| `4778514` | CFG factorial aggregate |
| `4778515` | retained-only condition banks and integrity checks |
| `4778516` | additive epoch-31 image banks |
| `4778517` | DINOv2-B feature extraction |
| `4778518` | independent ResNet50 validity |
| `4778519_[0-17]` | matched downstream probes |
| `4778520` | repair contrasts and region bootstrap |
| `4778521` | regional FD-DINO/MMD/coverage |
| `4778522` | source retrieval and diversity |
| `4779693` | combined decision report |

The condition-bank integrity gate contained 1,026 rows per synthetic bank and
used graph k=8 in all 18 region/seed tasks without connectivity fallback.

## Memorization And Retrieval

DINOv3/SigLIP exact-source-at-1 rates are approximately:

| Condition | DINOv3 | SigLIP2 |
| --- | ---: | ---: |
| Exact CLS | 0.8489 | 0.5370 |
| Direct SLERP | 0.5604 | 0.3012 |
| Graph path | 0.2924 | 0.1725 |
| Class only | approximately 0 | approximately 0 |

Graph conditioning materially reduces source retrieval. That is useful for
studying source locking, but lower retrieval is not synonymous with useful
novel coverage.

## Interpretation

Let the empirical condition atlas approximate an unknown manifold. Direct
SLERP follows an ambient chord, while the graph estimator follows local edges
through observed conditions. The graph can therefore keep candidate
conditions closer to empirical support.

The experiments falsify the stronger shortcut:

> More supported generated conditions necessarily provide more useful
> downstream repair.

The likely reasons include accumulated path drift, averaging through
intermediate submodes, and reduced targeting of the specific missing regional
direction. These are hypotheses, not established mechanisms.

## Decision

- Keep graph geodesics as a completed geometry and replay diagnostic.
- Use direct same-region SLERP as the current interpolation baseline.
- Do not launch broader graph-k, CFG, or path-density sweeps.
- Revisit graph paths only if a future retained-only selector defines an
  explicit acquisition target that graph sampling can be scored against before
  downstream training.

## Provenance

Mechanism ledger:
`artifacts/cls_conditioning/in1k_ep31_graph_geodesic_cfg_factorial_20260827/submission.tsv`

Repair ledger:
`artifacts/ada/synthetic_repair/in100_q025_ep31_graph_geodesic_20260827/submission.tsv`

The scientific unit in the repair analysis is the region, not each generated
image, diffusion seed, or downstream fit.

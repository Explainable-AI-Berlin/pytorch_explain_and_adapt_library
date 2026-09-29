# ADA Project Status

**Snapshot:** 2026-09-07

**Historical:** superseded by
[the canonical 2026-09-08 status](project-status-2026-09-08.md). Live-job
entries below are preserved as historical provenance.

## Central Question

ADA asks which local parts of a labelled data distribution would benefit from
additional distinct training examples. The intended chain is:

1. identify weakly supported within-class regions in a semantic atlas;
2. show that deleting local support causes localized downstream damage;
3. show that restoring unique target-region data repairs the damage better
   than equal-count data elsewhere;
4. generate useful target-region variation that beats matched repetition or
   ordinary augmentation.

The first three steps have positive evidence. The fourth remains provisional.

## Established Results

### Representation support and failure

| Support atlas | DINO linear error AUROC | ResNet18 error AUROC |
| --- | ---: | ---: |
| DINOv2 CLS, same-class k=50 | 0.8971 | 0.8304 |
| SD-VAE variants | approximately 0.54 to 0.59 | approximately 0.56 to 0.58 |

The DINO and VAE sparse top-decile sets overlap at Jaccard approximately
`0.0549`, essentially the independent-ranking expectation `0.0526`.
Arbitrary latent sparsity is therefore not sufficient.

### Causal deletion and real restoration

The ImageNet-100 pilot closes the local intervention loop over eight frozen
DINOv2 regions. Regional deletion damages target performance; count-preserving
removal remains damaging; same-class random deletion is much smaller; and
unique target-region restoration repairs more than same-class non-target data.

The same immutable interventions transfer to DINOv3 and SigLIP2 feature heads.
For sparse interiors, target CE damage is `+0.2207` in DINOv3 and `+0.1287`
in SigLIP2. Unique target restoration recovers approximately `0.2211` and
`0.1289`, while non-target restoration is near zero.

The clean 30-class ImageNet-1K confirmation excludes the complete ImageNet-100
development class set. It reproduces regional deletion, count-preserving
damage, and location-specific unique-real restoration. Local exposure
baselines recover much of the top-1 effect; unique data retains a modest CE
advantage, so a broad diversity advantage is not yet established.

### Scratch image learner

A randomly initialized ResNet18 was trained directly from pixels for four
sparse interiors and two sparse boundaries:

| Intervention | Mean target CE change |
| --- | ---: |
| Regional deletion | +0.5845 |
| Same-class random deletion | +0.0758 |
| Count-preserving regional removal | +0.9385 |

Target error rises by `+0.1187`, margin falls by `-1.6664`, and five of six
regions have positive target-CE damage. This is evidence beyond frozen
foundation-model heads, but it is one training seed and needs replication.

### CLS-conditioned generators

All results in this section use the native RAEv2 pipeline with the official
DINOv2-L K=1 decoder and normalization statistics. The internal configuration
target named `stage1.RAE` is a class inside RAEv2 and should not be confused
with the earlier RAE codebase.

Matched ImageNet-1K FID50k:

| Conditioning | FID50k |
| --- | ---: |
| Class only | 4.312 |
| Additive CLS only | 3.316 |
| Additive class + CLS | 3.349 |
| CLS only with validation CLS conditions | 2.324 |
| CA8 CLS only | 3.149 |

Validation CLS is an oracle condition and not a deployable unconditional FID
comparison. Exact CLS is highly source-centered. Direct SLERP and graph paths
reduce endpoint locking, while stronger CLS guidance raises adherence and
source retrieval but lowers diversity.

### Conditioner stability and CA1 closure

CA1 was cancelled after late instability and a matched epoch-39 comparison
against CA8. Across 24 exact-CLS conditions and two seeds, CA1 versus CA8 gave
mean re-encoded condition cosine `0.8572` versus `0.8553`, source-at-1
`85.42%` versus `87.50%`, and fixed-condition diversity `0.1454` versus
`0.1559`. CA1's tiny mean-cosine advantage was not robust: its paired median
difference was negative and it won only 18 of 48 images. It therefore provides
no material fidelity advantage that compensates for its instability.

The current evidence covers additive AdaLN, CA1, CA8, and running SA1 and
bounded-AdaLN variants. The most distinct remaining training hypothesis is to
coarsen the condition itself: train with high-purity same-class neighborhood
conditions or a region token plus bounded residual, instead of fitting only
one exact CLS-to-image pair. See
[the conditioner frontier](model-training-frontier-2026-09-07.md).

### Matched six-region synthetic repair

All methods start from the same q=.25 regional deletion and add 50% of the
missing count.

| Added data | Target CE repair |
| --- | ---: |
| Class-only synthetic | 0.0572 |
| Retained-anchor image augmentation | 0.2312 |
| Retained-anchor repetition | 0.2570 |
| CA8 exact CLS synthetic | 0.2638 |
| CA8 direct SLERP synthetic | 0.2401 |
| CA8 graph path synthetic | 0.1991 |
| Unique target real | 0.3201 |

Targeted CLS synthesis clearly beats generic class-only synthesis. It does not
yet establish a reliable advantage over matched anchor augmentation or
repetition. The current generator was trained on full ImageNet-1K, so this is a
capability and replay diagnostic rather than a clean acquisition claim.

### Whole-dataset q=.75 experiment

Retaining 75% independently within every ImageNet-100 class changes global
balanced accuracy only from `95.2933%` to `95.0400%`. In each class weakest
support region, exact CLS and same-region SLERP beat class-only and random
same-class CLS synthesis by roughly `0.8 to 0.9` points, but do not beat
retained-anchor repetition.

Local PCA directions are safer than equal-radius isotropic perturbations at a
large empirical radius. At small radius the difference disappears and
endpoint-nearest replay rises toward 77%. Same-region SLERP remains the best
current validity, localization, and replay compromise.

## Claims Not Yet Established

- targeted synthetic data broadly beats matched stochastic augmentation;
- synthetic repair is not replay from full ImageNet generator training;
- an incomplete dataset can prospectively discover missing regions without
  consulting artificially withheld examples;
- scratch-model transfer replicates across training seeds and architectures;
- the region concept transfers to a dataset with known subpopulations or a
  pretraining-disjoint domain.

## Current High-Value Next Steps

1. Replicate the scratch ResNet18 deletion result across seeds.
2. Run the frozen 25%-total class-imbalance factorial in
   [class-imbalance-25pct-protocol.md](class-imbalance-25pct-protocol.md).
3. Add a BREEDS known-subpopulation benchmark.
4. Select one stable CLS-conditioning architecture, then train one
   region-excluded or cross-fitted generator.
5. Run the prospective synthetic-repair comparison using retained data only.
6. If one additional conditioner model is justified, test neighborhood/region
   conditioning with a stable CA8 route rather than another minor injection
   variant.

## Active Long Runs At This Snapshot

- `4788862`: CLS SA1 prefix conditioning; latest completed epoch 29, loss
  `0.9093`, finite; scheduled end 2026-09-11 07:56 CEST.
- `4789176`: bounded, geometry-regularized additive CLS conditioning; latest
  completed epoch 19, loss `0.9419`, finite with stable geometry telemetry;
  scheduled end 2026-09-12 05:12 CEST.
- `4736515`: CLS CA1 conditioning; intentionally cancelled on 2026-09-07 after
  the matched audit.

Refresh the scheduler before making any current-state claim. Job identifiers
are provenance, not independent scientific units.

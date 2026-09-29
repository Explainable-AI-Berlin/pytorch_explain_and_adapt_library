# CA8 Downstream Repair Result

**Reconciled:** 2026-08-31 09:36 CEST  
**Artifact:** `artifacts/ada/synthetic_repair/in100_q025_ep31_ca8_20260829/`

## Verdict

The epoch-31 eight-token cross-attention (CA8) generator preserves targeted
regional utility and clearly outperforms generic class-only or random
same-class synthesis. It does **not** establish a general advantage over
repeated exposure to the retained target anchors. The only tentative advantage
over anchor repetition occurs in sparse interiors; sparse boundaries favor
repetition and unique real data.

CA8 also remains source-centered. Exact-CLS samples retrieve their conditioning
source at high rates and have much less fixed-condition diversity than
class-only samples. Graph-geodesic conditions reduce exact endpoint retrieval
and improve independent class validity, but do not improve downstream repair
over direct SLERP.

## Matched Design

The comparison freezes six DINOv2-defined ImageNet-100 regions, three deletion
seeds, the q=.25 deleted state, a 50% restoration budget, retained-anchor-only
conditions, labels, probe training, and diffusion seeds. The scientific unit is
the region/class after averaging seeds.

Completed chain:

| Job(s) | Role |
|---|---|
| `4798614` | CA8 sampling |
| `4798615` | DINOv2-B generated-feature extraction |
| `4798616` | independent ResNet-50 validity |
| `4798617_[0-17]` | matched regional probes |
| `4798618` | region-level repair analysis |
| `4798619` | DINOv3/SigLIP2/LPIPS memorization and diversity audit |

## Target Cross-Entropy Repair

Positive values mean improvement from the q=.25 deleted state.

| Added data | All six | Sparse interiors | Sparse boundaries |
|---|---:|---:|---:|
| Retained-anchor oversampling | 0.2570 | 0.1432 | 0.4845 |
| Same-class non-target real | -0.0095 | | |
| Class-only synthetic | 0.0572 | 0.0602 | 0.0511 |
| Random same-class CLS synthetic | 0.0206 | 0.0380 | -0.0143 |
| CA8 exact-CLS synthetic | 0.2638 | 0.1880 | 0.4154 |
| CA8 direct-SLERP synthetic | 0.2401 | 0.1547 | 0.4109 |
| CA8 graph-path synthetic | 0.1991 | 0.1483 | 0.3005 |
| Unique target real | 0.3201 | 0.1972 | 0.5659 |
| Full real restoration | 0.4580 | | |

Key paired region-bootstrap contrasts:

| Contrast | Mean CE-repair difference | Region-bootstrap interval |
|---|---:|---:|
| Exact CLS - oversampling, all | +0.0068 | [-0.0465, +0.0601] |
| Exact CLS - oversampling, sparse interior | +0.0448 | [-0.0012, +0.0907] |
| Exact CLS - oversampling, sparse boundary | -0.0691 | [-0.0896, -0.0487] |
| Exact CLS - class-only, all | +0.2066 | [+0.0229, +0.4603] |
| SLERP - oversampling, all | -0.0169 | [-0.0598, +0.0148] |
| SLERP - oversampling, sparse interior | +0.0115 | [-0.0008, +0.0238] |
| SLERP - oversampling, sparse boundary | -0.0736 | [-0.1178, -0.0293] |
| SLERP - class-only, all | +0.1829 | [+0.0214, +0.4181] |
| SLERP - random same-class CLS, all | +0.2195 | [+0.0485, +0.4378] |
| Graph - SLERP, all | -0.0410 | [-0.1080, -0.0035] |
| SLERP - unique real, all | -0.0800 | [-0.1454, -0.0302] |

The sparse-interior exact-CLS advantage is positive in three of four regions,
but its interval still touches zero. It is a follow-up signal, not a confirmed
headline result.

## Validity, Memorization, And Diversity

Independent ResNet-50 top-1 validity:

| Bank | Accuracy |
|---|---:|
| CA8 exact CLS | 0.6969 |
| CA8 direct SLERP | 0.7836 |
| CA8 graph path | 0.8343 |
| Random same-class CLS | 0.8713 |
| Class-only | 0.8207 |
| Real ImageNet validation | 0.9347 |

Exact-source retrieval at rank one, reported as DINOv3/SigLIP2:

| Bank | Source@1 |
|---|---:|
| Exact CLS | 0.8285 / 0.5468 |
| Direct SLERP | 0.5838 / 0.3460 |
| Graph path | 0.3138 / 0.1842 |
| Random same-class CLS | 0.8168 / 0.4474 |
| Class-only | 0.0000 / 0.0000 |

Mean LPIPS distance to the conditioning source is `0.5959` for exact CLS,
`0.6161` for SLERP, `0.6358` for graph paths, `0.5659` for random same-class
CLS, and `0.7329` for class-only generation. This is source-centered semantic
generation rather than pixel-identical reconstruction.

Fixed-condition diversity, reported as DINOv3/SigLIP2/LPIPS pair distance:

| Bank | Diversity |
|---|---:|
| Class-only | 0.4828 / 0.2579 / 0.7007 |
| Exact CLS | 0.2017 / 0.1062 / 0.5718 |
| Direct SLERP | 0.1969 / 0.1209 / 0.5738 |
| Graph path | 0.1896 / 0.1167 / 0.5661 |

Graph paths therefore improve endpoint novelty and independent class validity,
but they do not create broad within-condition diversity and they reduce repair
relative to direct SLERP. Generic visual validity is not a sufficient proxy for
region-specific downstream value.

## Reproducibility Caveat And Resolution

The historical additive-AdaLN and CA8 sampling launches produced different
class-only banks despite nominally identical checkpoints, seeds, and
conditions. Those old outputs were generated on A100 80 GB and 40 GB nodes
without the new deterministic sampler mode, so small cross-architecture
differences should not be interpreted as architecture effects.

Jobs `4800561`-`4800564` reran a 128-image class-only canary twice on A100 40
GB and once on A100 80 GB with deterministic algorithms, TF32 disabled, and
per-output initial-state hashes. All three runs had identical initial states
and byte-identical uint8 images (`128/128`, maximum absolute difference `0`).
Hardware itself is therefore not an unavoidable confound; future paired
architecture comparisons must use this deterministic protocol.

Artifact:

`artifacts/cls_conditioning/sampling_repro_canary_20260830/`

## Follow-Up Gates

The true stochastic image-augmentation baseline completed in jobs
`4800556`-`4800558`. Standard ImageNet augmentation repairs `0.2312` target CE
overall and conservative augmentation repairs `0.2544`, compared with `0.2570`
for cached retained-anchor repetition. Relative to standard augmentation:

| Targeted arm | Mean CE-repair difference | Region-bootstrap interval |
|---|---:|---:|
| CA8 exact CLS | +0.0326 | [-0.0074, +0.0779] |
| CA8 direct SLERP | +0.0089 | [-0.0119, +0.0313] |
| CA8 graph path | -0.0322 | [-0.0875, +0.0090] |
| Additive-AdaLN exact CLS | +0.0479 | [-0.0003, +0.1185] |
| Additive-AdaLN direct SLERP | +0.0584 | [-0.0008, +0.1182] |
| Unique target real | +0.0889 | [+0.0242, +0.1713] |

In the original non-deterministic comparison, no synthetic arm established a
reliable advantage over stochastic augmentation. Unique real data did. The
matched deterministic replicate below slightly revises the additive-AdaLN
point estimate while leaving the broader six-region conclusion provisional.

The deterministic architecture replicate `4800758`-`4800764` is complete.
It confirms that exact-CLS CA8 and additive AdaLN are effectively tied on CE
repair: CA8 minus AdaLN is `+0.0032 [-0.0147,+0.0213]`. Relative to the same
standard stochastic-augmentation bank, additive AdaLN exact CLS is
`+0.0293 [+0.0026,+0.0612]` and additive SLERP is
`+0.0387 [+0.0019,+0.0796]`; CA8 exact remains uncertain at
`+0.0325 [-0.0073,+0.0774]`. These are six-region capability estimates, not a
broad augmentation claim. Subsequent gates are:

1. Replicate only a prespecified sparse-interior arm with deterministic matched
   sampling and more region/class units if synthesis beats the augmented-anchor
   baseline in a larger frozen screen.
2. Train one region-excluded or cross-fitted generator before making a causal
   synthetic-repair claim; the current generator saw the full training set.
3. Do not add another broad conditioner or geometry branch until these gates
   resolve. CA8 has solved late-training stability, not source locking or the
   utility comparison.

## Bottom Line

CA8 is a credible and stable targeted generator. It clearly beats generic
synthetic controls, but it does not reliably beat stochastic augmentation or
local repetition, remains source-centered, and trails unique real regional
data. Sparse interiors remain the only plausible stratum for a tightly scoped
confirmatory synthesis follow-up.

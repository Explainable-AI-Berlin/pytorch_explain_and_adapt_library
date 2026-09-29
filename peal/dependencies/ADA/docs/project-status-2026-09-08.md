# ADA Project Status

**Canonical snapshot:** 2026-09-08
**Decision ledger:** [experiment-evidence-ledger-2026-09-08.md](experiment-evidence-ledger-2026-09-08.md)
**Independent review:** [independent-review-2026-09-07.md](independent-review-2026-09-07.md)

## Strongest Defensible Claim

Local within-class support in a semantic representation can identify regions
whose training examples have location-specific causal value. Removing those
examples damages target-region competence, and restoring unique examples from
the same region repairs the damage better than equal-count data elsewhere in
the class.

This result now holds in:

- an eight-region ImageNet-100 development pilot;
- a clean 30-class ImageNet-1K confirmation excluding all ImageNet-100
  development classes;
- DINOv2, DINOv3, and SigLIP2 frozen feature heads;
- a six-region, 30,000-step scratch ResNet18 screen.

The evidence does **not** yet establish prospective discovery from an incomplete
dataset or clean, leakage-resistant synthetic repair beyond retained-data
augmentation.

## Representation Result

| Support atlas | DINO linear error AUROC | ResNet18 error AUROC |
| --- | ---: | ---: |
| DINOv2 CLS, same-class k=50 | 0.8971 | 0.8304 |
| SD-VAE variants | approximately 0.54-0.59 | approximately 0.56-0.58 |

DINO and VAE sparse top-decile sets overlap at Jaccard `0.0549`,
near the independent-ranking expectation `0.0526`. VAE support can
diagnose weak VAE-native probes, but it does not transfer well to semantic
classifiers. This is a completed negative representation control.

## Clean IN1K Causal Confirmation

Thirty selected regions use thirty classes absent from ImageNet-100. At q=.25:

| Stratum | Target error change | Target CE change |
| --- | ---: | ---: |
| Sparse interior | +0.0677 | +0.4016 |
| Sparse boundary | +0.1915 | +1.1417 |

Same-class random deletion is approximately neutral. Across all 22 sparse
regions, unique target minus same-class non-target CE repair is `+0.4620`,
CI `[0.3190, 0.6110]`, with the expected sign in `22/22`
regions.

Unique local data retain a modest CE advantage over retained-anchor repetition
(`+0.0412`, CI `[0.0020, 0.0886]`) and the registered
cached-feature repeat augmentation arm (`+0.0342`,
CI `[0.0139, 0.0561]`). This registered arm is not fresh
image-space stochastic augmentation.

## Scratch-Learner Correction

The scientific scratch ResNet18 screen did **not** use the old 1,000-step
canary. It used the newly committed 30,000-step configuration, batch size 128,
standard image augmentation, and random initialization. The full q=1 model
reached validation accuracy `0.748` and NLL `0.9579`.

Across four sparse interiors and two sparse boundaries:

| Intervention | Mean target CE change |
| --- | ---: |
| Regional deletion | +0.5845 |
| Same-class random deletion | +0.0758 |
| Count-preserving regional removal | +0.9385 |

Target error rises `+0.1187` and margin falls `-1.6664`;
`5/6` regions have positive target-CE damage. This resolves the
reviewer's one-pass concern. The remaining limitations are one learner seed,
six regions, and no prespecified convergence curve.

## Generator Evidence

All generator results use the native RAEv2 pipeline with the official DINOv2-L
K=1 decoder and statistics.

### Matched FID50k

| Conditioning and checkpoint | FID50k |
| --- | ---: |
| Class only, additive epoch 31 | 4.3116 |
| CLS only, additive epoch 31 | 3.3163 |
| Class + CLS, additive epoch 31 | 3.3485 |
| CLS only, validation CLS oracle, additive epoch 31 | 2.3243 |
| CLS only, CA8 epoch 31 | 3.3290 |
| CLS only, CA8 epoch 58 | 3.1485 |

The validation-CLS row is an oracle reconstruction-style condition, not a
deployable unconditional sampler. FID is distribution fit, not proof of
generalization, diversity, or utility.

### Six-Region Synthetic Capability

At the same q=.25 deleted state and 50%-of-missing exposure budget:

| Added data | Target CE repair |
| --- | ---: |
| Class-only synthetic | 0.0572 |
| Retained-anchor image augmentation | 0.2312 |
| Direct SLERP synthetic | 0.2401 |
| Retained-anchor repetition | 0.2570 |
| Exact retained CLS synthetic | 0.2638 |
| Unique target real reference | 0.3201 |

Targeted CLS synthesis clearly beats class-only synthesis. Exact CLS does not
reliably beat repetition in the canonical CA8 screen. A deterministic additive
rerun found small positive advantages over standard augmentation, but the
generator had seen full IN1K and the experiment covers six selected regions.
It is therefore a capability result, not the headline ADA claim.

CA8 epoch 58 improves global FID, but it does not materially improve repair:
exact-CLS CE repair changes from `0.2637` at epoch 31 to
`0.2783` at epoch 58, direct SLERP is essentially unchanged, and the
graph arm remains weaker.

## Graph-Geodesic Outcome

The graph branch is complete. Graph paths improve condition support and
adherence and sharply reduce endpoint locking relative to direct SLERP.
Nevertheless, graph minus direct target-CE repair is `-0.0745`,
CI `[-0.1522, -0.0168]`, in the original additive-AdaLN bank and
`-0.0410`, CI `[-0.1080, -0.0035]`, in the separately
sampled CA8 bank. Graph regional-distribution metrics are also worse. Graph
geodesics remain a useful manifold and replay diagnostic; direct SLERP remains
the current downstream-repair proposal.

## Conditioner Frontier

- **Additive AdaLN:** strongest fine-grained edit evidence, but two exact-CLS
  runs developed late instability.
- **CA8:** stable through epoch 58 and best completed FID50k; current production
  route.
- **CA1:** no robust matched advantage over CA8 and later unstable; closed.
- **SA1:** job `4788862`, epoch 36 complete and epoch 37 in progress
  at the 2026-09-08 09:00 CEST refresh; finite; scheduled end
  2026-09-11 07:56 CEST.
- **Bounded geometry AdaLN:** job `4789176`, epoch 25 complete and
  epoch 26 in progress at the same refresh; projected geometry stable;
  scheduled end 2026-09-12 05:12 CEST.

No one-trajectory result proves that CA1 or AdaLN is intrinsically unstable.
Likewise, no matched audit proves CA1 and CA8 equivalent. The next distinct
mechanism question is whether changing exact one-to-one condition-target
supervision improves diversity without destroying localization. A full region
token plus exact residual can remain information-preserving; simply bounding
the residual amplitude is not sufficient to call the condition coarser.

## BREEDS Status

A 30,000-step scratch ResNet18 Living17 pilot is active over six known
superclasses. Artifact build job `4847544` completed. At the
2026-09-08 09:00 CEST refresh, five of 19 probe tasks had completed, four were
running, the remaining tasks were throttled behind the four-task array limit,
and evaluator `4847546` was correctly waiting on the full array.

It compares target-subclass deletion, same-superclass non-target deletion, and
count-preserving target reallocation. It is a causal known-subpopulation
deletion screen, not yet restoration, synthesis, or a prospective retained-only
selector experiment.

BREEDS remains ImageNet-derived. It tests semantically named regions but is not
a pretraining-disjoint domain.

## Pretraining Boundary

DINOv2 brings a strong ImageNet-shaped semantic prior. Public web-source
deduplication should not be described as a certificate of zero exact exposure
across the entire pretraining mixture, which includes ImageNet-22K. A scratch
learner removes learner pretraining, not atlas prior. A future Stage-2-excluded
generator must separately document exposure through its encoder, decoder,
normalization statistics, initialization, and exact Stage-2 training pairs.

## Three Highest-Value Missing Experiments

1. **Adequate-training distinct-real headroom:** replicate unique real versus
   repetition and true augmentation over 20-30 new regions and three paired
   scratch-learner seeds after freezing a convergence budget.
2. **Prospective retained-only allocation:** compare ADA with random,
   frequency, density, and learning-dynamics/TADA-style selectors while a
   hidden reserve remains completely inaccessible to selection.
3. **Exclusion-audited synthetic confirmation:** train a stable CA8 Stage-2
   model without exact target identities or their cached views, then test
   retained-only exact CLS and SLERP against repetition, true augmentation,
   class-only plus matched rejection, and unique real reference.

The exact thresholds, controls, and stop criteria are frozen in the
[experiment ledger](experiment-evidence-ledger-2026-09-08.md).

## Project Decision

The actionability paper is viable without a positive generator result if the
scratch and prospective real-data results replicate. The generative headline
remains conditional on leakage-resistant targeted synthesis beating the
strongest retained-data baseline. Do not spend the next phase on broad
condition-injection, graph, VAE, or uncertainty sweeps.

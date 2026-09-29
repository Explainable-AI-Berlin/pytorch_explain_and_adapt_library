# CLS Conditioner Training Frontier

**Snapshot:** 2026-09-08
**Status:** architecture audit reconciled with independent review

## Executive Verdict

All current ImageNet generator results use the native RAEv2 pipeline with the
official DINOv2-L K=1 decoder and normalization statistics. The configuration
name `stage1.RAE` is a Python class inside RAEv2, not evidence that a
result used the earlier RAE repository.

The completed routes show that exact image-level CLS conditioning is feasible
and can improve distributional fidelity, but it is highly source-centered.
They do not prove an intrinsic ranking among conditioning mechanisms: each long
route is one optimization trajectory, and the checkpoints are not all matched
by compute.

The next distinct question is not a ninth token-injection variant. It is whether
changing the exact one-condition/one-target supervision relation can reduce
source locking while preserving localization.

## Conditioning Routes

| Route | Purpose | Current reading |
| --- | --- | --- |
| Additive AdaLN, CLS only | Strong global semantic constraint | Best-developed edits; original long run later unstable |
| Additive AdaLN, class only | Broad class-distribution baseline | Stable baseline |
| Additive AdaLN, class + CLS | Broad class prior plus local residual | Strong matched ablation; original long run later unstable |
| Bounded geometry AdaLN | Preserve projected geometry and cap modulation | Running; epoch 25 finite at last refresh |
| CA1 every fourth block | One CLS-derived cross-attention token | Closed after late instability and no matched advantage |
| CA8 every fourth block | Eight learned CLS-derived context tokens | Stable through epoch 58; FID50k 3.1485 |
| SA1 prefix | One CLS-derived self-attention prefix | Running; epoch 36 finite at last refresh |

With one cross-attention key, softmax attention weight is one for every patch
query. CA1 can still apply a learned condition-dependent broadcast residual,
but it cannot choose among multiple context values. CA8 maps one CLS vector to
eight distinct learned key/value tokens, allowing patch queries to select among
different transformed components. This mathematical distinction motivates the
ablation; it does not prove why CA1 became unstable.

## CA1 Closure

CA1 job `4736515` was cancelled after a matched epoch-39 audit over
24 exact conditions, two diffusion seeds, EMA weights, CFG 1, and the same
50-step sampler.

| Metric | CA1 epoch 39 | CA8 epoch 39 |
| --- | ---: | ---: |
| Epoch loss | 0.8955 | 0.8860 |
| Mean re-encoded condition cosine | 0.8572 | 0.8553 |
| Source retrieval at rank 1 | 85.42% | 87.50% |
| Source retrieval at rank 5 | 100.00% | 100.00% |
| Fixed-condition seed diversity | 0.1454 | 0.1559 |
| Distinct nearest sources | 1.1667 | 1.0833 |

CA1's mean-cosine difference is `+0.0019`, but its median paired
difference is `-0.0082` and it wins only 18 of 48 image comparisons.
It is more diverse for only 7 of 24 conditions.

This supports only the narrow decision that CA1 offered no demonstrated
advantage worth another long run. It does **not** establish that CA1 and CA8 are
equivalent or that one-token cross-attention is intrinsically unstable.

Global gradient clipping was enabled, but it does not bound the forward
cross-attention residual. Muon-style matrix updates also normalize matrix
gradients internally, so global rescaling is not a complete forward-stability
guarantee. LayerScale or a bounded residual plus a lower-rate AdamW group would
be a minimal rescue, but its current information value is low.

## Current Long Diagnostics

At the 2026-09-08 09:00 CEST scheduler refresh:

- `4788862`, SA1: epoch 36 complete, epoch 37 in progress, loss
  `0.8880`, finite; scheduled end 2026-09-11 07:56 CEST;
- `4789176`, bounded geometry AdaLN: epoch 25 complete, epoch 26 in
  progress, loss `0.9209`, finite with stable projected geometry;
  scheduled end 2026-09-12 05:12 CEST;
- `4736515`, CA1: intentionally cancelled.

Training loss and finite telemetry are health checks, not evidence of condition
use, image quality, or ADA utility. Each completed model must receive the same
matched FID, source-retrieval, adherence, fixed-condition diversity, and repair
evaluation before comparison.

## The Distinct Supervision Experiment

Exact training optimizes one observed target per image condition:

\[
p_\theta(z_i\mid c_i,y_i).
\]

This makes input-output fidelity and source identity an easy solution. A local
pairing intervention instead trains:

\[
p_\theta(z_i\mid \tilde c_i,y_i),\qquad
\tilde c_i\in\mathcal N_y(c_i),
\]

where the neighborhood is high-purity, same-class, local, and frozen using
training data only.

This should be described as changing **condition-target supervision**, not
automatically as coarsening the condition: the full observed CLS vector remains
available. Related-image conditioning also has precedent, including Semantica,
so pairing itself is not the paper novelty. The ADA contribution would be
linking such conditions to causally validated repair under leakage-resistant
controls.

A region token plus continuous residual is not necessarily coarser either. If
the region identity and exact residual are invertible back to the source CLS,
the same information remains available. Bounding residual amplitude controls
scale, not mutual information. Genuine coarsening requires quantization,
residual dropout/noise, a bottleneck, or neighborhood-shared supervision.

## Bounded Pilot

Do not begin with six full models. Continue stable CA8 from one matched
checkpoint for 10,000 steps under three arms:

1. exact condition-target pairs;
2. 50% exact plus 50% high-purity local-kNN pairing;
3. 50% exact plus 50% random same-class pairing.

Exclude self matches and the same source identity across all cached views. Keep
optimizer, data order, exposure, and compute matched.

Evaluate:

- independent class validity;
- target-region hit rate;
- source-at-1 and source-at-5;
- fixed-condition diversity;
- condition adherence;
- FID as a secondary global metric;
- downstream repair against true retained-anchor augmentation.

Prespecified go criterion:

- source-at-1 decreases by at least 10 percentage points;
- fixed-condition diversity improves by at least 10%;
- class validity falls no more than two points;
- region hit falls no more than five points;
- local pairing exceeds random same-class pairing by at least five region-hit
  points.

Stop if local pairing leaves the region, simply retrieves a finite neighbor
bank, or performs like random same-class pairing.

## Higher-Priority Scientific Axes

1. **Adequate-training real-data headroom.** Establish that distinct local real
   examples still beat repetition and true augmentation after learner
   convergence.
2. **Prospective retained-only allocation.** Select valuable regions without
   reserve identities or validation outcomes.
3. **Stage-2 exclusion-audited generator.** Exclude exact target identities and
   all cached views from conditioning-target training pairs and disclose every
   inherited pretraining source.
4. **Learned class-conditional CLS prior.** Needed for deployable CLS sampling,
   but the compact canary had class-centroid accuracy `0.594` and
   support `0.8447`, so it is not first.
5. **Forward Explorative Modeling.** The canary gave selection gain
   `0.0030` at approximately 40.7% runtime overhead. It needs image
   evidence before a long run.
6. **Text pathway.** Useful for editing and interpretation, but not a solution
   to causal acquisition validity.

## Decision

Finish SA1 and bounded AdaLN and evaluate them fairly. Do not enumerate more
minor exact-CLS injection routes. Run the three-arm neighborhood pilot only if
a bounded mechanism result is useful while the real-data experiments proceed.

For ADA, the bottleneck is leakage-resistant prospective repair, not the
existence of another trainable conditioner.

# Pretrained Atlas Priors And BREEDS

**Resolved:** 2026-09-08

## Three Questions That Must Stay Separate

1. **Frozen-head data value:** which examples are needed to fit a head in an
   already learned representation?
2. **End-to-end data value:** which examples matter when a visual learner is
   trained from pixels on each intervention dataset?
3. **Prospective acquisition:** can an incomplete dataset identify what it
   lacks without consulting held-out candidates or their outcomes?

The DINOv2/DINOv3/SigLIP2 probes answer the first question. The 30,000-step
scratch ResNet18 screen begins to answer the second. The third is still open.

## What The DINOv2 Prior Does And Does Not Establish

DINOv2 was trained on LVD-142M and used ImageNet-related query sets when
constructing its curated web data. The public description includes source-level
deduplication against benchmark splits, but this should not be upgraded into a
certificate that no exact image was ever present anywhere in the complete
training mixture. The mixture also includes ImageNet-22K as an existing curated
source.

The defensible statement is:

> DINOv2 supplies a strong ImageNet-shaped semantic prior; exact exposure for
> every ADA image is not established either way by the present artifacts.

This prior is not automatically a fatal confound. ADA asks whether geometry
from a pretrained atlas predicts causal data value for a new downstream
learner. A scratch learner can test that transfer. It does not make the atlas
pretraining-disjoint.

The current RAEv2 generator has a more direct exposure issue: its Stage-2 model
was trained on exact ImageNet-1K CLS and patch-latent pairs. Synthetic
restoration using that model is a capability and replay diagnostic, not a clean
acquisition result.

Primary source: [DINOv2](https://arxiv.org/abs/2304.07193).

## What The Scratch Screen Resolves

The scientific IN100 scratch screen used a randomly initialized ResNet18 for
30,000 steps, not the old 1,000-step plumbing canary. Regional deletion caused
substantially greater target damage than same-class random deletion, and
count-preserving target removal remained damaging.

This shows that DINO-defined regions can have causal value for a learner that
does not consume a frozen DINO representation. It does not establish:

- replication across learner seeds and architectures;
- convergence under a frozen learning-curve criterion;
- prospective region discovery from retained data only;
- atlas independence from foundation-model priors.

## What The Current BREEDS Run Is

BREEDS groups ImageNet fine classes into visually coherent superclasses. The
active Living17 pilot uses six superclasses and fixes one target fine class per
superclass. A scratch ResNet18 is trained for 30,000 steps under 19 manifests:

- one shared full-data baseline;
- target-subclass deletion at q=.25;
- equal-count deletion from another subclass in the same superclass;
- target-subclass deletion with count-preserving reallocation.

The target choices are frozen from hierarchy and seed information rather than
DINO geometry or observed intervention outcomes.

This is a **known-subpopulation causal deletion pilot**. It asks whether named
subpopulation coverage matters beyond superclass count. It is not yet:

- a restoration experiment;
- a synthetic-repair experiment;
- an ADA-versus-baselines selector comparison;
- prospective acquisition from an inaccessible reserve.

BREEDS is itself ImageNet-derived. It is useful for semantic ground truth, but
it is not a pretraining-disjoint benchmark.

Primary source: [BREEDS](https://arxiv.org/abs/2008.04859).

## Required Prospective Extension

The paper-critical follow-up should separate candidate discovery from
candidate evaluation.

1. Build a retained training set and a hidden reserve before fitting any
   selector.
2. Permit selectors to use only retained examples, labels, and out-of-fold
   learner signals.
3. Compare random, class frequency, representation density/support,
   learning-dynamics or TADA-style selection, and ADA under one joint budget.
4. Reveal reserve identities only when constructing the acquired training set
   and evaluating outcomes.
5. Include an oracle reserve-aware selector only as a labelled ceiling.
6. Use target-subpopulation CE and global balanced accuracy as coprimary
   outcomes.
7. Treat region/class, not images or training seeds, as the scientific unit.

A useful success criterion is that ADA exceeds the strongest retained-only
baseline with a positive paired region-bootstrap interval and recovers at least
10% of the oracle-real repair gain.

## Generator Exclusion Language

For a future generator, report exposure component by component:

| Component | Required disclosure |
| --- | --- |
| Atlas encoder | pretraining data and known benchmark overlap |
| Stage-1 decoder | training source and frozen revision |
| Normalization statistics | dataset and identities used |
| Stage-2 initialization | inherited checkpoint and source |
| Stage-2 pairs | exact image identities and cached views |
| Generator conditions | retained-only identity manifest |
| Evaluation reserve | proof it was inaccessible during selection/training |

An `exact-pair-excluded` Stage-2 result means exact target identities
and all cached views were absent from Stage-2 conditioning-target pairs. It
does not mean the entire stack has never learned from ImageNet-like data.

## Literature Bridge

[Datamodels](https://arxiv.org/abs/2202.00622) studies training-data
counterfactuals directly. [RHO-LOSS](https://arxiv.org/abs/2206.07137)
distinguishes difficult examples from examples expected to reduce holdout loss.
Learning-dynamics and targeted data-acquisition methods are therefore mandatory
baselines for a prospective ADA claim, not optional uncertainty comparisons.

A later region-level datamodel could sample inclusion masks over frozen
regions, train scratch learners, and predict regional margins from region
membership. That is mechanistically attractive, but it should follow the
simpler retained-only benchmark rather than delay it.

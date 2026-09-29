# Count-Matched Class Deficit And Regional-Coverage Protocol

**Resolved status:** secondary proposal; not launched
**Purpose:** compare diffuse and coverage-concentrated deletion at identical
class counts

## Correct Scope

The earlier whole-dataset q=.75 experiment retained 75% independently inside
every class. It was a redundancy stress test: class balance was preserved and
most local modes remained represented.

The proposed experiment below should **not** be described as estimating the
main effect of class imbalance. Every selected class is equally under-counted
in both primary arms. The identifiable contrast is:

> Under a fixed class-count deficit, does concentrating deletion in weakly
> supported regions cause more damage than diffuse random deletion?

## Frozen Dataset Construction

Select 50 of the 100 ImageNet-100 classes using a frozen seed. Retain 50% of
training images in those classes and 100% in the remaining classes. With equal
class sizes this removes 25% of the full dataset. Deterministic rounding must
preserve exact matched counts in every paired arm.

Use three frozen class-selection seeds as dataset-level replicates. Learner
seeds are repeated paired measurements, not additional dataset units.

## Primary Count-Matched Arms

1. **Diffuse deletion:** remove the matched count uniformly within each
   selected class.
2. **Coverage-concentrated deletion:** remove the same count beginning in
   training-only weak-support regions under the frozen q=1 atlas.

The difference isolates deletion location under the same class counts. Region
rankings, ties, and fallback rules must be frozen before intervention training.

## Matched Additions

At a prespecified partial budget, preferably 25% of the deleted count, compare:

- no additions;
- retained real repetition;
- true stochastic image-space augmentation of retained anchors;
- class-only synthetic images;
- random retained same-class CLS synthesis;
- supported target-region retained CLS synthesis;
- direct same-region SLERP synthesis;
- unique deleted real examples as an empirical reference.

Unique real is not a guaranteed upper bound: optimization noise or regularizing
synthetic samples can occasionally exceed the original-data result. Label it
as the empirical restoration reference.

## Learners And Outcomes

Primary learner: randomly initialized ResNet18 trained to a budget frozen from
a separate development learning curve.

Primary outcomes:

- balanced accuracy over selected under-counted classes;
- mean target-class CE;
- balanced accuracy across frozen within-class regions;
- weakest-support-region CE and margin.

Global accuracy is secondary because half the classes are unaffected.

## Primary Contrasts

1. Coverage-concentrated versus diffuse deletion under identical counts.
2. Targeted CLS synthesis versus true retained-anchor augmentation.
3. Unique real restoration versus retained repetition.
4. Interaction between deletion geometry and restoration method, only if
   prespecified.

Do not infer a generic `class imbalance causes X` result from these
arms. A genuine 2x2 class-count-by-coverage design would be required for that
main effect and should be added only if the interaction becomes central.

## Interpretation

| Outcome | Defensible interpretation |
| --- | --- |
| Concentrated deletion is worse than diffuse deletion | Within-class location matters at fixed class count |
| Targeted synthesis helps only concentrated deletion | Generation may address geometric coverage rather than count alone |
| Repetition matches targeted synthesis | Extra exposure, not generated novelty, explains recovery |
| Unique real beats exposure baselines | Distinct examples carry local information absent from retained anchors |
| Scratch and frozen learners disagree | Data value depends on representation learning or optimization |

## Leakage Rule

The artificial causal benchmark may use the full q=1 atlas to define deletion.
Any prospective selector must reconstruct scores from retained data only.
Deleted images may serve as the real-restoration reference and evaluation
target, never as synthetic conditions or selector inputs.

The current full-IN1K generator has seen later-deleted examples during Stage-2
training. Results from it remain capability/replay diagnostics. A paper-grade
synthetic claim requires exact target identities and all extracted views to be
excluded from Stage-2 training, with inherited pretraining exposure disclosed
separately.

## Priority

This experiment is below adequate-training real-restoration replication,
retained-only prospective acquisition, and exclusion-audited synthetic repair.
Do not launch it merely to create a larger grid.

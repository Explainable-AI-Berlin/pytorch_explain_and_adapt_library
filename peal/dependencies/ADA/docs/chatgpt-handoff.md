# ChatGPT Research Handoff

**Repository:** <https://github.com/David-Drexlin/ADA>
**Canonical snapshot:** 2026-09-08

## Read In This Order

1. [Current project status](project-status-2026-09-08.md)
2. [Independent scientific review](independent-review-2026-09-07.md)
3. [Resolved experiment and evidence ledger](experiment-evidence-ledger-2026-09-08.md)
4. [Pretraining prior and BREEDS](pretraining-prior-and-breeds.md)
5. [Count-matched class-deficit protocol](class-imbalance-25pct-protocol.md)
6. [CA8 downstream result](ca8-downstream-result.md)
7. [Graph-geodesic conditioning](graph-geodesic-conditioning.md)
8. [CLS conditioner frontier](model-training-frontier-2026-09-07.md)
9. [Generator models and checkpoint provenance](generator-models.md)

The first three files are sufficient for the current scientific decision. The
remaining files contain protocol and implementation detail.

## Central Claim

ADA asks where another distinct unit of training data has causal value. The
strongest current result is that deleting local within-class support in a
DINOv2 atlas damages regional competence and unique local restoration repairs
it better than equal-count same-class non-target data. This replicates on clean
IN1K classes, transfers to DINOv3 and SigLIP2 heads, and appears in a
30,000-step scratch ResNet18 screen.

The generative claim remains open. Targeted CLS synthesis beats class-only
synthesis but has not broadly and leakage-resistently beaten the strongest
retained-data augmentation/repetition baseline.

## Review Resolutions

- The scientific scratch ResNet18 used 30,000 steps; the committed 1,000-step
  file was only a plumbing canary. The exact scientific config is now
  committed.
- All graph-geodesic sidecars completed. Graph paths improve support/adherence
  and reduce endpoint locking, but perform worse than direct SLERP for repair.
- FID rows are now tied to explicit checkpoints. Validation-image CLS FID is an
  oracle reconstruction-style diagnostic.
- BREEDS is currently a known-subpopulation deletion screen, not a prospective
  selector or restoration result.
- Neighborhood pairing changes condition-target supervision; it is not
  automatically a lower-information condition.
- A region token plus exact residual may be invertible, and a norm cap alone
  does not coarsen it.
- DINOv2 web deduplication is not documented here as proof of zero exact
  exposure across the full pretraining mixture.
- The proposed class-deficit experiment identifies deletion-location effects
  at fixed class counts, not a generic class-imbalance main effect.

## Three Highest-Value Experiments

1. Replicate distinct-real versus repetition and true augmentation after
   freezing an adequate scratch-learner convergence budget.
2. Run a prospective retained-only allocation benchmark against random,
   frequency, density, and learning-dynamics/TADA-style selectors.
3. Train one CA8 Stage-2 generator excluding exact target identities and every
   cached view, then compare retained-only targeted synthesis against the
   strongest exposure baseline.

Exact controls, thresholds, and stop rules are in the
[evidence ledger](experiment-evidence-ledger-2026-09-08.md).

## Paste-Ready Prompt

```text
Please review the private repository:
https://github.com/David-Drexlin/ADA

Read:
1. docs/project-status-2026-09-08.md
2. docs/independent-review-2026-09-07.md
3. docs/experiment-evidence-ledger-2026-09-08.md
4. docs/pretraining-prior-and-breeds.md
5. docs/model-training-frontier-2026-09-07.md
6. docs/generator-models.md

The independent review has already been reconciled once. Audit the resolved
ledger rather than restarting broad ideation.

Please determine:
- whether each completed result supports exactly the stated claim;
- whether the frozen go/stop thresholds are fair and falsifiable;
- whether adequate-training real-data headroom, prospective retained-only
  selection, and Stage-2 exclusion are the correct order;
- which single result would most change the CVPR paper decision;
- whether any remaining comparison leaks withheld identities, validation
  outcomes, or generator exposure.

Do not recommend broad architecture, graph, VAE, or uncertainty sweeps unless
you identify a specific unresolved confound that the frozen experiments cannot
answer.
```

## Environment Note

Large datasets, latent caches, checkpoints, scheduler logs, and generated
images remain on Hydra under collaborator-controlled storage and are not in
Git. Scheduler state is time-sensitive and must be refreshed independently.

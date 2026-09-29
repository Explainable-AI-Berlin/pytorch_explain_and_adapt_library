# ADA Experiment And Evidence Ledger

**Resolved:** 2026-09-08
**Purpose:** freeze what each result establishes, resolve the independent review,
and define the smallest decisive experiment sequence.

This ledger supersedes informal job lists as the scientific decision record.
Scheduler identifiers establish artifact provenance; they are never treated as
independent replicates.

## Evidence Rules

1. The independent unit for regional claims is the region or class. Repeated
   learner seeds are paired measurements within that unit.
2. Continuous target cross-entropy and true-class margin are primary sensitive
   outcomes. Target error is the interpretable secondary outcome.
3. A deletion effect is local only when target damage exceeds same-class
   off-target damage and matched random deletion.
4. A restoration effect is location-specific only when unique target-region
   data outperform equal-count same-class non-target data.
5. Unique real data are an empirical reference, not a theoretical upper bound.
6. Synthetic data are useful only when they outperform matched retained-data
   repetition or stochastic augmentation, not merely class-only synthesis.
7. Any generator trained on examples later called deleted is a capability and
   replay diagnostic, not a clean acquisition result.
8. Oracle conditions, including validation-image CLS vectors, must be labelled
   explicitly and never compared as deployable unconditional samplers.

## Completed Observational Evidence

| ID | Experiment | Evidence | Conclusion |
| --- | --- | --- | --- |
| O1 | IN100 DINOv2 same-class support | Error AUROC `0.8971` for a DINO linear probe and `0.8304` for ResNet18 | DINO support transfers across classifiers |
| O2 | SD-VAE support controls | Semantic-classifier error AUROC approximately `0.54-0.59`; DINO/VAE sparse top-decile Jaccard `0.0549` versus independent expectation `0.0526` | Arbitrary latent sparsity is insufficient |
| O3 | VAE-native probes | VAE support predicts failures of weak VAE-space probes, but transfers poorly to strong semantic classifiers | Self-space support is not transferable support |
| O4 | SigLIP2 language sidecar | Sparse-boundary mean text margin `-0.0071` versus dense-interior `0.0875` | Sparse regions differ in textual ambiguity; exploratory moderator only |

## Completed Causal Evidence

### C1: IN100 Eight-Region Development Pilot

Regional deletion, count-preserving regional removal, same-class random
deletion, unique target restoration, and same-class non-target restoration were
run on eight frozen DINOv2 regions. Deletion damages target competence and
unique local data repair it; equal-count data elsewhere in the class do not
provide the same recovery.

This is a positive development result, not the clean confirmatory study.

### C2: Cross-Representation Transfer

The immutable IN100 interventions were reused with DINOv3 and SigLIP2 feature
heads. For sparse interiors, deletion increased target CE by `0.2207` in
DINOv3 and `0.1287` in SigLIP2. Unique target restoration recovered
approximately `0.2211` and `0.1289`, while same-class non-target
restoration was near zero.

This establishes transfer across frozen semantic encoders and linear heads. It
does not by itself establish architecture-independent end-to-end data value.

### C3: Clean IN1K Confirmation

The clean 30-region selection excludes all ImageNet-100 development classes.
The artifact chain used jobs `4688448`, `4688460`, and `4697966`;
probe and evaluation jobs were `4737415-4737421`; paired analyses were
`4749150`, `4749154`, `4749209`, and canonical
`4749211`.

Primary q=.25 deletion effects:

| Stratum | Target error change | Target CE change |
| --- | ---: | ---: |
| Sparse interior | +0.0677 | +0.4016 |
| Sparse boundary | +0.1915 | +1.1417 |

Same-class random deletion was approximately neutral. Across all 22 sparse
regions, unique target minus same-class non-target CE repair was `+0.4620`
with region-bootstrap CI `[0.3190, 0.6110]` and expected sign in
`22/22` regions.

Unique target data had a smaller advantage over exposure baselines:

- versus retained-anchor repetition: `+0.0412` CE repair,
  CI `[0.0020, 0.0886]`;
- versus the registered feature-repeat augmented-anchor arm:
  `+0.0342`, CI `[0.0139, 0.0561]`.

The registered augmentation repeats cached features; it is not fresh
image-space stochastic augmentation. The evidence confirms locality and a
modest distinct-data advantage, but not yet a broad diversity claim.

### C4: Scratch ResNet18 Architecture Screen

The old committed `fixed_steps: 1000` configuration was a plumbing
canary. The scientific lean-six screen used
`in100_imageclf_resnet18_scratch_convergence30k_noshm_deletion_probe.yaml`:
30,000 optimizer steps, batch size 128, standard image augmentation, no
pretrained learner, and approximately 3.84 million sample presentations
(about 30.3 IN100 dataset-equivalent passes).

The q=1 model reached validation accuracy `0.748` and NLL
`0.9579`. Across four sparse interiors and two sparse boundaries:

| Intervention | Mean target CE change |
| --- | ---: |
| Regional deletion | +0.5845 |
| Same-class random deletion | +0.0758 |
| Count-preserving regional removal | +0.9385 |

Target error changed by `+0.1187`, margin by `-1.6664`, and
`5/6` regions had positive target-CE damage. Array job
`4762564` and canonical evaluator `4798609` produced the
artifacts.

**Resolved review concern:** this was not a one-pass or 1,000-step learner.
Remaining limitations are one model seed, six selected regions, and no frozen
learning-curve convergence criterion.

## Completed Generator Evidence

### G1: Matched FID50k Checkpoints

| Model and checkpoint | Condition source | FID50k |
| --- | --- | ---: |
| Class only, additive, epoch 31 | sampled class labels | 4.3116 |
| CLS only, additive, epoch 31 | train CLS | 3.3163 |
| Class + CLS, additive, epoch 31 | train CLS and labels | 3.3485 |
| CLS only, additive, epoch 31 | **validation CLS oracle** | 2.3243 |
| CLS only, CA8, epoch 31 | train CLS | 3.3290 |
| CLS only, CA8, epoch 58 | train CLS | 3.1485 |

Additive/class jobs were `4718744-4718751`, the validation-oracle
chain was `4718814-4718815`, and the CA8 epoch-58 chain was
`4805154-4805165`. FID measures distribution fit, not memorization or
downstream utility.

### G2: Condition Dependence

The corrected source and epoch-zero controls have no CLS effect. At epochs 10
and 20, paired CLS conditions beat within-class shuffled conditions in
approximately `99.4%-99.6%` of evaluated rows. Fixed-noise output
movement disappears when CLS guidance is zero. The generator therefore uses
CLS, but this does not establish useful novelty.

### G3: Six-Region Capability Screen

At q=.25 and a 50%-of-missing exposure budget, CA8 epoch-31 target CE repair was:

| Added data | Target CE repair |
| --- | ---: |
| Class-only synthetic | 0.0572 |
| Standard retained-anchor image augmentation | 0.2312 |
| Direct retained-anchor SLERP synthetic | 0.2401 |
| Retained-anchor repetition | 0.2570 |
| Exact retained-anchor CLS synthetic | 0.2638 |
| Unique target real reference | 0.3201 |

Exact CLS versus repetition was only `+0.0068` and its interval included
zero. A deterministic additive rerun found exact and SLERP advantages over
standard augmentation of `+0.0293` and `+0.0387`,
respectively, with positive intervals, but this remains a selected six-region
capability result from a generator trained on full IN1K.

### G4: Graph-Geodesic Closure

All graph jobs are complete. At CFG 1, graph minus direct SLERP produced:

- generated support distance: `-1.5689`;
- condition adherence: `+0.0431`;
- endpoint lock: `-0.2792`.

However, in the original additive-AdaLN epoch-31 bank, graph minus
direct target-CE repair was `-0.0745` with CI
`[-0.1522, -0.0168]`. Graph paths were also worse by
`+0.0109` FD-DINO-RP64 and `+0.0077` MMD. In the separately
sampled CA8 epoch-31 bank, graph minus direct was `-0.0410` with CI
`[-0.1080, -0.0035]`. Exact, SLERP, graph, and class-only source-at-1
rates in the original DINOv3/SigLIP source audit were `0.8489`,
`0.5604`, `0.2924`, and approximately zero.

**Decision:** graph paths are a useful geometry/replay diagnostic. Direct SLERP
remains the current repair proposal. Do not launch a broader graph sweep.

### G5: CA1 Closure

At matched epoch 39 over 24 exact conditions and two diffusion seeds, CA1 versus
CA8 had mean re-encoded cosine `0.8572` versus `0.8553`,
source-at-1 `85.42%` versus `87.50%`, and diversity
`0.1454` versus `0.1559`. CA1 won only 18 of 48 paired
cosine comparisons and later became unstable.

**Decision:** this one-trajectory diagnostic does not establish conditioner
equivalence or an intrinsic CA1 defect. It provides no reason to continue CA1.

## Active Diagnostics At The Last Scheduler Refresh

| ID | Job | Current meaning |
| --- | --- | --- |
| A1 | `4788862` SA1 | Epoch 36 complete, loss `0.8880`, finite; conditioning-route stability diagnostic |
| A2 | `4789176` bounded geometry AdaLN | Epoch 25 complete, loss `0.9209`, projected geometry stable; stabilization diagnostic |
| A3 | `4847544-4847546` BREEDS Living17 | 30k-step scratch causal known-subpopulation screen; 19 manifests, six superclasses, one model seed |

The BREEDS arms are target-subclass deletion, same-superclass non-target
deletion, and count-preserving target reallocation at q=.25. The current run
does not include restoration or synthesis and is not a prospective selector
test.

## Gated Experiments

### E1: Adequate-Training Distinct-Real Headroom

**Question:** after learner convergence, is distinct local real data still
better than repeated or augmented retained anchors?

- 20-30 new regions, frozen without intervention outcomes.
- Three paired scratch-learner seeds.
- Development learning curve first; freeze the training budget when the next
  checkpoint changes accuracy by less than 0.5 percentage points and NLL by
  less than 0.02.
- Compare deleted baseline, unique target real, retained repetition, true
  image-space augmentation, and same-class non-target real.
- Primary normalized advantage:

  `(repair_unique - repair_retained) / (CE_deleted - CE_full)`.

**Go:** mean normalized advantage at least `0.10`, lower
region-bootstrap interval above zero, and global balanced accuracy no worse
than `0.2` percentage points.
**Stop:** no positive unique-data advantage after convergence or effects are
only global class damage.

### E2: Prospective Retained-Only Allocation

**Question:** can an incomplete dataset identify valuable missing coverage
without inspecting the hidden reserve?

- Freeze known-subpopulation candidates before outcomes.
- Reserve candidate examples and forbid their use in atlas fitting, score
  construction, generator conditions, and hyperparameter choice.
- Compare random, frequency, density/support, learning-dynamics or TADA-style,
  and ADA allocation under the same joint acquisition budget.
- Primary budget: 1% of retained training size.
- Evaluate hidden-subpopulation CE and global balanced accuracy.

**Go:** ADA exceeds the strongest baseline with lower paired interval above zero
and captures at least 10% of the oracle-real repair gain.
**Stop:** target discovery requires reserve identities or validation errors.

### E3: Exclusion-Audited Synthetic Confirmation

**Question:** can targeted synthesis add information without having trained on
the withheld target examples?

- Stable CA8 route.
- At least 20 frozen sparse interiors and three paired downstream seeds.
- Stage-2 training excludes exact target identities and every extracted view of
  those identities.
- Explicitly document encoder, decoder, normalization, and initialization
  exposure; exclusion is about Stage-2 pairs, not a claim of universal
  pretraining disjointness.
- Conditions come only from retained q=.25 anchors.
- Compare exact retained CLS, direct SLERP, class-only, class-only plus matched
  rejection, random same-class CLS, repetition, true augmentation, and unique
  real reference.

**Go:** targeted synthesis has normalized repair advantage at least
`0.10` over the strongest retained-data baseline with lower interval
above zero, while preserving independent class validity and global accuracy.
**Stop:** no advantage over augmentation/repetition, or nearest-source audits
indicate replay.

### E4: Bounded Neighborhood-Pairing Pilot

This is a generator mechanism pilot, not a paper-critical launch.

Use stable CA8 and compare only:

1. exact pairing;
2. 50% exact plus 50% high-purity local-kNN pairing;
3. 50% exact plus 50% random same-class pairing.

Continue each from the same checkpoint for 10k matched steps. Exclude identity
matches across all cached views. Do not call this condition coarsening: it
changes the condition-target supervision relation, while retaining the full
observed CLS vector.

**Go:** local pairing reduces source-at-1 by at least 10 percentage points,
raises fixed-condition diversity by at least 10%, loses no more than two
class-validity points and five region-hit points, and exceeds random same-class
pairing by at least five region-hit points.
**Stop:** it merely retrieves a finite neighbor bank, leaves the target region,
or behaves like random same-class pairing.

Related-image pairing has precedent, including Semantica. Any novelty claim
must rest on the causal repair protocol, not pairing alone.

### E5: Count-Matched Imbalance Secondary Study

If retained, frame this as diffuse versus coverage-concentrated deletion under
the same class-count deficit. It does not estimate a class-imbalance main
effect. Use exact count matching and report interactions only if they are
prespecified.

## Proposed-Run Field Matrix

This table is the launch gate. A run is not ready until every field is
materialized in a frozen config or manifest.

| Field | E1 real headroom | E2 prospective allocation | E3 excluded synthetic | E4 neighborhood pilot | E5 count-matched deficit |
| --- | --- | --- | --- | --- | --- |
| Hypothesis | Distinct local real data retain value beyond exposure after adequate training | Retained-only ADA selects jointly useful additions better than strong selectors | Retained-only targeted synthesis adds information without exact Stage-2 target exposure | Local pairing improves the diversity-validity-localization trade-off over exact and class-random pairing | Concentrated deletion is more damaging than diffuse deletion at equal class counts |
| Arms | full, deleted, random deletion, repetition, true augmentation, partial unique target, non-target real | random, frequency, density/support, learning dynamics/TADA-style, ADA, labelled oracle | exact CLS, SLERP, class-only, class-only plus rejection, random same-class CLS, repetition, true augmentation, unique real | exact, 50% local kNN, 50% random same class | diffuse deletion, concentrated deletion; matched additions only if promoted |
| Allowed data | frozen q=1 development atlas for causal deletion; intervention learner sees only its manifest | retained train only for selection; hidden reserve only after choices freeze | retained q=.25 conditions; deleted reserve only for final evaluation and real reference | Stage-2 training cache excluding same identity across views for paired condition; no evaluation reserve | q=1 training atlas may define artificial deletion; prospective arm retained-only |
| Exclusion policy | new regions/classes, no outcome-based selection | reserve identities, coordinates, missing count, and outcomes inaccessible | exact target IDs and all cached views excluded from Stage-2; inherited component exposure disclosed | self and alternate-view identity excluded; target sampling unchanged | paired arms share exact realized per-class counts |
| Independent units | 20-30 regions/classes | known subpopulation/class, not selected images | at least 20 sparse-interior regions/classes | conditions are repeated measures; training run is the replication unit | class-selection dataset seed/class |
| Target budget | q=.25 primary; partial real restoration | joint additions equal to 1% retained train | q=.25 sparse interiors | 10k matched continuation steps | 50 selected classes at fixed realized counts |
| Exposure budget | additions matched by presentations and optimization | identical number of acquired unique examples and training compute | 50% of missing primary; attempted and accepted generations both counted | identical batches, target distribution, optimizer policy, and steps | identical class counts and downstream compute |
| Primary metric | normalized target-CE advantage over strongest retained baseline | realized hidden-subpopulation CE and global balanced accuracy | normalized target-CE advantage over strongest retained baseline | source retrieval and diversity at matched validity/localization | target CE difference, concentrated minus diffuse |
| Practical threshold | normalized advantage >=0.10, lower CI >0, global BA loss <=0.2 pp | lower paired CI >0 and >=10% oracle-real gain | normalized advantage >=0.10, lower CI >0, validity/global gates pass | source@1 -10 pp, diversity +10%, validity -2 pp max, region hit -5 pp max, local hit +5 pp over random | predeclare minimum after pilot variance; no launch until set |
| Stop decision | stop if no distinct-data headroom after convergence | stop if reserve knowledge is required or no gain over strongest baseline | stop if no exposure-baseline advantage or replay audit fails | stop if neighbor-bank replay, region escape, or no local-over-random gain | keep secondary unless interaction is paper-relevant |

## Historic Provenance Availability

| Evidence | Launch provenance | Canonical source |
| --- | --- | --- |
| IN100 eight-region DINOv2 pilot | historic job IDs are incomplete in this curated repository | frozen manifests, evaluations, and original RAE notes |
| DINOv3/SigLIP2 transfer | historic arrays are retained in RAE notes/artifacts; not all launch scripts are exported here | immutable aligned manifests and evaluation CSVs |
| Clean IN1K confirmation | `4688448`, `4688460`, `4697966`, `4737415-4737421`, `4749150`, `4749154`, `4749209`, `4749211` | clean selection metadata and paired analysis |
| Scratch ResNet18 lean-six | `4762564`; evaluator `4798609` | committed 30k config and model metadata |
| Additive/class FID | `4718744-4718751` | FID artifacts and checkpoint paths |
| Validation-CLS oracle FID | `4718814-4718815` | oracle-condition FID artifact |
| CA8 epoch-58 FID/repair | `4805154-4805165` | epoch-58 comparison artifact |
| Graph sidecars | `4778513_[0-4]`, `4778514`, `4778515-4778522`, `4779693` | graph submission ledgers and reports |
| CA1 matched audit | training job `4736515`; exact evaluator job IDs not exported into ADA | matched epoch-39 report |
| SA1/bounded AdaLN | `4788862` and `4789176` | live logs; final quality evaluations pending |
| BREEDS Living17 | `4847544-4847546` | frozen config, 19 manifests, eventual evaluator |

Unavailable provenance remains explicitly unavailable; filenames are not used
to invent missing scheduler history.

## Decision Order

1. Finish and evaluate BREEDS, SA1, and bounded AdaLN; do not infer outcomes
   from training loss alone.
2. Run E1 before another expensive generator.
3. If E1 establishes headroom, run E2.
4. Run E3 only after the exclusion manifest and matched controls are frozen.
5. Run E4 only as a bounded mechanism pilot and stop broad injection or graph
   sweeps.
6. Keep E5 secondary unless class-count interaction becomes a central claim.

## Pretraining Boundary

DINOv2 supplies a strong ImageNet-shaped semantic prior. Web-source
deduplication is not a certificate of zero exact exposure across the complete
pretraining mixture, which also includes ImageNet-22K. A scratch downstream
learner removes learner pretraining, not atlas pretraining. Likewise, a
Stage-2-excluded generator does not erase exposure inherited from the frozen
encoder, decoder, normalization statistics, or initialization. Each claim must
name exactly which component and sample identity was excluded.

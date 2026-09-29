ADA — Independent scientific review and Codex decision note

Review date: 2026-09-07
Reviewed repository: David-Drexlin/ADA
Reviewed commit: 6b82921d9712e78ad12535047ab5d0192c9c4b05
Previous reviewed commit: ecadfd831f4ae8439695add5226ea7beb1d11b6a
Suggested repository location: docs/independent-review-2026-09-07.md

Scope and provenance

The handoff and all seven documents in its reading order were reviewed. The commit comparison and selected configuration were also checked. Experimental values below are repository-reported results, not independently reproduced measurements. Hydra checkpoints, prediction files, images, launch overrides, and live scheduler state were not inspected. No repository files, jobs, or checkpoints were modified by this review. The new commit updates documentation; it does not establish that previously identified implementation concerns were fixed. [R1–R9]

This is a scientific assessment and proposed decision protocol, not authorization to launch all experiments or a claim that the proposals are already implemented. Historical results should retain their original provenance. Freeze new protocols before confirmation; do not retrospectively relabel exploratory results as confirmatory.

Executive decision

Keep two distinct questions, not three new projects:

ADA, the primary paper: can observed-data-only selection identify useful local augmentation targets, and can a chosen augmentation action improve downstream performance beyond strong matched alternatives?

CLS-conditioning study: what information and variation does the conditional generator preserve, and how do training relationships and conditioning routes affect generalization? The currently coherent deliverable is a technical report. Promote it to a full independent paper only after a distinctive, replicated result beyond established image-variation behavior.

Do not replace the second question with a new atlas-distribution optimizer. Do not open a language-conditioning or global-manifold-learning project now. Geometry, guidance, and re-encoding belong as supporting tools.

Closing CA1 is a sensible engineering decision. Preserve the audit, but do not describe the 24-condition screen as a statistical equivalence study or a proof that one-token cross-attention is intrinsically unstable. One bounded neighborhood-pairing pilot is justified as a mechanism experiment; another unrestricted model sweep is not. [R6]

What changed in this snapshot

The new documentation explicitly identifies the native RAEv2 pipeline, including the official DINOv2-L K=1 decoder and normalization assets. stage1.RAE is an internal class name, not evidence of using the older RAE codebase. [R2, R7]

At matched epoch 39, the CA1/CA8 audit reports:

Quantity

CA1

CA8

Mean re-encoded condition cosine

0.8572

0.8553

Source retrieval at rank 1

85.42%

87.50%

Fixed-condition diversity

0.1454

0.1559

CA1's median paired cosine difference is negative and it wins 18 of 48 image comparisons. Combined with the documented later instability, this is adequate reason not to spend on another full CA1 rescue. The 48 images are repeated measurements across 24 conditions, not 48 independent training replications. [R6]

SA1 and bounded AdaLN are reported as finite at epochs 29 and 19, respectively, at the recorded scheduler refresh. These are training-state observations, not quality, repair, or completed-run results. Do not infer their current state from this note. [R2, R6]

Strongest defensible current claims

ADA: Under the documented fixed-atlas interventions and specified learners, removing local semantic support causes localized damage; partially restoring unique local examples is more effective than restoring equal-count nonlocal examples. Some important effects transfer across frozen feature heads. Scratch training provides a preliminary screen, not yet a broad final-performance confirmation. [R2]

Generation: CLS conditions provide useful regional targeting relative to generic synthetic controls. Stable CA8 does not establish a broad information advantage over retained-anchor exposure. The six-region report gives exact-CLS repair 0.2638 versus repetition 0.2570, with a paired difference interval spanning zero. The later deterministic additive-AdaLN comparisons against standard augmentation are positive but remain selected six-region capability estimates. [R4]

The evidence does not yet establish prospective deficit discovery, broad superiority to strong augmentation, exposure-independent synthetic acquisition, or a general law equating sparsity with data value. [R2, R4]

Pretraining: narrow the claim, do not abandon the atlas

A fixed pretrained atlas is a legitimate measurement and selection tool. Its pretraining does not, by itself, invalidate an intervention contrast. It limits claims of prior-free discovery and makes independent learners and domains important.

The pretraining note correctly distinguishes a pretrained atlas from a generator trained on the exact experimental dataset. However, web-pool deduplication is not a certificate of zero ImageNet exposure for the full encoder. DINOv2 Appendix A.3 describes relative deduplication of its uncurated source, while Table 15 also lists ImageNet-22k as included "as is". Do not claim that the encoder never saw exact ImageNet-related images without an identity-level audit of all components. This observation also does not prove memorization. [R3, P1]

Scratch downstream training removes the downstream learner's pretrained features; it does not remove the atlas prior. Excluding Stage-2 training pairs does not automatically exclude decoder training, normalization assets, initialization checkpoints, alternate crops, or near-duplicate source identities. Document these layers separately. [R7]

Highest-value remaining experiments, in order

E1 — Establish robust, useful real-restoration headroom

First recover the actual resolved historical scratch configuration. The committed configuration specifies 1,000 steps at batch size 128. If not overridden, that is 128,000 presentations, approximately one ImageNet-100 dataset-sized pass, not convincing convergence evidence. Do not merely repeat that schedule with additional seeds. [R8]

On development data, establish a suitable learning curve and freeze training duration and checkpoint rules. Then confirm on new regions/classes with paired initialization seeds. Include full data, regional deletion, same-class random deletion, repetition, stochastic anchor augmentation, partial unique local restoration, and equal-count nonlocal real restoration. Preserve the meaning of any count-preserving control.

Primary diagnostic: whether distinct local real additions improve target loss beyond the strongest matched retained-data augmentation after adequate training. Report classification error alongside CE and report unaffected-region performance. A CE gain alone should not be narrated as an accuracy gain.

Start with a bounded screen; expand independent region/class units and learner seeds only after its integrity and learning behavior are credible. Approximately 20–30 new regions and three paired learner seeds can be a planning target, not a universal power guarantee. Select the final sample size from region-level variability and a predeclared minimum useful effect. A second scratch architecture should confirm the primary contrast rather than repeating every exploratory arm.

E2 — Prospective regional selection using a real-data diagnostic

Use one BREEDS configuration, such as Living-17, to obtain interpretable known subpopulations. Its superclass/fine-class construction provides semantic ground truth, not independence from ImageNet pretraining. Start with partial deficits, such as q=0.25 or q=0.5; reserve q=0 for a separately declared zero-anchor setting. [R3, P2]

Let the algorithm rebuild its regions and scores from retained data only. Hide the full-data target identity, deleted coordinates, missing count, and corruption specification. Use a budget based on an observable quantity, such as a fraction of retained dataset size. A hidden restoration reserve can evaluate the selections but must not supply embeddings to the selector.

Compare random, class-frequency, density-only, learner-difficulty/learning-dynamics, and the proposed combined selector. Distinguish a reproduced TADA pipeline from a simplified TADA-style selector. Fit selector hyperparameters on development data, then freeze them. [P3]

Evaluate the actual jointly augmented dataset, not only a correlation with regional error or a sum of independently estimated region gains. The selection claim fails if it does not improve realized real-addition utility over strong controls on new units. A useful causal data-value result remains possible without synthesis, but counterfactual training-data prediction itself has precedent in Datamodels. [P4]

E3 — One exclusion-audited synthetic repair confirmation

Use a stable fixed route, preferably CA8 unless matched evidence selects otherwise. Train or use a checkpoint compliant with the declared exclusions. Fine-tuning a full-data-exposed Stage-2 checkpoint does not erase that exposure. Exclude identities across all cached views; a checkpoint reused across experiments must satisfy each experiment's exclusion policy.

Separate same-generator/different-selector contrasts from same-selector/different-action contrasts. Include exact retained CLS, one supported variation policy such as direct SLERP, matched anchor repetition/augmentation, random CLS allocation with matched class allocation, class-only synthesis, and unique-real restoration as a reference. Include a strong targeted augmentation comparator where feasible.

Measure attempted generations as well as accepted images, especially if filtering is used. Class-only generation with regional rejection is a valuable efficiency control: it distinguishes faster targeting from better examples conditional on reaching a region.

The primary synthetic contrast is against the strongest matched retained-data action, not just generic synthesis. Define the useful effect threshold before evaluation. Wide intervals mean inconclusive; an upper confidence limit below the practical threshold supports stopping expansion in that setting.

Neighborhood pairing versus region-plus-residual

Recommendation

Neighborhood pairing is the cleaner first intervention into the condition-target training relationship. Use the stable CA8 route, preserve the target sampling schedule, and run a bounded matched continuation study before considering new full-scale training.

Do not claim a novel general principle merely because pairing reduces reconstruction-like behavior. Semantica explicitly compares same-image reconstruction with related-image pairing, filters pairs by similarity, and includes an ImageNet label-grouped baseline. ADA's contribution must be a more specific finding about local semantic compatibility, source dependence, transfer, or downstream utility. [P5]

Why a bounded residual is not automatically a coarse condition

If the condition is (region_id, residual) with residual = c - prototype(region_id), the original c is recoverable. Multiplication by a nonzero scalar, or a mathematically invertible bounded transform, does not guarantee information loss. A norm cap may also leave all in-range vectors untouched.

Thus distinguish forward-amplitude control from condition information reduction. Region-only conditioning is a true shared condition. Quantization, stochastic corruption, dropout, or a lossy residual representation can change information, but their effective loss of source information must be measured rather than inferred from smaller magnitudes or fewer dimensions alone.

Neighborhood pairing is not literally coarsening the observed vector either: it changes which targets are valid under a condition. It is a clean test of the training relation, not proof that the original vector contained less information.

Small pilot design

Use three matched arms:

Arm

Condition given the same target image

Exact continuation

Its normal same-source CLS

Local pairing

A frozen mixture of exact CLS and distinct-source, same-class local-neighbor CLS

Class-random pairing

The same mixture weight, with a distinct-source random same-class CLS

A 50% replacement probability is a reasonable development starting point, not an established optimum. Hold architecture, target batches, training steps, optimizer reset/resume policy, and sampling configuration fixed. If resource-limited, run exact/local first, but class-random is required before attributing an effect specifically to local geometry.

Sample targets uniformly according to the original training distribution. Audit how frequently conditions are reused: nearest-neighbor hubs can change condition frequencies. Distinguish changed pairing from accidental target or class reweighting. Exclude identical source IDs across crops, and inspect whether candidate neighbors differ only in nuisance/background or violate fine-grained semantics.

A matched continuation experiment diagnoses adaptation from the current checkpoint. It does not establish a from-scratch training law or clean no-exposure repair claim. If it is promising, confirm the selected comparison in an appropriately excluded setting.

Evaluation must match the new conditional task

Measure semantic-region hit rate, independent validity, same-condition diversity, nearest-condition-source retrieval, and nearest-training-image similarity. Include a real-anchor augmentation/retrieval baseline and use source-matched real-image validity, not only a generic validation accuracy reference.

Use held-out source images and held-out local target examples. A model that learns to replay each conditioning anchor's finite training neighbors can look diverse and successful in a small bank; it has not necessarily learned useful local generalization.

Compare diversity at matched semantic fidelity/validity. Lower source cosine can be expected when the supervision now permits neighboring targets. Treating it as failure would prejudge the answer; treating it as sufficient success would be equally wrong. Report both behavior changes and utility.

Region-only or genuinely lossy region-plus-residual conditions should be a later control, not an immediate six-model sweep. Prototype-vector conditioning can preserve the existing numerical interface better than introducing an independently learned region-ID table, but it still changes the conditional task.

The exact-25%-total protocol needs narrower interpretation

The two proposed arms are a useful count-matched comparison of random versus concentrated deletion under the same class imbalance. They do not separately identify an imbalance main effect because both are imbalanced. [R5]

A full 2×2 design would cross balanced/imbalanced class counts with diffuse/concentrated within-class deletion, with equal total retained counts. That is optional if the paper only claims a count-matched location effect; do not multiply the experiment size unnecessarily.

With unequal realized class sizes, exactly halving 50 selected classes and removing exactly 25% globally need not coincide. Define the quota rule explicitly, and use identical realized per-class counts across paired arms. Distinguish the budget based on hidden deleted count in an oracle diagnostic from an observable budget in the prospective experiment.

Concentrated deletion changes difficulty and semantic composition as well as coverage. Matching counts does not isolate a single geometric variable. Freeze evaluation weights and declare any zero-anchor regions.

The interpretation table currently overstates several implications. Equal repair by class-only synthesis does not uniquely prove that count is the dominant deficit. Matching repetition does not prove generated examples contain no new information. Unique real restoration is an empirical comparator, not a mathematical upper bound for every learner and finite budget. Phrase these as evidence consistent with an explanation, then test competing explanations.

Report/paper boundary

CLS technical report or conditional-generalization paper

ADA paper

Matched conditioning routes; stability; guidance; condition fidelity; held-out-source dependence; neighborhood-pairing mechanism

Local data-value prediction; prospective target selection; deletion/restoration controls; exposure-audited repair; budget efficiency; multiple simultaneous deficits

Share infrastructure, checkpoints, representation definitions, and clearly attributed technical background. A report may transparently state that preliminary downstream capability comparisons did not establish broad augmentation superiority. Do not publish the same deletion/restoration matrix as the principal contribution of both works.

Keeping evidence out of the report does not manufacture novelty for ADA. ADA needs genuinely additional scientific results, principally prospective allocation and a rigorous repair test. The report is independently coherent as engineering evidence even if it does not yet meet the standard for a separate full CVPR paper. Public-report timing and venue rules should be checked separately before submission; this review does not infer current policy.

Stop and escalation rules

Stop further full CA1 rescue runs absent a new specific failure diagnosis. Do not infer that CA8 is universally stable from one completed training trajectory. Finish evaluating existing SA1 and bounded-AdaLN work under the existing compute plan; their mere completion does not require a new downstream factorial.

Stop broad geometry/injection expansion until the primary repair gates resolve. Keep direct SLERP as the simple current proposal. A neighborhood pilot may proceed as bounded mechanism work, but cannot replace E1–E3.

Escalate neighborhood training only if it improves the validity–localization–variation trade-off on held-out units, beyond ordinary guidance reduction and class-random pairing. Escalate the synthetic-repair claim only if the exclusion-audited comparison clears a prespecified practically useful effect against the strongest matched retained-data baseline.

If unique real additions help but synthesis does not, retain the causal data-value result and stop the current synthetic expansion. If even unique real additions provide no headroom after adequate training, reconsider whether this setting measures an information deficit rather than optimization, weighting, or metric effects.

Immediate Codex deliverable

Produce a resolved experiment-and-evidence ledger before expensive launches. For every proposed run record the hypothesis, compared arms, allowed data, source-exclusion policy, independent statistical units, target and exposure budgets, primary metric, practical threshold, and stopping decision. Attach actual historic launch metadata where available and explicitly mark unavailable provenance.

Reconcile documentation language about matched epochs, live-job timestamps, graph outcomes, and deployment versus oracle conditions. Previous source-level concerns should be verified against historical execution rather than assumed repaired or assumed fatal. Do not rewrite the architecture as part of this review.

The next substantive result should answer a scientific decision, not merely add another completed job.

Sources

All repository links below are pinned to the reviewed commit. Source-derived claims are referenced above; recommendations and mathematical observations are the reviewer's proposed analysis.

[R1] [Research handoff](https://github.com/David-Drexlin/ADA/blob/6b82921d9712e78ad12535047ab5d0192c9c4b05/docs/chatgpt-handoff.md)

[R2] [Project status](https://github.com/David-Drexlin/ADA/blob/6b82921d9712e78ad12535047ab5d0192c9c4b05/docs/project-status-2026-09-07.md)

[R3] [Pretraining prior and BREEDS](https://github.com/David-Drexlin/ADA/blob/6b82921d9712e78ad12535047ab5d0192c9c4b05/docs/pretraining-prior-and-breeds.md)

[R4] [CA8 repair report](https://github.com/David-Drexlin/ADA/blob/6b82921d9712e78ad12535047ab5d0192c9c4b05/docs/ca8-downstream-result.md)

[R5] [25%-total protocol](https://github.com/David-Drexlin/ADA/blob/6b82921d9712e78ad12535047ab5d0192c9c4b05/docs/class-imbalance-25pct-protocol.md)

[R6] [Conditioner frontier](https://github.com/David-Drexlin/ADA/blob/6b82921d9712e78ad12535047ab5d0192c9c4b05/docs/model-training-frontier-2026-09-07.md)

[R7] [Generator models](https://github.com/David-Drexlin/ADA/blob/6b82921d9712e78ad12535047ab5d0192c9c4b05/docs/generator-models.md)

[R8] [Scratch ResNet18 configuration](https://github.com/David-Drexlin/ADA/blob/6b82921d9712e78ad12535047ab5d0192c9c4b05/configs/ada/actionability/in100_imageclf_resnet18_scratch_deletion_probe.yaml)

[R9] [Graph-geodesic protocol and outcomes](https://github.com/David-Drexlin/ADA/blob/6b82921d9712e78ad12535047ab5d0192c9c4b05/docs/graph-geodesic-conditioning.md)

[P1] Oquab et al., DINOv2: Learning Robust Visual Features without Supervision, Appendix A and Table 15. [https://arxiv.org/html/2304.07193v2](https://arxiv.org/html/2304.07193v2)

[P2] Santurkar et al., BREEDS: Benchmarks for Subpopulation Shift. [https://arxiv.org/abs/2008.04859](https://arxiv.org/abs/2008.04859)

[P3] Nguyen et al., Do We Need All the Synthetic Data? Targeted Image Augmentation via Diffusion Models, TADA. [https://arxiv.org/html/2505.21574v3](https://arxiv.org/html/2505.21574v3)

[P4] Ilyas et al., Datamodels: Predicting Predictions from Training Data. [https://arxiv.org/abs/2202.00622](https://arxiv.org/abs/2202.00622)

[P5] Kumar et al., Conditional Diffusion on Web-Scale Image Pairs leads to Diverse Image Variations, Semantica, especially Sections 4–5, 7, and 8.3. [https://arxiv.org/html/2405.14857v3](https://arxiv.org/html/2405.14857v3)
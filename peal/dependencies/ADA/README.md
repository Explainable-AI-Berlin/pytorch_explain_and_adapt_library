# ADA: Actionable Data Atlases

Research code for identifying under-covered semantic regions, validating their
causal data value through controlled deletion and restoration, and evaluating
representation-conditioned generative repair.

For a compact research handoff, start with
[docs/chatgpt-handoff.md](docs/chatgpt-handoff.md),
[docs/project-status-2026-09-08.md](docs/project-status-2026-09-08.md), and the
[resolved experiment ledger](docs/experiment-evidence-ledger-2026-09-08.md).
The preserved [independent scientific review](docs/independent-review-2026-09-07.md)
records the external critique that the ledger resolves. The current generator
architecture audit is in
[docs/model-training-frontier-2026-09-07.md](docs/model-training-frontier-2026-09-07.md).

## Repository scope

This repository contains:

- `src/ada/`: atlas, actionability, and interpretability code;
- `configs/ada/`: frozen experiment configurations;
- `scripts/ada/`: local and Slurm workflow entry points;
- `scripts/generator/`: generator diagnostics and CLS export utilities;
- `tests/ada/`: unit tests for the ADA core;
- `integrations/raev2/`: the pinned RAEv2 integration used for CLS-conditioned
  latent diffusion.

It deliberately does **not** contain ImageNet images, embeddings, latent caches,
checkpoints, job logs, or generated samples. On Hydra, collaborator-facing data
live under `/home/space/pathomics/ADA`.

## Setup

```bash
git clone --recurse-submodules <ADA_GITHUB_URL>
cd ADA
python -m venv .venv
source .venv/bin/activate
pip install -e '.[analysis,vision,dev]'
cp .env.example .env
```

To reproduce the current generator integration:

```bash
bash integrations/raev2/apply_overlay.sh
```

The script checks the pinned upstream RAEv2 commit before applying the explicit
development patch and new overlay files.

## Tests

```bash
pytest -q tests/ada
pytest -q integrations/raev2/overlay/tests
```

## Generator models

The three matched class-only, CLS-only, and class+CLS model weights are stored
as compact EMA bundles outside Git. A separately trained CA8 cross-attention
CLS model is distributed as a stable architecture ablation. See
[docs/generator-models.md](docs/generator-models.md) for the pinned decoder,
model manifest, training launchers, and unified sampling commands.

## Condition geometry

The release includes direct SLERP and finite-atlas graph-geodesic CLS paths,
plus a condition-bank builder that can feed the same RAEv2 sampler. See
[docs/graph-geodesic-conditioning.md](docs/graph-geodesic-conditioning.md) for
the frozen construction, matched downstream results, and interpretation.

## Data and artifacts

See [docs/data-access.md](docs/data-access.md). The shared DINOv2-L statistics
bundle contains raw CLS vectors only; patch latents are intentionally excluded.

## Status

This is research code under active development. ImageNet-100 development and
the clean ImageNet-1K causal confirmation are complete. A 30,000-step scratch
ResNet18 screen supports architecture-level transfer, but needs broader
replication. Targeted synthesis beats generic synthetic controls but has not
yet established a broad, leakage-resistant advantage over matched retained-data
augmentation or repetition. Adequate-training real-data headroom, prospective
retained-only allocation, and a Stage-2 exclusion-audited generator are the
three frozen next gates.

## Third-party code

RAEv2 is included as a pinned Git submodule and is distributed upstream under
CC BY-NC 4.0. See [integrations/raev2/README.md](integrations/raev2/README.md)
and the upstream license before redistribution or commercial use.

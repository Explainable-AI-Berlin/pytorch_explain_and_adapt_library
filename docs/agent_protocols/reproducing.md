# Reproduction Protocol

> **For cluster-deployed execution agents.** This protocol defines how to run reproducible PEAL experiments across multiple seeds. Always read [`execution_agent.md`](execution_agent.md) first for general responsibilities.

## Overview

Reproduction in PEAL means running the **same experiment** with **different random seeds** to validate that results are consistent and not artifacts of a specific initialization. Each seed gets its own isolated data and output directories.

## ⚠️ Critical Rule: Read-Only Codebase

**Reproduction agents must NEVER modify any file in the codebase.** Your role is strictly to:

1. **Execute** the scripts as specified in the reproduction script
2. **Report** results, errors, and metrics via Mattermost and the handoff file
3. **Stop and report** if a script fails — do not attempt to fix it

If a reproduction run fails:
- Report the exact error to the human via Mattermost
- Do **not** edit Python files, config files, YAML files, or any other source file
- Do **not** apply bug fixes, even trivial ones
- The human will decide whether to fix the issue themselves or hand it to a debugging agent

This is different from the debugging agent role, which **is** allowed to make minimal code fixes. Reproduction agents are strictly execute-and-report.

## Communication

During reproduction runs, maintain contact with the human researcher via **Mattermost** (see [Mattermost Communication](#mattermost-communication) below). Report progress, failures, and aggregated results there. Accept suggestions from the human via Mattermost.

## Seeds

PEAL uses seeds from the set **{0, 1, 2, 3}**:

| Seed | Role |
|---|---|
| `0` | Default seed; used for the primary / development run |
| `1` | First reproduction seed |
| `2` | Second reproduction seed |
| `3` | Third reproduction seed |

A full reproduction requires running the experiment with **all 4 seeds** (0, 1, 2, 3).

## Directory Isolation

Each seed gets its own completely separate directory tree. This prevents any cross-contamination between runs:

```bash
# For seed N, set environment variables:
export PEAL_RUNS="${PEAL_RUNS}/${SEED}"    # e.g., ./peal_runs/2
export PEAL_DATA="${PEAL_DATA}/${SEED}"    # e.g., ./datasets/2
```

### Directory Layout Example

```
./peal_runs/0/  ← seed 0
./peal_runs/1/  ← seed 1
./peal_runs/2/  ← seed 2
./peal_runs/3/  ← seed 3

./datasets/0/   ← seed 0
./datasets/1/   ← seed 1
./datasets/2/   ← seed 2
./datasets/3/   ← seed 3
```

> **Note**: Seed 0 uses the `0` subdirectory explicitly.

## Running a Reproduction

### Step 1: Set Up Seed Environment

For each seed, set the environment variables before running any script:

```bash
SEED=2
export PEAL_RUNS="./peal_runs/${SEED}"
export PEAL_DATA="./datasets/${SEED}"
```

### Step 2: Pass the Seed Flag

**Every** Python script must be called with the `--seed` flag:

```bash
# Training a predictor
apptainer run --nv python_container.sif \
  python train_predictor.py --config "<PEAL_BASE>/configs/..." --seed $SEED

# Training a generator
apptainer run --nv python_container.sif \
  python train_generator.py --config "<PEAL_BASE>/configs/..." --seed $SEED

# Running CFKD
apptainer run --nv python_container.sif \
  python run_cfkd.py --config "<PEAL_BASE>/configs/..." --seed $SEED

# Running an explainer
apptainer run --nv python_container.sif \
  python run_explainer.py --config "<PEAL_BASE>/configs/..." --seed $SEED

# Running an adaptor
apptainer run --nv python_container.sif \
  python run_adaptor.py --config "<PEAL_BASE>/configs/..." --seed $SEED

# Running component analysis
apptainer run --nv python_container.sif \
  python run_component_analysis.py --config ... --sd_config ... --seed $SEED
```

### Step 3: Run the Full Pipeline Per Seed

For each seed, run the **entire pipeline** from the reproduction script. Example for DiDAE on the Square dataset (seed=2):

```bash
export SEED=2
export PEAL_RUNS="./peal_runs/${SEED}"
export PEAL_DATA="./datasets/${SEED}"

# Step-by-step (each command gets --seed $SEED)
apptainer run --nv python_container.sif \
  python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square_classifier_unpoisoned.yaml" --seed $SEED

apptainer run --nv python_container.sif \
  python train_generator.py --config "<PEAL_BASE>/configs/cfkd_experiments/generators/square1k_ddpm_poisoned098.yaml" --seed $SEED

apptainer run --nv python_container.sif \
  python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square1k_classifier_poisoned098.yaml" --seed $SEED

apptainer run --nv python_container.sif \
  python run_cfkd.py --config "<PEAL_BASE>/configs/didae_experiments/adaptors/square1k_resnet18_poisoned098_didae_procrustes_cfkd.yaml" --seed $SEED
```

### Step 4: Run All Seeds

Run seeds sequentially or in parallel (if GPU resources allow):

```bash
# Sequential (one GPU)
for SEED in 0 1 2 3; do
  export PEAL_RUNS="./peal_runs/${SEED}"
  export PEAL_DATA="./datasets/${SEED}"

  apptainer run --nv python_container.sif \
    python train_predictor.py --config "<PEAL_BASE>/configs/..." --seed $SEED
  # ... remaining steps ...
done

# Parallel (multiple GPUs via SLURM)
for SEED in 0 1 2 3; do
  sbatch --gres=gpu:1 --mem=64G --time=24:00:00 \
    --export=ALL,PEAL_RUNS="./peal_runs/${SEED}",PEAL_DATA="./datasets/${SEED}" \
    --wrap="apptainer run --nv python_container.sif python run_cfkd.py --config '<PEAL_BASE>/configs/...' --seed $SEED"
done
```

## Special Cases

### Strict Apptainer Sandboxing

To enforce the read-only codebase rule, you MUST run Apptainer with strict bind mounts. This ensures the container only has write access to the specific data and run directories, and makes the `.git` directory physically inaccessible.

Before running any Apptainer commands, create an empty directory to mask the `.git` folder:
```bash
mkdir -p /tmp/empty
```

Then, execute the container with these specific bindings and the `--no-home` flag to prevent your entire home directory from being automatically mounted:
```bash
apptainer run --nv --no-home \
  --bind $PEAL_BASE:$PEAL_BASE:ro \
  --bind $PEAL_RUNS:$PEAL_RUNS:rw \
  --bind $PEAL_DATA:$PEAL_DATA:rw \
  --bind /tmp/empty:$PEAL_BASE/.git \
  --pwd $PEAL_BASE \
  python_container.sif python train_predictor.py ...
```

- `$PEAL_BASE:$PEAL_BASE:ro` mounts the entire repository as Read-Only.
- `$PEAL_RUNS:$PEAL_RUNS:rw` explicitly grants Read-Write access to the output directory.
- `$PEAL_DATA:$PEAL_DATA:rw` explicitly grants Read-Write access to the dataset directory.
- `/tmp/empty:$PEAL_BASE/.git` physically hides the Git repository from the container, preventing any accidental or malicious history corruption.

### Using Cached Configs During Reproduction

If a seeded run crashes, you can resume from the cached config just like in debugging (see [`debugging.md`](debugging.md)):

```bash
# Resume seed 2 from cache
export PEAL_RUNS="./peal_runs/2"
export PEAL_DATA="./datasets/2"

apptainer run --nv python_container.sif \
  python run_cfkd.py --config "$PEAL_RUNS/<experiment-path>/config.yaml"
```

> **Important**: When resuming from a cached config, the seed is already embedded in the config. You do **not** need to pass `--seed` again.

### Synthetic Datasets

For synthetic datasets (like the Squares dataset), the dataset is generated fresh for each seed. This is correct — different seeds produce different random confounders, and the whole point of reproduction is to verify robustness across different random data.

## Collecting and Reporting Results

### Per-Seed Results

After all seeds complete, collect key metrics from each:

```bash
# TensorBoard logs per seed
tensorboard --logdir_spec \
  seed0:./peal_runs/0/experiment/logs,\
  seed1:./peal_runs/1/experiment/logs,\
  seed2:./peal_runs/2/experiment/logs,\
  seed3:./peal_runs/3/experiment/logs
```

### Aggregation Format

Report results in the handoff as a table with mean ± std:

```markdown
## Reproduction Results

| Metric | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Mean ± Std |
|--------|--------|--------|--------|--------|------------|
| Val Acc (biased) | 0.95 | 0.94 | 0.95 | 0.93 | 0.94 ± 0.01 |
| Val Acc (unbiased) | 0.82 | 0.80 | 0.83 | 0.81 | 0.82 ± 0.01 |
| Train Loss | 0.18 | 0.20 | 0.17 | 0.19 | 0.19 ± 0.01 |

All 4 seeds converged successfully. Results are consistent.
```

### What to Flag

- **High variance** (std > 5% of mean): Indicates instability; investigate
- **One outlier seed**: May indicate a rare initialization issue; document but don't discard
- **All seeds fail**: The bug is systematic, not seed-dependent; escalate via debugging protocol
- **Seed 0 already ran during debugging**: Reuse those results and only run seeds 1-3

## Mattermost Communication

Use your assigned Mattermost bot account to report progress during reproduction runs.

- **Server**: `https://mm.neuro.tu-berlin.de`
- **Bot accounts**: `sidbot0`, `sidbot1`, `sidbot2` (use the one assigned to your session)
- **When to message**: Report per-seed completion, failures, and final aggregated results
- **Accept suggestions**: The human researcher may reply with corrections — follow them
- **Message format**:
  ```
  🔵 REPRODUCTION: <experiment name>
  Seed <N>: ✅ complete | ❌ failed | ⏳ running
  ```
  ```
  📊 RESULTS: <experiment name>
  | Seed | Val Acc (unbiased) | Train Loss |
  |------|--------------------|------------|
  | 0    | 0.82               | 0.18       |
  | 1    | 0.80               | 0.20       |
  | ...  | ...                | ...        |
  Mean ± Std: 0.82 ± 0.01
  ```

## Anti-Patterns

❌ **Do not** modify any source code, config files, or scripts — you are read-only
❌ **Do not** attempt to fix bugs — report them and stop
❌ **Do not** run different seeds in the same `$PEAL_RUNS` directory — results will collide
❌ **Do not** forget to pass `--seed` to every script — this defeats the purpose
❌ **Do not** tune hyperparameters per-seed — use the same config for all seeds
❌ **Do not** cherry-pick seeds — report all 4, even if some look worse
❌ **Do not** modify the source config between seeds — that's a different experiment

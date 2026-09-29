# Debugging Protocol

> **For cluster-deployed execution agents.** This protocol defines how to debug PEAL experiments on the HPC cluster. Always read [`execution_agent.md`](execution_agent.md) first for general responsibilities.

## Overview

Debugging in PEAL follows a **cache-aware, incremental** approach. PEAL caches intermediate results (trained models, generated counterfactuals, etc.) in `$PEAL_RUNS`. When a run crashes, you should **resume from the last cached checkpoint** rather than restarting from scratch.

## Communication

During debugging, maintain contact with the human researcher via **Mattermost** (see [Mattermost Communication](#mattermost-communication) below). Report progress, errors, and proposed fixes there before applying them. Accept suggestions and corrections from the human via Mattermost.

## The Two Config Paths

Every PEAL experiment has **two config file locations**:

### 1. Source Config (in `configs/`)

This is the **original experiment definition** written by the authoring agent. Use this for the **first run** of an experiment.

```bash
# First run — uses the source config
apptainer run --nv python_container.sif \
  python run_cfkd.py --config "<PEAL_BASE>/configs/didae_experiments/adaptors/square1k_resnet18_poisoned098_didae_procrustes_cfkd.yaml"
```

### 2. Cached Config (in `$PEAL_RUNS/`)

When PEAL runs, it copies the resolved config to the output directory (e.g., `$PEAL_RUNS/.../config.yaml`). This cached config contains:
- All resolved paths (no `<PEAL_BASE>` or `$PEAL_RUNS` placeholders)
- The exact parameters used for the run
- References to cached intermediate results

**After the first run (successful or failed), always use the cached config to resume:**

```bash
# Resume from cache — skips already-completed steps
apptainer run --nv python_container.sif \
  python run_cfkd.py --config "$PEAL_RUNS/square1k/colora_confounding_colorb/torchvision/classifier_poisoned098/didae_cfkd/config.yaml"
```

### Why This Matters

| Scenario | Use Source Config | Use Cached Config |
|---|---|---|
| First run of a new experiment | ✅ | ❌ (doesn't exist yet) |
| Run crashed mid-training | ❌ | ✅ (resumes from cache) |
| Tuning hyperparameters | ❌ | ✅ (edit cached config, rerun) |
| After code fix | ❌ | ✅ (resumes with fixed code) |
| Starting completely fresh | ✅ (delete cached dir first) | ❌ |

## Debugging Workflow

### Step 1: First Run

Run the experiment using the source config:

```bash
apptainer run --nv python_container.sif \
  python run_cfkd.py --config "<PEAL_BASE>/configs/didae_experiments/adaptors/<experiment>.yaml"
```

If it completes successfully, you're done. If it crashes, proceed to Step 2.

### Step 2: Diagnose the Error

Read the error traceback carefully. Common categories:

| Error Type | Example | Typical Fix |
|---|---|---|
| **Import / Config** | `ModuleNotFoundError`, `ValidationError` | Fix import or add missing config field |
| **Data / Path** | `FileNotFoundError`, missing prerequisite | Run prerequisite step first, fix path |
| **CUDA / GPU** | `CUDA out of memory`, device mismatch | Reduce batch_size, fix `.to(device)` calls |
| **Shape / Type** | `RuntimeError: size mismatch`, dtype error | Fix tensor operations, add casts |
| **Logic** | Wrong results, NaN loss, non-convergent | Deeper investigation needed |

### Step 3: Fix the Code (Minimal Changes)

Apply the minimal fix to the Python source code. Follow the principles in [`execution_agent.md`](execution_agent.md):
- Fix the **immediate error**, don't redesign
- Add defensive code (type casts, shape checks) rather than changing logic
- Document what you changed and why

### Step 4: Resume from Cache

After fixing the code, **resume from the cached config**, not the source config:

```bash
apptainer run --nv python_container.sif \
  python run_cfkd.py --config "$PEAL_RUNS/<experiment-path>/config.yaml"
```

This skips all completed steps and resumes from where the crash occurred.

### Step 5: Iterate

If the run crashes again at a different point, repeat Steps 2-4. Each iteration should make progress through the pipeline.

### Step 6: Hyperparameter Tuning via Cached Config

If the run completes but results are poor, you can tune hyperparameters **by editing the cached config directly**:

```bash
# Edit the cached config
vim $PEAL_RUNS/<experiment-path>/config.yaml
# Change e.g. learning_rate, batch_size, loss weights...

# Re-run from cache (PEAL will recompute from the changed point)
apptainer run --nv python_container.sif \
  python run_cfkd.py --config "$PEAL_RUNS/<experiment-path>/config.yaml"
```

See [`hyperparameter_tuning.md`](hyperparameter_tuning.md) for tuning strategy.

> **Important**: When tuning via the cached config, also update the source config in `configs/` with the final tuned values so they are preserved in the repository.

## PEAL Script Reference

All scripts support `--config` and most support `--seed`:

| Script | Purpose | Seed Flag |
|---|---|---|
| `train_predictor.py` | Train a classifier / regressor | `--seed <N>` |
| `train_generator.py` | Train a generative model (DDPM, DAE, etc.) | `--seed <N>` |
| `run_cfkd.py` | Run CFKD adaptor (counterfactual generation + finetuning) | `--seed <N>` |
| `run_explainer.py` | Run an explainer on a predictor | `--seed <N>` |
| `run_adaptor.py` | Run a model adaptor | `--seed <N>` |
| `run_component_analysis.py` | Run sparse dictionary / component analysis | `--sd_config <path>` |
| `evaluate_predictor.py` | Evaluate a trained predictor | N/A |
| `generate_dataset.py` | Generate a synthetic dataset | N/A |

## Debugging a Multi-Step Pipeline

Many experiments are multi-step (e.g., DiDAE requires: train predictor → train generator → train DAE → component analysis → CFKD). When debugging:

1. **Identify which step failed** from the error traceback
2. **Check prerequisites**: Does the expected input exist in `$PEAL_RUNS`?
3. **Fix and re-run only the failed step** using its cached config
4. **Continue with subsequent steps** once the failed step completes

Example: DiDAE pipeline for Square dataset:
```bash
# Step 1: Train unpoisoned predictor (prerequisite)
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square_classifier_unpoisoned.yaml"

# Step 2: Train poisoned predictor
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square1k_classifier_poisoned098.yaml"

# Step 3: Train generator
python train_generator.py --config "<PEAL_BASE>/configs/cfkd_experiments/generators/square1k_ddpm_poisoned098.yaml"

# Step 4: Train diffusion autoencoder
python train_generator.py --config "<PEAL_BASE>/configs/didae_experiments/generators/square_diffusion_autoencoder.yaml"

# Step 5: Component analysis
python run_component_analysis.py --config $PEAL_RUNS/square/diffusion_autoencoder/config.yaml \
  --sd_config configs/didae_experiments/sparse_dictionaries/procrustes_sae_square.yaml

# Step 6: Run CFKD (this is where you'd debug most often)
python run_cfkd.py --config "<PEAL_BASE>/configs/didae_experiments/adaptors/square1k_resnet18_poisoned098_didae_procrustes_cfkd.yaml"
# On subsequent attempts:
python run_cfkd.py --config "$PEAL_RUNS/square1k/colora_confounding_colorb/torchvision/classifier_poisoned098/didae_cfkd/config.yaml"
```

## Mattermost Communication

Use your assigned Mattermost bot account to communicate during debugging sessions.

- **Server**: `https://mm.neuro.tu-berlin.de`
- **Bot accounts**: `sidbot0`, `sidbot1`, `sidbot2` (use the one assigned to your session)
- **When to message**: Report crashes, proposed fixes, tuning results, and completion status
- **Accept suggestions**: The human researcher may reply with corrections or alternative approaches — follow them
- **Message format**:
  ```
  🔴 ERROR: <short description>
  Traceback: <last 3 lines of traceback>
  Proposed fix: <what you plan to change>
  ```
  ```
  🟡 TUNING: <parameter> <old_value> → <new_value>
  Reason: <why>
  Result: <metric before> → <metric after>
  ```
  ```
  🟢 COMPLETE: <experiment name>
  Key metrics: <accuracy, loss, etc.>
  Output: <path to results>
  ```

## When to Escalate

Escalate to the authoring agent (via the handoff file) if:

1. The bug is in the core architecture, not a surface-level issue
2. You've tried 3+ fixes for the same error without resolution
3. The fix would require changing a public API or class structure
4. The error is in a dependency (`peal/dependencies/`) that you shouldn't modify

# Hyperparameter Tuning Protocol

> This document defines how the execution agent should approach hyperparameter tuning for PEAL experiments.

## Philosophy

Hyperparameter tuning in PEAL is **goal-directed, not exhaustive**. The execution agent should:
1. Start from the defaults provided by the authoring agent
2. Fix what's broken (debugging) before optimizing (tuning)
3. Change one variable at a time
4. Stop when expected outcomes from the handoff are met

## Tuning Hierarchy

Tune parameters in this order of priority. Stop as soon as results meet expectations.

### Level 0: Stability Fixes (Debug First)
These are not "tuning" — they're fixes to make the code run at all.

| Parameter | When to Touch | Common Values |
|---|---|---|
| `batch_size` | OOM errors | Halve until it fits; compensate with grad accumulation |
| `gradient_accumulation_steps` | After reducing batch size | Set to `original_batch / new_batch` |
| `num_workers` | DataLoader crashes, slow loading | 4, 8, or 0 (for debugging) |
| `precision` | OOM or numerical instability | `fp16`, `bf16`, `fp32` |

### Level 1: Learning Rate
The single most impactful hyperparameter.

**Strategy: Log-scale grid search**
```
candidates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
```

- Run each for a fixed number of steps (e.g., 10% of total training)
- Pick the one with the best validation metric
- If none converge, try `1e-6` or `5e-3` as fallbacks

**Signs of wrong learning rate:**
| Symptom | Diagnosis | Action |
|---|---|---|
| Loss increases | LR too high | Reduce by 5-10x |
| Loss plateaus immediately | LR too low | Increase by 5-10x |
| Loss oscillates wildly | LR too high | Reduce by 2-5x |
| Loss decreases smoothly | LR is good | Keep it |

### Level 2: Training Duration
| Parameter | Guidance |
|---|---|
| `num_epochs` | Monitor validation loss; stop when it plateaus or increases (overfitting) |
| `early_stopping_patience` | If available, set to 5-10 epochs |

### Level 3: Architecture / Component-Specific
Only tune these if Level 0-2 don't achieve expected outcomes.

**For generators (DDPM / Diffusion Autoencoders):**
| Parameter | Range | Notes |
|---|---|---|
| `num_timesteps` | 100, 250, 500, 1000 | More = better quality, slower |
| `noise_schedule` | `linear`, `cosine` | Cosine often better for small images |

**For explainers (SCE / counterfactual):**
| Parameter | Range | Notes |
|---|---|---|
| `num_steps` | 50, 100, 200 | Optimization steps per sample |
| `lambda_*` (loss weights) | 0.01 – 10.0 | Balance competing objectives |

**For adaptors (CFKD):**
| Parameter | Range | Notes |
|---|---|---|
| `finetune_epochs` | 5, 10, 20, 50 | Per CFKD round |
| `num_counterfactuals` | 100, 500, 1000 | More = better but slower |

**For component analysis (DiDAE):**
| Parameter | Range | Notes |
|---|---|---|
| `num_components` | 2, 4, 8, 16 | Depends on expected number of factors |
| `sparsity_lambda` | 0.001 – 0.1 | For sparse dictionary learning |

## Logging & Tracking

### Every Tuning Run Must Log

1. **Config snapshot**: Copy the YAML config used to `$PEAL_RUNS/<experiment>/config.yaml` (PEAL does this automatically)
2. **Seed**: Always set a seed; default to `seed=42` unless specified otherwise
3. **Key metrics**: Final train loss, final val loss, best val accuracy, wall-clock time

### Tuning Log Format

Maintain a tuning log in the handoff under `## Hyperparameters Tuned`:

```markdown
## Hyperparameters Tuned

| Run | Parameter Changed | Value | Val Acc (unbiased) | Train Loss | Notes |
|-----|-------------------|-------|---------------------|------------|-------|
| 1   | baseline          | —     | 0.72               | 0.31       | Default config |
| 2   | lr                | 5e-4  | 0.78               | 0.24       | Better convergence |
| 3   | lr                | 1e-3  | 0.65               | 0.45       | Unstable, reverted |
| 4   | lr=5e-4, bs=32    | —     | 0.81               | 0.19       | **Best** ✓ |

Selected configuration: Run 4 (lr=5e-4, batch_size=32)
```

## Config Changes

When you tune a hyperparameter:

1. **Modify the YAML config file directly** if the tuned value is a better default for this experiment
2. **Document the change** in the handoff
3. **Do not change the Pydantic default** in Python code — that's the authoring agent's call

Example config change:
```yaml
# $PEAL_RUNS/square1k/.../didae_cfkd/config.yaml
# Changed by execution agent: lr 1e-4 → 5e-4 (faster convergence, same final loss)
learning_rate: 5e-4
# Changed by execution agent: batch_size 64 → 32 (OOM on A100-80GB with full model)
batch_size: 32
gradient_accumulation_steps: 2  # Added to compensate batch_size reduction
```

> **Important**: Edit the **cached config** in `$PEAL_RUNS/` for iterating, then copy the final tuned values back to the **source config** in `configs/` so they are preserved in the repository. See [`debugging.md`](debugging.md) for the full cache-aware workflow.

## Mattermost Communication

Report tuning progress via your assigned Mattermost bot account:

- **Server**: `https://mm.neuro.tu-berlin.de`
- **Bot accounts**: `sidbot0`, `sidbot1`, `sidbot2`
- **Accept suggestions**: The human may redirect tuning — always follow their guidance
- **Message format**:
  ```
  🟡 TUNING: <experiment name>
  | Run | Parameter | Value | Val Acc | Notes |
  |-----|-----------|-------|---------|-------|
  | 1   | baseline  | —     | 0.72    | Default |
  | 2   | lr        | 5e-4  | 0.78    | Better ✓ |
  
  Continuing with lr=5e-4, next: trying batch_size=32
  ```

## When to Stop Tuning

Stop and hand back to the authoring agent when:

1. ✅ **Expected outcomes are met** — metrics match what the handoff specified
2. ⏱ **Diminishing returns** — last 3 runs improved by less than 1%
3. 🚫 **Blocked** — the issue is architectural, not parametric (hand back with `blocked` status)
4. 📊 **Good enough** — results are reasonable even if not perfect; document and let the researcher decide

## Anti-Patterns

❌ **Do not** run a massive grid search without checking results incrementally
❌ **Do not** tune parameters that the authoring agent explicitly set with a rationale
❌ **Do not** change the seed to get better numbers — that's p-hacking
❌ **Do not** tune on the test set — only use validation metrics
❌ **Do not** silently change configs without documenting the change

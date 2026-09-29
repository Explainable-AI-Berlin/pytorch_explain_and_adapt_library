# Authoring Agent Protocol

> **You are the Authoring Agent.** You operate on the researcher's local workstation. Your job is to write high-quality, well-tested code and hand it off for GPU execution and validation.

## Identity & Scope

- **Agent name**: Antigravity (may also be Claude Code, Codex, or similar)
- **Environment**: Developer workstation, no GPU, full codebase access
- **Repository root**: The workspace directory containing this file
- **Container**: Changes will ultimately run inside `python_container.sif` (Apptainer) on the cluster

## Your Responsibilities

### ✅ You MUST

1. **Write production-quality Python code** that follows the project's contribution guidelines (see `README.md` §Contribution Guidelines).
2. **Write and update YAML config files** for experiments, following existing patterns in `configs/`.
3. **Write unit tests** for every new component (place in `tests/`). Tests must be CPU-executable.
4. **Write reproduction scripts** for new experiments (place in `reproduction_scripts/`).
5. **Run linting** (`ruff check .`) and fix any issues before handoff.
6. **Run CPU-only unit tests** (`python -m pytest tests/`) and ensure they pass.
7. **Write the handoff file** (`.agent/handoff.md`).
8. **Notify the human** so they can commit and push the changes.
9. **Use Pydantic** for all new configuration classes, following existing patterns.
10. **Preserve existing functionality** — extend, don't replace.

### ❌ You MUST NOT

1. **Run GPU-dependent code** — you have no GPU. Do not attempt training, generation, or evaluation.
2. **Tune hyperparameters** — this is the execution agent's responsibility.
3. **Create large binary files** (models, datasets, `.sif` containers) that would bloat the repository.
4. **Overwrite the execution agent's fixes** without discussion — if you disagree with a fix, flag it in the handoff.
5. **Access files outside the workspace** — see workspace rules.

## Workflow Steps

### Step 1: Understand the Task

Read the researcher's request carefully. Check:
- Is this a new feature, a bug fix, or a refactor?
- Which PEAL components are involved? (explainers, generators, adaptors, predictors, data, training)
- Are there existing configs or reproduction scripts to reference?

### Step 2: Design

For non-trivial changes:
1. Identify which classes and interfaces to extend.
2. Respect the compositional architecture: new components should be swappable via config.
3. Plan new config fields with sensible defaults so existing configs remain valid.

### Step 3: Implement

Follow these PEAL-specific patterns:

```python
# New component pattern (example: new explainer)
# File: peal/explainers/my_explainer.py

from peal.explainers.interfaces import Explainer, ExplainerConfig

class MyExplainerConfig(ExplainerConfig):
    """Configuration for MyExplainer.
    
    Parameters
    ----------
    my_param : float
        Description of param. Default: 0.5.
    """
    explainer_type: str = "MyExplainer"
    my_param: float = 0.5

class MyExplainer(Explainer):
    """Short description.
    
    Parameters
    ----------
    config : MyExplainerConfig
        Configuration object.
    """
    def __init__(self, config: MyExplainerConfig):
        super().__init__(config)
        # ...
```

Key patterns to follow:
- **Factory pattern**: Set `<component>_type` in config to auto-resolve the class.
- **Config inheritance**: Extend the base config (`ExplainerConfig`, `GeneratorConfig`, etc.).
- **NumPy docstring style** for Sphinx compatibility.
- **Black formatter** with 88 columns.
- **Max 500 lines per class, 100 lines per method.**

### Step 4: Write Configs

Place configs in the appropriate experiment directory:
```
configs/
├── sce_experiments/    # SCE paper experiments
├── cfkd_experiments/   # CFKD paper experiments
├── didae_experiments/  # DiDAE paper experiments
└── <new_experiments>/  # Your new experiment set
```

Reference paths using `<PEAL_BASE>`, `$PEAL_RUNS`, and `$PEAL_DATA` variables.

### Step 5: Write Tests

```python
# tests/test_my_component.py
import pytest

class TestMyComponent:
    """Tests for MyComponent — must run without GPU."""
    
    def test_basic_functionality(self):
        # Use small mock data, CPU tensors only
        ...
    
    def test_config_defaults(self):
        config = MyComponentConfig()
        assert config.my_param == 0.5
```

### Step 6: Validate Locally

When validating or running Python scripts locally to test your code, use the following interactive container command:

```bash
bash -ic 'apprun python <the_script_you_want_to_run.py>'
```

**CRITICAL RULE:** Be considerate that the local machine has only **4GB of VRAM**. When writing validation scripts or running generation loops, ensure batch sizes are kept to 1 or very small numbers, and loop through items one by one instead of allocating large batches on the GPU to prevent Out Of Memory (OOM) errors and numerical instability.

```bash
# Lint
ruff check .

# Unit tests
python -m pytest tests/ -v

# Type check (if applicable)
# mypy peal/ --ignore-missing-imports
```

### Step 7: Handoff

1. Write `.agent/handoff.md`
2. Notify the human researcher that the phase is complete so they can commit and push the changes.

### Example Handoff File

```markdown
# Agent Handoff

## Status: ready-for-execution
## From: antigravity
## To: openclaude
## Task-ID: didae-procrustes-v1
## Branch: agent/didae-procrustes-v1

## What Was Done
- Implemented `OrthogonalProcrustesDictionary` in `peal/sparse_dictionaries/`
- Added config `configs/didae_experiments/sparse_dictionaries/procrustes_sae_square.yaml`
- Added unit tests in `tests/test_procrustes_dictionary.py` (all pass on CPU)
- Updated `run_component_analysis.py` to accept `--sd_config` argument

## What To Do Next
- Run `reproduce_didae_results.sh` lines 19-31 (Square dataset Table 1)
- Verify training completes without errors on GPU
- Check that procrustes rotation produces meaningful component axes
- Tune learning rate if convergence is slow (try 1e-4, 5e-4, 1e-3)

## Known Issues
- Untested on GPU — potential CUDA dtype issues with complex tensors
- Batch size 64 may OOM on GPUs < 80GB

## Expected Outcomes
- Training loss should decrease monotonically after warmup
- Component analysis should produce 4 interpretable axes for Square dataset
- CFKD with DiDAE should achieve ≥90% unbiased accuracy on Square

## Files Changed
- peal/sparse_dictionaries/procrustes.py (new)
- peal/sparse_dictionaries/__init__.py (modified)
- configs/didae_experiments/sparse_dictionaries/procrustes_sae_square.yaml (new)
- tests/test_procrustes_dictionary.py (new)
- run_component_analysis.py (modified)
```

## Reviewing Execution Agent Changes

When the human researcher pulls verified changes and starts you:

1. Review the changes — especially:
   - Bug fixes in source code (should they be generalized?)
   - Config changes (are the tuned values sensible defaults or experiment-specific?)
   - Any new files the execution agent created
2. Refactor if needed, maintaining code quality standards
3. Notify the human to merge to `main` if ready

## Communication with Human Researcher

- Always explain your design decisions
- Flag when you're uncertain about scientific correctness (e.g., loss functions, metrics)
- Never make silent assumptions about experiment goals
- If the execution agent's fixes conflict with your design, present both options to the researcher

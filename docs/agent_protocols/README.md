# Agent-Driven Research Workflow

> **For AI Agents**: If you are an AI coding agent (Antigravity, OpenClaude, Claude Code, Codex, or similar), read the relevant protocol in `` **before** taking any action. This file is the high-level overview; the protocol docs contain your operational instructions.

## Overview

PEAL uses a **dual-agent research workflow** that separates code authoring from code execution, debugging, and hyperparameter optimization. This separation is enforced by two distinct environments and two distinct agent roles:

| Aspect | Authoring Agent (Local) | Execution Agent (Cluster) |
|---|---|---|
| **Agent** | Antigravity (or Claude Code / Codex) | OpenClaude (or Claude Code / Codex) |
| **Environment** | Developer workstation (no GPU) | HPC cluster (GPU nodes via SLURM) |
| **Primary role** | Write code, configs, and scripts | Debug, run, optimize, evaluate |
| **Writes to** | Source code, configs, tests, docs | Bug fixes, tuned configs, results |
| **Git management**| Human manages commits/pushes/pulls | Human manages commits/pushes/pulls |
| **Protocol doc** | [`authoring_agent.md`](authoring_agent.md) | [`execution_agent.md`](execution_agent.md) |

## Why This Workflow

1. **GPU scarcity**: The local workstation has no GPU. All training, generation, and evaluation must happen on the cluster.
2. **Iteration speed**: Debugging runtime errors (CUDA, OOM, shape mismatches) and tuning hyperparameters requires tight feedback loops with actual hardware — this is the execution agent's job.
3. **Code quality**: The authoring agent focuses on architecture, design patterns, tests, and documentation without being blocked by cluster queues.
4. **Reproducibility**: All changes flow through git, so every experiment is traceable to a commit.

## The Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AUTHORING AGENT (Local)                      │
│                                                                     │
│  1. Receive research task from human researcher                     │
│  2. Write / modify source code, configs, reproduction scripts       │
│  3. Run linting, type checks, and unit tests (CPU-only)             │
│  4. Write handoff file (.agent/handoff.md)                          │
│  5. Human commits and pushes to remote                              │
│                                                                     │
│  ─── handoff ──────────────────────────────────────────────────────  │
│                                                                     │
│                        EXECUTION AGENT (Cluster)                    │
│                                                                     │
│  6. Human pulls branch on cluster and starts Execution Agent        │
│  7. Run code inside Apptainer container on GPU node                 │
│  8. Debug runtime errors (fix code if needed)                       │
│  9. Tune hyperparameters based on metrics / loss curves             │
│ 10. Validate results against expected outcomes                      │
│ 11. Write handoff file (.agent/handoff.md)                          │
│ 12. Human commits and pushes to remote                              │
│                                                                     │
│  ─── return ───────────────────────────────────────────────────────  │
│                                                                     │
│                        AUTHORING AGENT (Local)                      │
│                                                                     │
│ 13. Human pulls changes locally and starts Authoring Agent          │
│ 14. Review, refactor if needed, human merges to main                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Handoff Protocol

The handoff between agents is mediated by files. There are no direct API calls or message queues between agents. Each handoff is marked by a **handoff file** at `.agent/handoff.md` that describes the current state, what was done, and what the next agent should do. The human will then commit and push these files.

### Commit Convention (For Human)

All agent-related commits should follow this format:

```
[<role>] <type>: <short description>

<body>

Agent: <agent-name>
Handoff-To: <next-agent-or-human>
Task-ID: <optional-task-reference>
```

**Roles**: `authoring`, `execution`
**Types**: `feat`, `fix`, `config`, `debug`, `tune`, `refactor`, `test`, `docs`

Examples:
```
[authoring] feat: add DiDAE sparse dictionary with Procrustes rotation

Implements OrthogonalProcrustesDictionary as a new sparse dictionary type.
Adds config files for square and celeba experiments.
Unit tests pass on CPU with mock data.

Agent: antigravity
Handoff-To: openclaude
Task-ID: didae-procrustes-v1
```

```
[execution] fix: resolve CUDA OOM in DiDAE training at batch_size=64

Reduced default batch_size to 32 in square_diffusion_autoencoder.yaml.
Gradient accumulation steps set to 2 to maintain effective batch size.
Training completes successfully on A100-80GB.

Agent: openclaude
Handoff-To: antigravity
Task-ID: didae-procrustes-v1
```

### Handoff File (`.agent/handoff.md`)

Every time an agent finishes its phase, it writes/updates `.agent/handoff.md`:

```markdown
# Agent Handoff

## Status: ready-for-execution | ready-for-review | blocked
## From: antigravity | openclaude
## To: openclaude | antigravity | human
## Task-ID: <identifier>
## Branch: <branch-name>

## What Was Done
- <bullet list of completed work>

## What To Do Next
- <bullet list of expected next steps>

## Known Issues
- <any issues the next agent should be aware of>

## Expected Outcomes
- <what success looks like for the next phase>

## Files Changed
- <list of changed files relevant to the task>
```

## Branching Strategy

- Feature branches: `agent/<task-id>` (e.g., `agent/didae-procrustes-v1`)
- Both agents work on the **same feature branch**.
- The human researcher merges to `main` after reviewing the execution agent's results.

## Mattermost Communication

All execution agents (debugging, tuning, reproduction) communicate with the human researcher via **Mattermost** during active sessions:

- **Server**: `https://mm.neuro.tu-berlin.de`
- **Bot accounts**: `sidbot0`, `sidbot1`, `sidbot2` — each agent session is assigned one
- Agents **report progress** (errors, proposed fixes, tuning results, completion)
- Agents **accept suggestions** from the human via Mattermost replies
- The human may redirect, correct, or override agent decisions at any time

## Detailed Protocols

- **Authoring Agent**: [`authoring_agent.md`](authoring_agent.md)
- **Execution Agent**: [`execution_agent.md`](execution_agent.md)
- **Debugging**: [`debugging.md`](debugging.md)
- **Reproduction**: [`reproducing.md`](reproducing.md)
- **Hyperparameter Tuning**: [`hyperparameter_tuning.md`](hyperparameter_tuning.md)
- **Handoff Specification**: [`handoff_spec.md`](handoff_spec.md)

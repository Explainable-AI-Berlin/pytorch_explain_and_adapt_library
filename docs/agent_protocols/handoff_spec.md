# Handoff Specification

> This document defines the structure and semantics of agent handoffs in the PEAL dual-agent workflow. Both the authoring and execution agents must follow this specification.

## Overview

Handoffs are the **only communication channel** between agents. They are:
- **Git-based**: encoded as a file (`.agent/handoff.md`) committed to the repo by the human
- **Structured**: follow a defined schema so both agents can parse them reliably
- **Append-friendly**: each handoff adds context rather than replacing previous context

## File Location

```
.agent/
├── handoff.md          # Current handoff state (always reflects latest)
└── handoff_history/    # Optional: archived previous handoffs
    ├── 001_authoring_2026-06-29.md
    └── 002_execution_2026-06-30.md
```

The `.agent/` directory is already in `.gitignore` as `.agent`. **You must remove this entry** or use `.agents` (plural) for other purposes. The handoff directory must be tracked by git.

> **Important**: Update `.gitignore` to track `.agent/handoff.md` and `.agent/handoff_history/` while keeping other agent working files ignored.

## Handoff Schema

Every `handoff.md` must contain these sections with the exact headers shown:

```markdown
# Agent Handoff

## Status: <status>
## From: <agent-name>
## To: <agent-name-or-human>
## Task-ID: <identifier>
## Branch: <branch-name>
## Timestamp: <ISO-8601>

## What Was Done
<completed work>

## What To Do Next
<expected next steps>

## Known Issues
<open problems>

## Expected Outcomes
<success criteria>

## Files Changed
<list of changed files>
```

### Status Values

| Status | Meaning | Set By |
|---|---|---|
| `ready-for-execution` | Code is written, tested on CPU, ready for GPU runs | Authoring agent |
| `ready-for-review` | Execution completed, results available, changes ready for commit | Execution agent |
| `blocked` | Cannot proceed without design changes or human input | Either agent |
| `in-progress` | Agent is actively working (intermediate state) | Either agent |
| `completed` | Task is done, ready for merge to main | Human researcher |

### Field Descriptions

| Field | Required | Description |
|---|---|---|
| `Status` | ✅ | Current handoff state (see table above) |
| `From` | ✅ | Agent that wrote this handoff (`antigravity`, `openclaude`, or other agent name) |
| `To` | ✅ | Intended recipient (`antigravity`, `openclaude`, or `human`) |
| `Task-ID` | ✅ | Unique task identifier matching the branch name suffix |
| `Branch` | ✅ | Git branch name (format: `agent/<task-id>`) |
| `Timestamp` | ✅ | ISO-8601 timestamp of when the handoff was written |
| `What Was Done` | ✅ | Bullet list of completed work with enough detail to understand without reading diffs |
| `What To Do Next` | ✅ | Explicit, actionable steps for the receiving agent |
| `Known Issues` | ✅ | Any problems, warnings, or uncertainties (write "None" if none) |
| `Expected Outcomes` | ✅ | Measurable success criteria (metrics, file paths, behavioral descriptions) |
| `Files Changed` | ✅ | All files added, modified, or deleted with brief descriptions |

### Optional Sections

These sections may be added as needed:

| Section | When to Use |
|---|---|
| `## Results` | Execution agent reporting metrics, paths to logs/outputs |
| `## Bugs Fixed` | Execution agent documenting runtime fixes |
| `## Hyperparameters Tuned` | Execution agent documenting config changes |
| `## Design Decisions` | Either agent explaining non-obvious choices |
| `## Recommended Follow-ups` | Execution agent suggesting refactors for authoring agent |
| `## Dependencies` | If prerequisite tasks must complete first |
| `## Reproduction Commands` | Exact shell commands to reproduce the experiment |

## Handoff Flow Examples

### Happy Path: New Feature

```
Authoring Agent                     Execution Agent
     │                                    │
     │  1. Write code + tests             │
     │  2. Write handoff                  │
     │  3. Notify human                   │
     │     Status: ready-for-execution    │
     │ ──────────────────────────────────> │
     │                                    │  4. Read handoff
     │                                    │  5. Debug + run + tune
     │                                    │  6. Write fixes + configs
     │                                    │  7. Write handoff
     │                                    │  8. Notify human
     │                                    │     Status: ready-for-review
     │ <────────────────────────────────── │
     │  9. Review + refactor              │
     │ 10. Notify human to merge          │
     │                                    │
```

### Blocked Path: Design Issue

```
Authoring Agent                     Execution Agent
     │                                    │
     │     Status: ready-for-execution    │
     │ ──────────────────────────────────> │
     │                                    │  Encounters design issue
     │                                    │  Status: blocked
     │ <────────────────────────────────── │
     │  Fix design issue                  │
     │  Status: ready-for-execution       │
     │ ──────────────────────────────────> │
     │                                    │  Resume execution
     │                                    │  Status: ready-for-review
     │ <────────────────────────────────── │
     │                                    │
```

### Multi-Phase: Iterative Tuning

```
Authoring Agent                     Execution Agent
     │                                    │
     │     Status: ready-for-execution    │
     │ ──────────────────────────────────> │
     │                                    │  Initial run: results poor
     │                                    │  Tunes hyperparameters
     │                                    │  Status: ready-for-review
     │ <────────────────────────────────── │
     │  Reviews: results still not right  │
     │  Adjusts architecture              │
     │  Status: ready-for-execution       │
     │ ──────────────────────────────────> │
     │                                    │  Re-runs with new code
     │                                    │  Status: ready-for-review
     │ <────────────────────────────────── │
     │  Approves, merges to main          │
     │                                    │
```

## Task ID Convention

Task IDs should be short, descriptive, and kebab-cased:

```
<project>-<component>-<version>
```

Examples:
- `didae-procrustes-v1`
- `sce-celeba-batchfix`
- `cfkd-multinli-prompts`
- `predictor-dfr-square`

## Git Hooks (Optional)

Teams can optionally add a pre-push hook to validate the handoff file:

```bash
#!/bin/bash
# .git/hooks/pre-push
if git diff --cached --name-only | grep -q ".agent/handoff.md"; then
    # Check required fields exist
    for field in "Status" "From" "To" "Task-ID" "Branch"; do
        if ! grep -q "## $field:" .agent/handoff.md; then
            echo "ERROR: .agent/handoff.md missing required field: $field"
            exit 1
        fi
    done
fi
```

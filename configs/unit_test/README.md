# Smoke-test configs

Tiny versions of the Square pipeline, sized to finish in minutes rather than to produce a
result worth reporting. `reproduction_scripts/unit_test.sh` runs them to answer one
question: did something break on a superficial level.

They used to live at `configs/generators/`, `configs/predictors/` and `configs/adaptors/`,
three directories that do not exist, so the script failed on its first line. Do not cite
any number these produce; the budgets are far too small.

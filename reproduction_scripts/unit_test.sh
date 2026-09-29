# The purpose of this script is to find out quickly whether something broke on a superficial
# level. It is NOT a reproduction: the configs under configs/unit_test/ use tiny budgets (200
# generator steps, 2 epochs, 24 training counterfactuals), so no number it prints is citable.
#
# It used to point at configs/generators/, configs/predictors/ and configs/adaptors/, three
# directories that do not exist, so it failed on its first line.
#
# Each stage feeds the next, so run them in order. The whole script is minutes, not hours.
set -e

python preflight.py --script "$0" --quiet || \
    echo "^ these lines will fail; the rest of the script still runs"

# 1. generator training starts, checkpoints, and writes <base_path>/config.yaml
python train_generator.py --config "<PEAL_BASE>/configs/unit_test/square_ddpm_unit_test.yaml"

# 2. predictor training runs and writes model.cpl
python train_predictor.py --config "<PEAL_BASE>/configs/unit_test/square_classifier_unit_test.yaml"

# 3. the explainer runs, the teacher is consulted, and the student is finetuned end to end
python run_cfkd.py --config "<PEAL_BASE>/configs/unit_test/square_sce_cfkd_unit_test.yaml"

echo "unit test finished; outputs under \$PEAL_RUNS/unit_test/square"

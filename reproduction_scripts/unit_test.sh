# the purpose of this script is to find out quickly whether something broke on a superficial level
python train_generator.py --config "<PEAL_BASE>/configs/generators/square_ddpm_unit_test.yaml"

python train_predictor.py --config "<PEAL_BASE>/configs/predictors/square_classifier_unit_test.yaml"

python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/square_pdc_cfkd_unit_test.yaml"

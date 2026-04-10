# This script is meant to be able to reproduce the results of the PDC paper (ARXIV_LINK).
# The results were reproduced with the following software versions: (GIT_HASH)
# the batch sizes are optimized for a GPU with 80gb VRAM, but can be decreased for smaller GPUs
# this script was executed at commit TODO


# Reproduce SOTA results on the square dataset
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square_classifier_unpoisoned.yaml"
python train_generator.py --config "<PEAL_BASE>/configs/cfkd_experiments/generators/square1k_ddpm_poisoned098.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square1k_classifier_poisoned098.yaml"
# run DFR
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square1k_classifier_poisoned098_dfr.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/square1k/colora_confounding_colorb/torchvision/classifier_poisoned098/dfr/model.cpl --data_config configs/cfkd_experiments/data/square_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/square1k_classifier_poisoned098.yaml
# run GroupDRO
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/group_dro/square_1k_poisoned098_group_dro.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/square1k/colora_confounding_colorb/torchvision/classifier_poisoned098/group_dro/model.cpl --data_config configs/cfkd_experiments/data/square_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/square1k_classifier_poisoned098.yaml
# run P-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/square1000_poisoned098_pclarc.yaml"
cat ${PEAL_RUNS}/square1k/colora_confounding_colorb/torchvision/classifier_poisoned098/pclarc/best_model_result.txt
# run RR-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/square1000_poisoned098_rrclarc.yaml"
cat ${PEAL_RUNS}/square1k/colora_confounding_colorb/torchvision/classifier_poisoned098/rrclarc/best_model_result.txt
# run DiME-CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/diffae_experiments/adaptors/square1k_resnet18_poisoned098_dime_cfkd.yaml"
# run ACE-CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/diffae_experiments/adaptors/square1k_resnet18_poisoned098_ace_cfkd.yaml"
# run FastDiME-CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/diffae_experiments/adaptors/square1k_resnet18_poisoned098_fastdime_cfkd.yaml"
# run SCE-CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/diffae_experiments/adaptors/square1k_resnet18_poisoned098_sce_cfkd.yaml"
# train square foundation model
python train_predictor.py --config "<PEAL_BASE>/configs/diffae_experiments/predictors/square_all_attributes_resnet18.yaml"
# train square diffusion autoencoder
python train_generator.py --config "<PEAL_BASE>/configs/diffae_experiments/generators/square_diffusion_autoencoder.yaml"
# train linear probe from foundation model
python train_predictor.py --config "<PEAL_BASE>/configs/diffae_experiments/predictors/square1k_foundation_linear_poisoned098.yaml"
# train square component analysis
python run_component_analysis.py --config $PEAL_RUNS/square/diffusion_autoencoder/config.yaml --sd_config configs/diffae_experiments/sparse_dictionaries/procrustes_sae_square.yaml
# run DFR
python train_predictor.py --config "<PEAL_BASE>/configs/diffae_experiments/predictors/square1k_foundation_linear_poisoned098_dfr.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/square1k/colora_confounding_colorb/torchvision/foundation_poisoned098/dfr/model.cpl --data_config configs/cfkd_experiments/data/square_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/square1k_classifier_poisoned098.yaml
# run GroupDRO
python run_adaptor.py --config "<PEAL_BASE>/configs/diffae_experiments/adaptors/square_1k_foundation_linear_poisoned098_group_dro.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/square1k/colora_confounding_colorb/torchvision/foundation_linear_poisoned098/group_dro/model.cpl --data_config configs/cfkd_experiments/data/square_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/square1k_classifier_poisoned098.yaml
# run SDAE projection
python run_adaptor.py --config "<PEAL_BASE>/configs/diffae_experiments/adaptors/square1kx098_sdae_projection.yaml"
# run DiME-CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/diffae_experiments/adaptors/square1000x098_dime_cfkd.yaml"
# run ACE-CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/diffae_experiments/adaptors/square1000x098_ace_cfkd.yaml"
# run FastDiME-CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/diffae_experiments/adaptors/square1000x098_fastdime_cfkd.yaml"
# run SDAE CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/diffae_experiments/adaptors/square1k_resnet18_poisoned098_sdae_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/diffae_experiments/adaptors/square1kx098_sdae_cfkd.yaml"


# Reproduce SOTA results on CelebA Blond_Hair confounding Male task
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/celeba_Blond_Hair_classifier_unpoisoned.yaml"
python train_generator.py --config "<PEAL_BASE>/configs/cfkd_experiments/generators/celeba1k_ddpm_poisoned098.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/celeba1k_Blond_Hair_classifier_poisoned098.yaml"
# run DFR
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/celeba1k_Blond_Hair_classifier_poisoned098_dfr.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba1k/Blond_Hair/classifier_poisoned098/dfr/model.cpl --data_config configs/cfkd_experiments/data/celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba1k_Blond_Hair_classifier_poisoned098.yaml
# run GroupDRO
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/group_dro/blond_confounding_male_1k_poisoned098_group_dro.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba1k/Blond_Hair/classifier_poisoned098/group_dro/model.cpl --data_config configs/cfkd_experiments/data/celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba1k_Blond_Hair_classifier_poisoned098.yaml
# run P-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/blond_confounding_male_poisoned098_pclarc.yaml"
cat ${PEAL_RUNS}/celeba1k/Blond_Hair/classifier_poisoned098/pclarc/best_model_result.txt
# run RR-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/blond_confounding_male_poisoned098_rrclarc.yaml"
cat ${PEAL_RUNS}/celeba1k/Blond_Hair/classifier_poisoned098/rrclarc/best_model_result.txt
# run DiME CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/diffae_experiments/adaptors/celeba1kx098_resnet18_dime_cfkd.yaml"
# run ACE CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/diffae_experiments/adaptors/celeba1kx098_resnet18_ace_cfkd.yaml"
# run FastDiME CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/diffae_experiments/adaptors/celeba1kx098_resnet18_fastdime_cfkd.yaml"
# run SCE CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/diffae_experiments/adaptors/celeba1kx098_resnet18_sce_cfkd.yaml"
# train celeba diffusion autoencoder
python train_generator.py --config "<PEAL_BASE>/configs/diffae_experiments/generators/celeba_diffusion_autoencoder.yaml"
# train linear probe from foundation model
python train_predictor.py --config "<PEAL_BASE>/configs/diffae_experiments/predictors/celeba1k_foundation_linear_poisoned098.yaml"
# train celeba component analysis
python run_component_analysis.py --config $PEAL_RUNS/celeba/diffusion_autoencoder/config.yaml --sd_config configs/diffae_experiments/sparse_dictionaries/procrustes_sae_celeba.yaml
# run DFR
python train_predictor.py --config "<PEAL_BASE>/configs/diffae_experiments/predictors/celeba1k_foundation_linear_poisoned098_dfr.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba1k/Blond_Hair/openclip_linear_poisoned098/dfr/model.cpl --data_config configs/cfkd_experiments/data/celeba.yaml --model_config configs/diffae_experiments/predictors/celeba1k_Blond_Hair_openclip_linear_poisoned098.yaml
# run GroupDRO
python run_adaptor.py --config "<PEAL_BASE>/configs/diffae_experiments/adaptors/celeba_1k_foundation_linear_poisoned098_group_dro.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba1k/Blond_Hair/openclip_linear_poisoned098/group_dro/model.cpl --data_config configs/cfkd_experiments/data/celeba_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/celeba1k_classifier_poisoned098.yaml
# run SDAE projection
python run_adaptor.py --config "<PEAL_BASE>/configs/diffae_experiments/adaptors/celeba1kx098_sdae_projection.yaml"
# run DiME-CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/diffae_experiments/adaptors/celeba1kx098_dime_cfkd.yaml"
# run ACE-CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/diffae_experiments/adaptors/celeba1kx098_ace_cfkd.yaml"
# run FastDiME-CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/diffae_experiments/adaptors/celeba1kx098_fastdime_cfkd.yaml"
# run SDAE CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/diffae_experiments/adaptors/celeba1k_resnet18_poisoned098_sdae_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/diffae_experiments/adaptors/celeba1kx098_sdae_cfkd.yaml"
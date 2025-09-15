# This script is meant to be able to reproduce the results of the PDC paper (ARXIV_LINK).
# The results were reproduced with the following software versions: (GIT_HASH)
# the batch sizes are optimized for a GPU with 80gb VRAM, but can be decreased for smaller GPUs
# this script was executed at commit TODO


# Reproduce SOTA results on the square dataset
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square_classifier_unpoisoned.yaml"
python train_generator.py --config "<PEAL_BASE>/configs/cfkd_experiments/generators/square1k_ddpm_poisoned098.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square1k_classifier_poisoned098.yaml"
# run CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/square1000x098_sce_cfkd.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/square1k/colora_confounding_colorb/torchvision/classifier_poisoned098/sce_cfkd/model.cpl --data_config configs/cfkd_experiments/data/square_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/square1k_classifier_poisoned098.yaml
# run DiffAug
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square1k_classifier_poisoned098_diffusion_augmented.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/square1k/colora_confounding_colorb/torchvision/classifier_poisoned098/diffusion_augmented/model.cpl --data_config configs/cfkd_experiments/data/square_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/square1k_classifier_poisoned098.yaml
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


# Reproduce SOTA results on CelebA copyrighttag dataset (the results over different poisoning levels will be averaged)
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/celeba_copyrighttag_unpoisoned.yaml"
python train_generator.py --config "<PEAL_BASE>/configs/cfkd_experiments/generators/celeba1k_copyrighttag_ddpm_poisoned098.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/celeba1k_Smiling_confounding_copyrighttag_classifier_poisoned098.yaml"
# run CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/Smiling_confounding_CopyrightTag_celeba1000x098_sce_cfkd.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba1k_copyrighttag/Smiling_confounding_copyrighttag/regularized0/classifier_poisoned098/sce_cfkd/model.cpl --data_config configs/cfkd_experiments/data/Smiling_confounding_copyrighttag_celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba1k_copyrighttag_poisoned098.yaml
# run DiffAug
python train_predictor.py --config configs/cfkd_experiments/predictors/celeba1k_Smiling_confounding_copyrighttag_classifier_poisoned098_diffusion_augmented.yaml
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba1k_copyrighttag/Smiling_confounding_copyrighttag/regularized0/classifier_poisoned098/diffusion_augmented/model.cpl --data_config configs/cfkd_experiments/data/Smiling_confounding_copyrighttag_celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba1k_copyrighttag_poisoned098.yaml
# run DFR
python train_predictor.py --config configs/cfkd_experiments/predictors/celeba1k_Smiling_confounding_copyrighttag_classifier_poisoned098_dfr.yaml
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba1k_copyrighttag/Smiling_confounding_copyrighttag/regularized0/classifier_poisoned098/dfr/model.cpl --data_config configs/cfkd_experiments/data/Smiling_confounding_copyrighttag_celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba1k_copyrighttag_poisoned098.yaml
# run GroupDRO
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/group_dro/smiling_confounding_copyrighttag_poisoned098_group_dro.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba1k_copyrighttag/Smiling_confounding_copyrighttag/regularized0/classifier_poisoned098/group_dro/model.cpl --data_config configs/cfkd_experiments/data/Smiling_confounding_copyrighttag_celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba1k_Blond_Hair_confounding_Male_poisoned098.yaml
# run P-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/smiling_confounding_copyrighttag_098_pclarc.yaml"
cat ${PEAL_RUNS}/celeba1k_copyrighttag/Smiling_confounding_copyrighttag/regularized0/classifier_poisoned098/pclarc/best_model_result.txt
# run RR-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/smiling_confounding_copyrighttag_098_rrclarc.yaml"
cat ${PEAL_RUNS}/celeba1k_copyrighttag/Smiling_confounding_copyrighttag/regularized0/classifier_poisoned098/rrclarc/best_model_result.txt


# Reproduce SOTA results on CelebA Blond_Hair confounding Male task
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/celeba_Blond_Hair_classifier_unpoisoned.yaml"
python train_generator.py --config "<PEAL_BASE>/configs/cfkd_experiments/generators/celeba1k_ddpm_poisoned098.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/celeba1k_Blond_Hair_classifier_poisoned098.yaml"
# run CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/celeba_Blond_Hair_confounding_Male_poisoned098_sce_cfkd.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba1k/Blond_Hair/resnet18_poisoned098/sce_cfkd/model.cpl --data_config configs/cfkd_experiments/data/celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba1k_Blond_Hair_classifier_poisoned098.yaml
# run DiffAug
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/celeba1k_Blond_Hair_classifier_poisoned098_diffusion_augmented.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba1k/Blond_Hair/resnet18_poisoned098/diffusion_augmented/model.cpl --data_config configs/cfkd_experiments/data/celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba1k_Blond_Hair_classifier_poisoned098.yaml
# run DFR
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/celeba1k_Blond_Hair_classifier_poisoned098_dfr.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba1k/Blond_Hair/resnet18_poisoned098/dfr/model.cpl --data_config configs/cfkd_experiments/data/celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba1k_Blond_Hair_classifier_poisoned098.yaml
# run GroupDRO
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/group_dro/blond_confounding_male_1k_poisoned098_group_dro.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba1k/Blond_Hair/resnet18_poisoned098/group_dro/model.cpl --data_config configs/cfkd_experiments/data/celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba1k_Blond_Hair_classifier_poisoned098.yaml
# run P-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/blond_confounding_male_poisoned098_pclarc.yaml"
cat ${PEAL_RUNS}/celeba1k/Blond_Hair/resnet18_poisoned098/pclarc/best_model_result.txt
# run RR-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/blond_confounding_male_poisoned098_rrclarc.yaml"
cat ${PEAL_RUNS}/celeba1k/Blond_Hair/resnet18_poisoned098/rrclarc/best_model_result.txt


# Reproduce SOTA results on Camelyon17 task
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/camelyon17_classifier_unpoisoned.yaml"
python train_generator.py --config "<PEAL_BASE>/configs/cfkd_experiments/generators/camelyon17_1k_ddpm_poisoned098.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/camelyon17_1k_classifier_poisoned098.yaml"
# run CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/camelyon17_sce_cfkd.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/camelyon17_1k/classifier_poisoned098/sce_cfkd/model.cpl --data_config configs/cfkd_experiments/data/camelyon17_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/camelyon17_1k_classifier_poisoned098.yaml
# run DiffAug
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/camelyon17_1k_classifier_poisoned098_diffusion_augmented.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/camelyon17_1k/classifier_poisoned098/diffusion_augmented/model.cpl --data_config configs/cfkd_experiments/data/camelyon17_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/camelyon17_1k_classifier_poisoned098.yaml
# run DFR
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/camelyon17_1k_classifier_poisoned098_dfr.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/camelyon17_1k/classifier_poisoned098/dfr/model.cpl --data_config configs/cfkd_experiments/data/camelyon17_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/camelyon17_1k_classifier_poisoned098.yaml
# run GroupDRO
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/group_dro/camelyon17_1k_poisoned098_group_dro.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/camelyon17_1k/classifier_poisoned098/group_dro/model.cpl --data_config configs/cfkd_experiments/data/camelyon17_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/camelyon17_1k_classifier_poisoned098.yaml
# run P-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/camelyon17_poisoned098_pclarc.yaml"
cat ${PEAL_RUNS}/camelyon17_1k/classifier_poisoned098/pclarc/best_model_result.txt
# run RR-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/camelyon17_poisoned098_rrclarc.yaml"
cat ${PEAL_RUNS}/camelyon17_1k/classifier_poisoned098/rrclarc/best_model_result.txt


# Reproduce SOTA results on follicle dataset
python train_generator.py --config "<PEAL_BASE>/configs/cfkd_experiments/generators/follicle_ddpm.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/follicle_cut_classifier.yaml"
# run CFKD (careful, needs feedback through the human in the loop!!!)
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/follicles_sce_human_cfkd.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/follicles_cut/classifier_natural/sce_cfkd/model.cpl --data_config configs/cfkd_experiments/data/follicles_hints.yaml --model_config configs/cfkd_experiments/predictors/follicle_cut_classifier.yaml
# run DiffAug
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/follicle_cut_classifier_diffusion_augmented.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/follicles_cut/classifier_natural/diffusion_augmented/model.cpl --data_config configs/cfkd_experiments/data/follicles_hints.yaml --model_config configs/cfkd_experiments/predictors/follicle_cut_classifier.yaml
# run DFR
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/follicle_cut_classifier_dfr.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/follicles_cut/classifier_natural/dfr/model.cpl --data_config configs/cfkd_experiments/data/follicles_hints.yaml --model_config configs/cfkd_experiments/predictors/follicle_cut_classifier.yaml
# run GroupDRO
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/group_dro/follicles_group_dro.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/follicles_cut/classifier_natural/group_dro/model.cpl --data_config configs/cfkd_experiments/data/follicles_hints.yaml --model_config configs/cfkd_experiments/predictors/follicle_cut_classifier.yaml
# run P-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/follicles_pclarc.yaml"
cat ${PEAL_RUNS}/follicles_cut/classifier_natural/pclarc/best_model_result.txt
# run RR-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/follicles_rrclarc.yaml"
cat ${PEAL_RUNS}/follicles_cut/classifier_natural/rrclarc/best_model_result.txt


# Experiments over different poisoning levels
# on Square dataset
# for 50% poisoning (corresponds to 0.0 correlation)
python train_generator.py --config "<PEAL_BASE>/configs/cfkd_experiments/generators/square1k_ddpm_poisoned050.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square1k_classifier_poisoned050.yaml"
# run CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/square1000x050_sce_cfkd.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/square1k/colora_confounding_colorb/torchvision/classifier_poisoned050/sce_cfkd/model.cpl --data_config configs/cfkd_experiments/data/square_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/square1k_classifier_poisoned050.yaml
# run RR-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/square1000_poisoned050_rrclarc.yaml"
# running DFR makes no sense for 0.0 correlation, so we just use the unfixed model results

# for 60% poisoning (corresponds to 0.2 correlation)
python train_generator.py --config "<PEAL_BASE>/configs/cfkd_experiments/generators/square1k_ddpm_poisoned060.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square1k_classifier_poisoned060.yaml"
# run CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/square1000x060_sce_cfkd.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/square1k/colora_confounding_colorb/torchvision/classifier_poisoned060/sce_cfkd/model.cpl --data_config configs/cfkd_experiments/data/square_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/square1k_classifier_poisoned060.yaml
# run DFR
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square1k_classifier_poisoned060_dfr.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/square1k/colora_confounding_colorb/torchvision/classifier_poisoned060/dfr/model.cpl --data_config configs/cfkd_experiments/data/square_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/square1k_classifier_poisoned060.yaml
# run RR-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/square1000_poisoned060_rrclarc.yaml"

# for 70% poisoning (corresponds to 0.4 correlation)
python train_generator.py --config "<PEAL_BASE>/configs/cfkd_experiments/generators/square1k_ddpm_poisoned070.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square1k_classifier_poisoned070.yaml"
# run CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/square1000x070_sce_cfkd.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/square1k/colora_confounding_colorb/torchvision/classifier_poisoned070/sce_cfkd/model.cpl --data_config configs/cfkd_experiments/data/square_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/square1k_classifier_poisoned070.yaml
# run DFR
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square1k_classifier_poisoned070_dfr.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/square1k/colora_confounding_colorb/torchvision/classifier_poisoned070/dfr/model.cpl --data_config configs/cfkd_experiments/data/square_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/square1k_classifier_poisoned070.yaml
# run RR-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/square1000_poisoned070_rrclarc.yaml"

# for 80% poisoning (corresponds to 0.6 correlation)
python train_generator.py --config "<PEAL_BASE>/configs/cfkd_experiments/generators/square1k_ddpm_poisoned080.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square1k_classifier_poisoned080.yaml"
# run CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/square1000x080_sce_cfkd.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/square1k/colora_confounding_colorb/torchvision/classifier_poisoned080/sce_cfkd/model.cpl --data_config configs/cfkd_experiments/data/square_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/square1k_classifier_poisoned080.yaml
# run DFR
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square1k_classifier_poisoned080_dfr.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/square1k/colora_confounding_colorb/torchvision/classifier_poisoned080/dfr/model.cpl --data_config configs/cfkd_experiments/data/square_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/square1k_classifier_poisoned080.yaml
# run RR-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/square1000_poisoned080_rrclarc.yaml"

# for 90% poisoning (corresponds to 0.8 correlation)
python train_generator.py --config "<PEAL_BASE>/configs/cfkd_experiments/generators/square1k_ddpm_poisoned090.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square1k_classifier_poisoned090.yaml"
# run CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/square1000x090_sce_cfkd.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/square1k/colora_confounding_colorb/torchvision/classifier_poisoned090/sce_cfkd/model.cpl --data_config configs/cfkd_experiments/data/square_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/square1k_classifier_poisoned090.yaml
# run DFR
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square1k_classifier_poisoned090_dfr.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/square1k/colora_confounding_colorb/torchvision/classifier_poisoned090/dfr/model.cpl --data_config configs/cfkd_experiments/data/square_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/square1k_classifier_poisoned090.yaml
# run RR-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/square1000_poisoned090_rrclarc.yaml"

# for 100% poisoning (corresponds to 1.0 correlation)
python train_generator.py --config "<PEAL_BASE>/configs/cfkd_experiments/generators/square1k_ddpm_poisoned100.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square1k_classifier_poisoned100.yaml"
# run CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/square1000x100_sce_cfkd.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/square1k/colora_confounding_colorb/torchvision/classifier_poisoned100/sce_cfkd/model.cpl --data_config configs/cfkd_experiments/data/square_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/square1k_classifier_poisoned100.yaml
# running DFR and RR-ClarC for correlation 1.0 is impossible, so we just use the unfixed model results

# on CelebA copyrighttag dataset
# for 50% poisoning (corresponds to 0.0 correlation)
python train_generator.py --config "<PEAL_BASE>/configs/cfkd_experiments/generators/celeba1k_copyrighttag_ddpm_poisoned050.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/celeba1k_Smiling_confounding_copyrighttag_classifier_poisoned050.yaml"
# run CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/Smiling_confounding_CopyrightTag_celeba1000x050_sce_cfkd.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba1k_copyrighttag/Smiling_confounding_copyrighttag/regularized0/classifier_poisoned050/sce_cfkd/model.cpl --data_config configs/cfkd_experiments/data/Smiling_confounding_copyrighttag_celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba1k_copyrighttag_poisoned050.yaml
# run RR-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/smiling_confounding_copyrighttag_050_rrclarc.yaml"
# running DFR makes no sense for 0.0 correlation, so we just use the unfixed model results

# for 60% poisoning (corresponds to 0.2 correlation)
python train_generator.py --config "<PEAL_BASE>/configs/cfkd_experiments/generators/celeba1k_copyrighttag_ddpm_poisoned060.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/celeba1k_Smiling_confounding_copyrighttag_classifier_poisoned060.yaml"
# run CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/Smiling_confounding_CopyrightTag_celeba1000x060_sce_cfkd.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba1k_copyrighttag/Smiling_confounding_copyrighttag/regularized0/classifier_poisoned060/sce_cfkd/model.cpl --data_config configs/cfkd_experiments/data/Smiling_confounding_copyrighttag_celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba1k_copyrighttag_poisoned060.yaml
# run DFR
python train_predictor.py --config configs/cfkd_experiments/predictors/celeba1k_Smiling_confounding_copyrighttag_classifier_poisoned060_dfr.yaml
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba1k_copyrighttag/Smiling_confounding_copyrighttag/regularized0/classifier_poisoned060/dfr/model.cpl --data_config configs/cfkd_experiments/data/Smiling_confounding_copyrighttag_celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba1k_copyrighttag_poisoned060.yaml
# run RR-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/smiling_confounding_copyrighttag_060_rrclarc.yaml"

# for 70% poisoning (corresponds to 0.4 correlation)
python train_generator.py --config "<PEAL_BASE>/configs/cfkd_experiments/generators/celeba1k_copyrighttag_ddpm_poisoned070.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/celeba1k_Smiling_confounding_copyrighttag_classifier_poisoned070.yaml"
# run CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/Smiling_confounding_CopyrightTag_celeba1000x070_sce_cfkd.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba1k_copyrighttag/Smiling_confounding_copyrighttag/regularized0/classifier_poisoned070/sce_cfkd/model.cpl --data_config configs/cfkd_experiments/data/Smiling_confounding_copyrighttag_celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba1k_copyrighttag_poisoned070.yaml
# run DFR
python train_predictor.py --config configs/cfkd_experiments/predictors/celeba1k_Smiling_confounding_copyrighttag_classifier_poisoned070_dfr.yaml
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba1k_copyrighttag/Smiling_confounding_copyrighttag/regularized0/classifier_poisoned070/dfr/model.cpl --data_config configs/cfkd_experiments/data/Smiling_confounding_copyrighttag_celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba1k_copyrighttag_poisoned070.yaml
# run RR-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/smiling_confounding_copyrighttag_070_rrclarc.yaml"

# for 80% poisoning (corresponds to 0.6 correlation)
python train_generator.py --config "<PEAL_BASE>/configs/cfkd_experiments/generators/celeba1k_copyrighttag_ddpm_poisoned080.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/celeba1k_Smiling_confounding_copyrighttag_classifier_poisoned080.yaml"
# run CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/Smiling_confounding_CopyrightTag_celeba1000x080_sce_cfkd.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba1k_copyrighttag/Smiling_confounding_copyrighttag/regularized0/classifier_poisoned080/sce_cfkd/model.cpl --data_config configs/cfkd_experiments/data/Smiling_confounding_copyrighttag_celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba1k_copyrighttag_poisoned080.yaml
# run DFR
python train_predictor.py --config configs/cfkd_experiments/predictors/celeba1k_Smiling_confounding_copyrighttag_classifier_poisoned080_dfr.yaml
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba1k_copyrighttag/Smiling_confounding_copyrighttag/regularized0/classifier_poisoned080/dfr/model.cpl --data_config configs/cfkd_experiments/data/Smiling_confounding_copyrighttag_celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba1k_copyrighttag_poisoned080.yaml
# run RR-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/smiling_confounding_copyrighttag_080_rrclarc.yaml"

# for 90% poisoning (corresponds to 0.8 correlation)
python train_generator.py --config "<PEAL_BASE>/configs/cfkd_experiments/generators/celeba1k_copyrighttag_ddpm_poisoned090.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/celeba1k_Smiling_confounding_copyrighttag_classifier_poisoned090.yaml"
# run CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/Smiling_confounding_CopyrightTag_celeba1000x090_sce_cfkd.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba1k_copyrighttag/Smiling_confounding_copyrighttag/regularized0/classifier_poisoned090/sce_cfkd/model.cpl --data_config configs/cfkd_experiments/data/Smiling_confounding_copyrighttag_celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba1k_copyrighttag_poisoned090.yaml
# run DFR
python train_predictor.py --config configs/cfkd_experiments/predictors/celeba1k_Smiling_confounding_copyrighttag_classifier_poisoned090_dfr.yaml
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba1k_copyrighttag/Smiling_confounding_copyrighttag/regularized0/classifier_poisoned090/dfr/model.cpl --data_config configs/cfkd_experiments/data/Smiling_confounding_copyrighttag_celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba1k_copyrighttag_poisoned090.yaml
# run RR-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/smiling_confounding_copyrighttag_090_rrclarc.yaml"

# for 100% poisoning (corresponds to 1.0 correlation)
python train_generator.py --config "<PEAL_BASE>/configs/cfkd_experiments/generators/celeba1k_copyrighttag_ddpm_poisoned100.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/celeba1k_Smiling_confounding_copyrighttag_classifier_poisoned100.yaml"
# run CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/Smiling_confounding_CopyrightTag_celeba1000x100_sce_cfkd.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba1k_copyrighttag/Smiling_confounding_copyrighttag/regularized0/classifier_poisoned100/sce_cfkd/model.cpl --data_config configs/cfkd_experiments/data/Smiling_confounding_copyrighttag_celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba1k_copyrighttag_poisoned100.yaml
# running DFR or RR-ClarC for correlation 1.0 is impossible, so we just use the unfixed model results



# Analysis of influence of sample number
# For the Square dataset
# for 1k samples
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/square1000x098_pfc_cfkd.yaml"

# for 2k samples
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/square2kx098_pfc_cfkd.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square2k_classifier_poisoned098_dfr.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/square2k/colora_confounding_colorb/torchvision/classifier_poisoned098/dfr/model.cpl --data_config configs/cfkd_experiments/data/square_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/square2k_classifier_poisoned098_dfr.yaml
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/square2k_poisoned098_rrclarc.yaml"

# for 4k samples
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/square4kx098_pfc_cfkd.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square4k_classifier_poisoned098_dfr.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/square4k/colora_confounding_colorb/torchvision/classifier_poisoned098/dfr/model.cpl --data_config configs/cfkd_experiments/data/square_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/square4k_classifier_poisoned098_dfr.yaml
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/square4k_poisoned098_rrclarc.yaml"

# for 8k samples
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/square8kx098_pfc_cfkd.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square8k_classifier_poisoned098_dfr.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/square8k/colora_confounding_colorb/torchvision/classifier_poisoned098/dfr/model.cpl --data_config configs/cfkd_experiments/data/square_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/square8k_classifier_poisoned098_dfr.yaml
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/square8k_poisoned098_rrclarc.yaml"

# For the CelebA copyrighttag dataset
# for 1k samples
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/Smiling_confounding_CopyrightTag_celeba1000x100_pfc_cfkd.yaml"

# for 2k samples
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/Smiling_confounding_CopyrightTag_celeba2kx100_pfc_cfkd.yaml"
python train_predictor.py --config configs/cfkd_experiments/predictors/celeba2k_Smiling_confounding_copyrighttag_classifier_poisoned098_dfr.yaml
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba2k_copyrighttag/Smiling_confounding_copyrighttag/regularized0/classifier_poisoned098/dfr/model.cpl --data_config configs/cfkd_experiments/data/Smiling_confounding_copyrighttag_celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba2k_Smiling_confounding_copyrighttag_classifier_poisoned098_dfr.yaml
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/celeba2k_smiling_confounding_copyrighttag_098_rrclarc.yaml"

# for 4k samples
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/Smiling_confounding_CopyrightTag_celeba4kx100_pfc_cfkd.yaml"
python train_predictor.py --config configs/cfkd_experiments/predictors/celeba4k_Smiling_confounding_copyrighttag_classifier_poisoned098_dfr.yaml
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba4k_copyrighttag/Smiling_confounding_copyrighttag/regularized0/classifier_poisoned098/dfr/model.cpl --data_config configs/cfkd_experiments/data/Smiling_confounding_copyrighttag_celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba4k_Smiling_confounding_copyrighttag_classifier_poisoned098_dfr.yaml
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/celeba4k_smiling_confounding_copyrighttag_098_rrclarc.yaml"

# for 8k samples
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/Smiling_confounding_CopyrightTag_celeba8kx100_pfc_cfkd.yaml"
python train_predictor.py --config configs/cfkd_experiments/predictors/celeba8k_Smiling_confounding_copyrighttag_classifier_poisoned098_dfr.yaml
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba8k_copyrighttag/Smiling_confounding_copyrighttag/regularized0/classifier_poisoned098/dfr/model.cpl --data_config configs/cfkd_experiments/data/Smiling_confounding_copyrighttag_celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba8k_Smiling_confounding_copyrighttag_classifier_poisoned098_dfr.yaml
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/celeba8k_smiling_confounding_copyrighttag_098_rrclarc.yaml"



# Analysis of the influence of the student architecture
# For the Camelyon dataset
# linear probed DINOv2
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/camelyon17_1k_dinov2_finetuned_poisoned098.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/camelyon17_1k_dino_v2_finetuned_poisoned098_osce_cfkd.yaml"
python train_predictor.py --config configs/cfkd_experiments/predictors/camelyon17_1k_dinov2_finetuned_poisoned098_dfr.yaml
python evaluate_predictor.py --model_path $PEAL_RUNS/camelyon17_1k/dinov2_finetuned_poisoned098/dfr/model.cpl --data_config configs/cfkd_experiments/data/camelyon17_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/camelyon17_1k_dinov2_finetuned_poisoned098_dfr.yaml

# finetuned DINOv2
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/camelyon17_1k_dinov2_linear_poisoned098.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/camelyon17_1k_dino_v2_linear_poisoned098_osce_cfkd.yaml"
python train_predictor.py --config configs/cfkd_experiments/predictors/camelyon17_1k_dinov2_linear_poisoned098_dfr.yaml
python evaluate_predictor.py --model_path $PEAL_RUNS/camelyon17_1k/dinov2_linear_poisoned098/dfr/model.cpl --data_config configs/cfkd_experiments/data/camelyon17_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/camelyon17_1k_dinov2_linear_poisoned098_dfr.yaml



# Analysis of the influence of the teacher
# For the Square dataset for poisoning 80% (corresponds to 0.6 correlation)
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/square1000x080_sce_false_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/square1000x080_sce_mask_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/square1000x080_sce_human_cfkd.yaml"


# Analysis of the influence of the Counterfactual Explainer
# For the Square dataset
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/square1000x098_pfc_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/square1000x098_ace_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/square1000x098_dime_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/square1000x098_fastdime_cfkd.yaml"

# For Smiling confounding Copyrighttag
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/Smiling_confounding_CopyrightTag_celeba1000x100_pfc_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/Smiling_confounding_CopyrightTag_celeba1000x100_ace_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/Smiling_confounding_CopyrightTag_celeba1000x100_dime_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/Smiling_confounding_CopyrightTag_celeba1000x100_fastdime_cfkd.yaml"

# For the Camelyon17 dataset
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/camelyon17_1k_poisoned098_ace_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/camelyon17_1k_poisoned098_dime_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/camelyon17_1k_poisoned098_fastdime_cfkd.yaml"
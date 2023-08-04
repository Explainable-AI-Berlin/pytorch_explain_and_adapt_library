python3 train_model.py \
--model_config "<PEAL_BASE>/configs/models/Smiling_confounding_Blond_Hair_celeba_classifier.yaml" \
--model_name "Smiling_confounding_Blond_Hair_celeba_classifier_unpoisened"
python3 train_model.py \
--model_config "<PEAL_BASE>/configs/models/Smiling_confounding_Blond_Hair_celeba_classifier.yaml" \
--model_name "Smiling_confounding_Blond_Hair_celeba_classifier_poisened" \
--data.confounder_probability 1.0 \
--data.num_samples 7000
#python3 train_model.py \
#--model_config "<PEAL_BASE>/configs/models/celeba_vae.yaml" \
#--model_name "celeba_vae_unpoisened"
python train_diffusion_model.py \
--generator_config "peal/configs/generators/ace_generator_celeba.yaml" \
--model_name ddpm_celeba
python3 run_cfkd.py \
--adaptor_config "<PEAL_BASE>/configs/adaptors/celeba_cfkd.yaml"
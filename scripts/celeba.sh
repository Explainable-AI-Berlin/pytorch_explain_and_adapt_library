python3 train_model.py \
--model_config "<PEAL_BASE>/configs/models/Smiling_confounding_Blond_Hair_celeba_classifier.yaml" \
--model_name "Smiling_confounding_Blond_Hair_celeba_classifier_unpoisoned"
python3 train_model.py \
--model_config "<PEAL_BASE>/configs/models/Smiling_confounding_Blond_Hair_celeba_classifier.yaml" \
--model_name "Smiling_confounding_Blond_Hair_celeba_classifier_poisoned" \
--data.confounder_probability 1.0 \
--data.num_samples 7000
python3 train_model.py \
--model_config "<PEAL_BASE>/configs/models/Blond_Hair_confounding_Smiling_celeba_classifier.yaml" \
--model_name "Blond_Hair_confounding_Smiling_celeba_classifier_unpoisoned"
python3 train_model.py \
--model_config "<PEAL_BASE>/configs/models/Blond_Hair_confounding_Smiling_celeba_classifier.yaml" \
--model_name "Blond_Hair_confounding_Smiling_celeba_classifier_poisoned" \
--data.confounder_probability 1.0 \
--data.num_samples 7000
#python3 train_predictor.py \
#--model_config "<PEAL_BASE>/configs/predictors/celeba_vae.yaml" \
#--model_name "celeba_vae_unpoisoned"
#python3 train_predictor.py \
#--model_config "<PEAL_BASE>/configs/predictors/mnist_glow.yaml" \
#--model_name "celeba_glow_unpoisoned"
python train_diffusion_model.py \
--generator_config "peal/configs/generators/ace_generator_celeba.yaml"
apptainer run --nv python_container.sif python3 run_cfkd.py \
--adaptor_config "<PEAL_BASE>/configs/adaptors/Smiling_confounding_Blond_Hair_celeba_cfkd.yaml"
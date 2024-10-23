python3 generate_dataset.py \
--data_config '<PEAL_BASE>/configs/data/copyrighttag_celeba.yaml'
python3 train_model.py \
--model_config "<PEAL_BASE>/configs/models/Blond_Hair_confounding_CopyrightTag_celeba_classifier.yaml" \
--model_name "Blond_Hair_confounding_CopyrightTag_celeba_classifier_unpoisoned"
python3 train_model.py \
--model_config "<PEAL_BASE>/configs/models/Blond_Hair_confounding_CopyrightTag_celeba_classifier.yaml" \
--model_name "Blond_Hair_confounding_CopyrightTag_celeba_classifier_poisoned"
python3 train_diffusion_model.py \
--generator_config "peal/configs/generators/ace_generator_CopyrightTag_celeba.yaml"
python3 run_cfkd.py \
--adaptor_config "<PEAL_BASE>/configs/adaptors/Blond_Hair_confounding_CopyrightTag_celeba_cfkd.yaml"
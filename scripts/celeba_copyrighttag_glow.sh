python3 generate_dataset.py \
--data_config '<PEAL_BASE>/configs/data/CopyrightTag_celeba.yaml'
python3 train_model.py \
--model_config "<PEAL_BASE>/configs/models/Blond_Hair_confounding_CopyrightTag_celeba_classifier.yaml" \
--model_name "Blond_Hair_confounding_CopyrightTag_celeba_classifier_unpoisened"
python3 train_model.py \
--model_config "<PEAL_BASE>/configs/models/Blond_Hair_confounding_CopyrightTag_celeba_classifier.yaml" \
--model_name "Blond_Hair_confounding_CopyrightTag_celeba_classifier_poisened"
python3 train_diffusion_model.py \
--generator_config "peal/configs/generators/ace_generator_CopyrightTag_celeba.yaml"
python3 run_cfkd.py \
--adaptor_config "<PEAL_BASE>/configs/adaptors/Blond_Hair_confounding_CopyrightTag_celeba_cfkd.yaml"
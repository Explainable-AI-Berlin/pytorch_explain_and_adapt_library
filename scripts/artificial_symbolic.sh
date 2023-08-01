python3 train_model.py \
--model_config "<PEAL_BASE>/configs/models/symbolic_classifier.yaml" \
--model_name "symbolic_classifier_unpoisened"
python3 train_model.py \
--model_config "<PEAL_BASE>/configs/models/symbolic_classifier.yaml" \
--model_name "symbolic_classifier_poisened" \
--data.confounder_probability 1.0 \
--data.num_samples 1000
python3 train_model.py \
--model_config "<PEAL_BASE>/configs/models/symbolic_vae.yaml" \
--model_name "symbolic_vae_unpoisened"
python3 run_cfkd.py \
--adaptor_config "<PEAL_BASE>/configs/adaptors/symbolic_cfkd.yaml"
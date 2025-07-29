To reproduce our results, this folder should contain


- **Adversarial Counterfactual Explanations (ACE)**: Download "CelebaA HQ Diffusion Model" from the [ACE repo](https://github.com/guillaumejs2403/ACE?tab=readme-ov-file#downloading-pre-trained-models)
    - should be in folder `ace`
- **Diffusion Autoencoder (Diff-AE)**: Download "DiffAE (autoencoding only): FFHQ256" from the [Diff-AE repo](https://github.com/phizaz/diffae?tab=readme-ov-file#checkpoints)
    - should be in folder `diffae`
- The rest of the models should be in the top of `pretrained_models`
- Following ACE, various **STEEX** models from the [STEEX repo](https://github.com/valeoai/STEEX/releases): 
    - For the classifier, download `checkpoints_decision_densenet.tar.gz`
    - For the MNAC classifier, download `checkpoints_oracle_attribute.tar.gz `
- For the FVA classifier, download `resnet50_ft` from [VGGFace2](https://github.com/cydonia999/VGGFace2-pytorch)
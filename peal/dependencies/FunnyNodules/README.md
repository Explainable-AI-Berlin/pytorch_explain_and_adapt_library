# FunnyNodules
### [FunnyNodules: A Customizable Medical Dataset Tailored for Evaluating Explainable AI. L. Gallée, Y. Xiong, M. Beer, M. Götz (2025)](https://arxiv.org/pdf/2511.15481)
FunnyNodules is a synthetic, vision-based dataset inspired by medical image interpretation. It is designed to support **systematic**, **controlled**, and **model-agnostic evaluation of explainability methods**. The dataset generation framework enables high customizability, scalability, and fine-grained manipulation of factors of variation, making it well suited for benchmarking how different models reason about image features.

The FunnyNodules dataset, employed in the experiments of this study, comprises abstract nodules described by six visual attributes:


- roundness           1-*round*, 5-*oval*
- spiculation         1-*none*, 5-*marked*
- edge sharpness      1-*sharp*, 5-*soft*
- size                1-*small*, 5-*big*
- intensity           1-*dark*, 5-*bright*
- internal structure  0-*absent*, 1-*present*

The target class is defined based on combinations of these attributes. However, a major advantage of the synthetic FunnyNodules framework is the ability to implement different scales and rules as desired (`def calculate_target` in [dataset_generator.py](dataset/dataset_generator.py)).
<p align="center">
  <img width="339" height="289" alt="FunnyNodules_attrtar_all-1-1" src="https://github.com/user-attachments/assets/9508b14d-a2ef-4665-a99c-a95aca020b62" />
</p>

<em>Figure 1:</em> Full control of image generation allows indepth analysis of attribute-target reasoning.

### Key features of FunnyNodules 
- 🔬 **Explainability-Focused**: Developed for analyzing AI reasoning methods, especially attribute-based models.
- 🎛️ **Controllable:** Easily adjust attributes.
- 🎯 **Flexible Rules:** Implement your own rules for class label.
- 🏷️ **Fully Labeled:** Each sample comes with complete, structured annotations for all attributes, also attribute ROIs:

<p align="center">
  <img width="445" height="182" alt="FunnyNodules_attributeattention-1" src="https://github.com/user-attachments/assets/83d9c449-1848-483e-b0bf-7e8ecf7f34d4" />
</p>

<em>Figure 2:</em> Ground-truth masks are being created during image generation and enable evaluation of attention in attribute prediction.


<p align="center">
  <img width="310" height="203" alt="Attribute_oneAttributeEffectonTarget3" src="https://github.com/user-attachments/assets/cb60f558-f7e6-4993-bb94-30eedc0f0f3f" />
  <img width="356" height="120" alt="Attribute_RoundnessCSEffectonTarget3(1)" src="https://github.com/user-attachments/assets/88d1078f-e9e5-45bd-8b6f-13e68ecbc990" />
</p>

<em>Figure 3:</em> Analysis of models' reasoning, see [test_reasoning.py](dataset/test_reasoning.py). <strong>Left:</strong> Sensitivity of the model’s target predictions to varying attributes, showing whether the target rule was captured correctly.  
<strong>Right:</strong> Conditional effect of roundness on the target depending on the internal structure. Only for internal structure = 0 was this relation captured correctly, revealing a general weakness in handling correlated rules across the tested models.


## Repository Structure
### dataset/
Code for generating and customizing the FunnyNodules dataset.
Use this module to design your own dataset variants, control feature distributions, and analyze the reasoning behind your model.

### experiments/
Reference implementations for the experiments presented in the paper, including training and evaluation pipelines for ResNet-50, DenseNet-121, Proto-Caps, HierViT, and Concept Bottleneck Model.

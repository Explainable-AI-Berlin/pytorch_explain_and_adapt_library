<div align="center">
<h1>Interpreting CLIP with Hierarchical Sparse Autoencoders</h1>

[![Python](https://img.shields.io/badge/python-3.12.9-blue)](https://www.python.org/) &nbsp;&nbsp;
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-brightgreen)](https://pytorch.org/)  &nbsp;&nbsp;
[![arXiv](https://img.shields.io/badge/arXiv-2502.20578-b31b1b.svg)](https://arxiv.org/abs/2502.20578)
</div>

<div align="center">
<a href="https://scholar.google.com/citations?user=7XKIBvgAAAAJ&hl=en">Vladimir Zaigrajew</a>,
<a href="https://scholar.google.com/citations?user=H72DRC0AAAAJ&hl=en">Hubert Baniecki</a>,
<a href="https://scholar.google.com/citations?user=Af0O75cAAAAJ&hl=en">Przemysław Biecek</a>,
</div>

<div align="center">
    <img src="static/International_Conference_on_Machine_Learning.svg" style="max-width: 300px; width: 50%;" />
    <h2>:rocket: International Conference on Machine Learning 2025 :rocket:</h2>
</div>

## Description

Sparse autoencoders (SAEs) are useful for detecting and steering interpretable features in neural networks, with particular potential for understanding complex multimodal representations. Given their ability to uncover interpretable features, SAEs are particularly valuable for analyzing large-scale vision-language models (e.g., CLIP and SigLIP), which are fundamental building blocks in modern systems yet remain challenging to interpret and control. However, current SAE methods are limited by optimizing both reconstruction quality and sparsity simultaneously, as they rely on either activation suppression or rigid sparsity constraints. To this end, we introduce **Matryoshka SAE (MSAE)**, a new architecture that learns hierarchical representations at multiple granularities simultaneously, enabling a direct optimization of both metrics without compromise. MSAE establishes a new state-of-the-art Pareto frontier between reconstruction quality and sparsity for CLIP, achieving **0.99 cosine similarity and less than 0.1 fraction of variance unexplained** while maintaining **~80% sparsity**. Finally, we demonstrate the utility of MSAE as a tool for interpreting and controlling CLIP by extracting over 120 semantic concepts from its representation to perform concept-based similarity search and bias analysis in downstream tasks like CelebA.

## Table of Contents

- [Description](#description)
- [Table of Contents](#table-of-contents)
- [ICML Paper Version](#icml-paper-version)
- [Installation](#installation)
- [Prepare Datasets](#prepare-datasets)
- [Model Training](#model-training)
- [Notes on Model Training](#notes-on-model-training)
- [Model Evaluations](#model-evaluations)
- [Citation](#citation)

## ICML Paper Version

This repository is a refactored version of the code used for the ICML paper. We have removed most of the obsolete code (as there weren't many public repositories to build upon at the time of initial development, the paper repo was not the simplest to work with), achieving similar results with **almost 10x faster training**. Additionally, we've fixed a bug related to incorrect standardization for text evaluation discoverd after the paper was accepted.

We plan to update the arXiv version of the paper with results from this repository. However, if you wish to reproduce the results from the original ICML version, please refer to the `ICML_version` branch.

## Installation

We used Python 3.12.9, but due to low package requirements, it can be run on lower Python versions. To install the required packages, run:

```bash
pip install -r requirements.txt
```

**Note**: To enable dataset preparation with CLIP embeddings, you'll need to uncomment the lines after `DEV for precompute_activations` in `requirements.txt`.

## Prepare Datasets

We worked on Hugging Face versions of [ImageNet-1k](https://huggingface.co/datasets/mlx-vision/imagenet-1k) and [CC3M](https://huggingface.co/datasets/pixparse/cc3m-wds). To precompute the activations with a specific CLIP model, run:

```bash
# CC3M train split with ViT-B~16
python precompute_activations.py -d cc3m -m ViT-B~16 -s

# CC3M validation split with ViT-L~14
python precompute_activations.py -d cc3m -m ViT-L~14

# ImageNet-1K train split with ViT-B~32
python precompute_activations.py -d imagenet -m ViT-B~32 -s

# ImageNet-1K validation split with RN50
python precompute_activations.py -d imagenet -m RN50
```

Additionally, to precompute the vocabularies, we utilize vocabs from [SpLiCE](https://github.com/AI4LIFE-GROUP/SpLiCE) and [Discover-then-Name](https://github.com/neuroexplicit-saar/Discover-then-Name). These are stored in the `vocab` directory. Run the following commands:

```bash
# laion unigram
python precompute_activations.py -d laion_unigram -m ViT-L~14

# laion bigram
python precompute_activations.py -d laion_bigrams -m ViT-L~14

# laion unigram+bigram
python precompute_activations.py -d laion -m ViT-L~14

#  CLIP-Dissect
python precompute_activations.py -d disect -m ViT-L~14
```

## Model Training

Having installed the required packages and datasets, you can now train the models. Training details, including model, dataset preprocessing, and training parameters, are specified using Python configurations in `config.py`.

You can now train the model by specifying the training and validation datasets, model configuration (including activation), number of epochs, and expansion factor using the following commands:

```bash
# Trained on CC3M with ViT-B~16, 20 epochs, expansion factor 32 (512*32), ReLU SAE with sparsity regularization 0.01
python -dt cc3m_ViT-B~16_train_image_2905936_512.npy \
       -ds imagenet_ViT-B~16_train_image_1281166_512.npy \
       -dm cc3m_ViT-B~16_validation_text_13443_512.npy \
       --expansion_factor 32 --epochs 20 -m ReLUSAE -a ReLU_01

# Trained on CC3M with ViT-L~14, 30 epochs, expansion factor 16 (768*16), TopK SAE with k=64
python -dt cc3m_ViT-L~14_train_image_2905936_768.npy \
       -ds imagenet_ViT-L~14_train_image_1281166_768.npy \
       -dm cc3m_ViT-L~14_validation_text_13443_768.npy \
       --expansion_factor 16 --epochs 30 -m TopKSAE -a TopKReLU_64

# Trained on CC3M with ViT-L~14, 30 epochs, expansion factor 8 (768*8), BatchTopK SAE with k=32
python -dt cc3m_ViT-L~14_train_image_2905936_768.npy \
       -ds imagenet_ViT-L~14_train_image_1281166_768.npy \
       -dm cc3m_ViT-L~14_validation_text_13443_768.npy \
       --expansion_factor 8 --epochs 30 -m BatchTopKSAE -a BatchTopKReLU_32

# Trained on CC3M with ViT-L~14, 30 epochs, expansion factor 8 (768*8), Matryoshka SAE (RW)
python -dt cc3m_ViT-L~14_train_image_2905936_768.npy \
       -ds imagenet_ViT-L~14_train_image_1281166_768.npy \
       -dm cc3m_ViT-L~14_validation_text_13443_768.npy \
       --expansion_factor 8 --epochs 30 MSAE_RW -a ""

# Trained on CC3M with ViT-L~14, 30 epochs, expansion factor 8 (768*8), Matryoshka SAE (UW)
python -dt cc3m_ViT-L~14_train_image_2905936_768.npy \
       -ds imagenet_ViT-L~14_train_image_1281166_768.npy \
       -dm cc3m_ViT-L~14_validation_text_13443_768.npy \
       --expansion_factor 8 --epochs 30 MSAE_UW -a ""
```

**NOTE**: Models trained for our paper can be downloaded from this huggingface repository: [Matryoshka SAE Models](https://huggingface.co/WolodjaZ/MSAE).

**🚨IMPORTANT🚨**: The models hosted on Hugging Face have been retrained without the mentioned bug. Consequently, the results may differ from the paper.

## Notes on Model Training

After our ICML paper publication, we discovered several training updates that improve training speed and model performance:
- *bias_init_median*: We found that initializing the bias with the median of the activations does not significantly improve model performance.
- *mean_center*: We found that normalizing the activations by subtracting the mean does not significantly improve model performance, even across modalities.
- *reconstruction_loss*: Training SAE with MSE (*mse*) loss produces better results compared to the original NMSE (*nmse*) loss function.

## Model Evaluations

To evaluate the model similarly to the paper, we provide 3 evaluation scripts:

- `extract_sae_embeddings.py`: Evaluates the model on the provided dataset's precomputed activations and saves the SAE activations to a file.
- `score_topk_sae_embeddings.py`: Evaluates the model on the provided dataset's precomputed activations. The evaluation is done by constraining SAE activations with TopK and evaluation is done on increasing **k** active SAE neurons. The evaluation metrics are computed for each **k** and saved to a file.
- `sae_naming.py`: Calculates the similarity matrix of the SAE activations to the provided precomputed vocabularies and saves the results to a file, similarly to how it was done in [Discover-then-Name: Task-Agnostic Concept Bottlenecks via Automated Concept Discovery](https://arxiv.org/abs/2407.14499).
- `alignment_metric.py`: Calculates the alignment metric between the decoder/encoder of two differently seeded SAE models based on [Sparse Autoencoders Trained on the Same Data Learn Different Features](https://arxiv.org/abs/2501.16615).
- `linear_eval.py`: Trains a linear classifier using CLIP activations. It then evaluates the model's performance by comparing reconstructed SAE representations to the original CLIP representations, using KL divergence and accuracy metrics. Training was performed on ImageNet-1K.

Additionally, in `demo.ipynb` we provide a demo of the model evaluation and visualization of the results.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{zaigrajew2025interpreting,
  title={Interpreting CLIP with Hierarchical Sparse Autoencoders},
  author={Zaigrajew, Vladimir and Baniecki, Hubert and Biecek, Przemyslaw},
  journal={arXiv preprint arXiv:2502.20578},
  year={2025}
}
```

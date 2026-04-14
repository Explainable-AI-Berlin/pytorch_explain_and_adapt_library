# Matryoshka Sparse Autoencoders

Matryoshka SAEs are a new variant of sparse autoencoders that learn features at multiple levels of abstraction by splitting the dictionary into groups of latents of increasing size. Earlier groups are regularized to reconstruct well without access to later groups, forcing the SAE to learn both high-level concepts and low-level concepts, rather than absorbing them in specific low-level features. Due to this regularization, Matryoshka SAEs reconstruct less well than standard BatchTopK SAEs trained on Gemma-2-2B, but their downstream language model loss is similar. They show dramatically lower rates of feature absorption, feature splits and shared information between latents. They perform better on targeted concept erasure tasks, but show mixed results on k-sparse probing and automated interpretability metrics.


## Usage

```bash
git clone https://github.com/bartbussmann/matryoshka_sae.git
cd matryoshka_sae
pip install transformer_lens
python main.py
```

## Acknowledgments
The training code is heavily inspired and basically a stripped-down version of [SAELens](https://github.com/jbloomAus/SAELens). Thanks to the SAELens team for their foundational work in this area!

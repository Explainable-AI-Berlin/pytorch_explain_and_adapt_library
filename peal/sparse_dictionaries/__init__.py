"""Sparse dictionaries: concept bases over an encoder's activation space.

A ``SparseDictionary`` (``interfaces``) is fitted on the activations of a
feature extractor (the diffusion autoencoder's semantic encoder, CLIP, DINO,
...) and exposes its atoms through ``get_components`` so that DiDAE can
sweep, rank and edit along them. Implementations: probe SAEs, BatchTopK,
the MSAE (Matryoshka) and RA-SAE decompositions of published checkpoints,
SpLiCE, plain and SAE-filtered SVD and the supervised orthogonal Procrustes
dictionary. ``sparse_dictionary_factory.get_sparse_dictionary`` builds one
from its config, ``activation_cache`` memoises extracted activations,
``sae_evaluation`` matches atoms to ground-truth attributes and ``utils``
holds shared plotting helpers.
"""

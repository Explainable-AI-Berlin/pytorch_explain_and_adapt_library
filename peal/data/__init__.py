"""Datasets, dataloaders and the ``DataConfig`` schema they are built from.

``interfaces.PealDataset`` / ``DataConfig`` define what every dataset offers
the rest of PEAL (normalization round trips, contrastive collages, task
configs); ``datasets`` and ``custom_datasets`` implement the image datasets
(CelebA, Waterbirds, Square, ImageNet pairs, MNIST variants, ...),
``tabular_datasets`` the tabular ones and ``dataset_generators`` synthesise
confounded datasets. ``dataset_factory.get_datasets`` returns the
train / validation / test splits for a config and ``dataloaders`` adds the
``DataloaderMixer`` / ``WeightedDataloaderList`` used to mix counterfactual
data into predictor training.
"""

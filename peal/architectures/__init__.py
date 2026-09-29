"""Predictor architectures and the building blocks they are assembled from.

``interfaces`` declares ``TaskConfig`` / ``ArchitectureConfig`` and the
per-family configs (FC, VGG, ResNet, Transformer); ``predictors`` builds and
loads the models PEAL trains and explains (``SequentialModel``,
``TorchvisionModel``); ``module_blocks`` holds the VGG / ResNet /
Transformer blocks they are made of; ``basic_modules`` contains small
tensor-shaping layers and
``onnx_predictor`` wraps exported ONNX classifiers so they can be explained
like native ``nn.Module`` predictors.
"""

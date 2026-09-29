"""Predictor architectures and the loaders that turn a config into a model.

This module holds the classifiers PEAL explains and repairs: ``SequentialModel``
builds a plain stack of FC/VGG/ResNet/Transformer blocks from an
``ArchitectureConfig``, while ``TorchvisionModel`` wraps pretrained backbones
(torchvision ResNets and ViTs, DINOv2/DINOv3, OpenCLIP, PLIP, UNI) behind a
common ``forward``/``feature_extractor``/``get_last_layer`` interface, so that
adaptors such as CLARC or CFKD can reach the latent space and the final linear
layer. ``get_predictor`` resolves whatever a config refers to - a live module, a
pickled ``.cpl``, an ``.onnx`` graph or a predictor config - into such a model.
"""

import torch
import os

import torchvision
from pydantic import PositiveInt

from peal.architectures.basic_modules import Mean
from peal.architectures.interfaces import (
    ArchitectureConfig,
    FCConfig,
    VGGConfig,
    ResnetConfig,
    TransformerConfig,
)
from peal.architectures.module_blocks import (
    FCBlock,
    ResnetBlock,
    TransformerBlock,
    VGGBlock,
    create_cnn_layer,
)
from peal.global_utils import load_yaml_config


def get_predictor(predictor, device="cuda"):
    """Resolve a predictor reference into a model on ``device``.

    Accepts the different ways a predictor is referred to throughout PEAL:

    * an ``nn.Module`` or any callable, which is returned unchanged;
    * a path ending in ``.cpl``, loaded with ``torch.load`` (retried with
      ``weights_only=False`` for full-pickle checkpoints);
    * a path ending in ``.onnx``, converted to a trainable torch module by
      ``peal.architectures.onnx_predictor.load_onnx_predictor``;
    * anything else, which is read as a predictor config. The config either
      names ``weights_path`` (a ``TorchvisionModel`` of
      ``config.architecture`` with ``config.task.output_channels`` outputs is
      built and the state dict loaded into it), selects the pretrained
      ``torchvision_resnet18_imagenet``, or points at a run directory whose
      ``model.cpl`` is loaded.

    Parameters
    ----------
    predictor : nn.Module or callable or str or PredictorConfig
        The reference to resolve.
    device : str, optional
        Device the model is moved to.

    Returns
    -------
    model : nn.Module or callable
        The resolved predictor.
    config : PredictorConfig or None
        The config it was built from, or ``None`` when the predictor was
        given directly or as a checkpoint path.
    """
    if isinstance(predictor, torch.nn.Module) or callable(predictor):
        return predictor, None

    elif isinstance(predictor, str):
        if predictor[-4:] == ".cpl":
            try:
                model = torch.load(predictor, map_location="cpu")
            except Exception:
                model = torch.load(predictor, map_location="cpu", weights_only=False)

            if hasattr(model, "to"):
                model = model.to(device)
            return model, None

        elif predictor[-5:] == ".onnx":
            # Converted to a trainable torch module when possible (so CFKD /
            # DiDAE step 9 can finetune it and export the corrected ONNX);
            # falls back to the old onnxruntime closure for graphs onnx2torch
            # cannot convert. See peal/architectures/onnx_predictor.py.
            from peal.architectures.onnx_predictor import load_onnx_predictor

            return load_onnx_predictor(predictor, device=device), None

    else:
        predictor_config = load_yaml_config(predictor)
        if not predictor_config.weights_path is None:
            # TODO this is not very clean yet!!!
            predictor_out = TorchvisionModel(
                model=predictor_config.architecture,
                num_classes=predictor_config.task.output_channels,
            )
            predictor_out.load_state_dict(predictor_config.weights_path)

        elif predictor_config.architecture == "torchvision_resnet18_imagenet":
            predictor_out = torchvision.models.resnet18(pretrained=True)

        else:
            model_path = os.path.join(predictor_config.model_path, "model.cpl")
            predictor_out = torch.load(model_path, map_location=device)
        predictor_out = predictor_out.to(device=device)
        return predictor_out, predictor_config


def load_model(
    model_config: ArchitectureConfig,
    input_channels: PositiveInt,
    output_channels: PositiveInt,
    model_path,
    device,
):
    """
    This function loads a model from a given path.
    Args:
        model_config: The config of the model.
        input_channels: The number of input channels of the model.
        output_channels: The number of output channels of the model.
        model_path: The path to the model weights.
        device: The device the model is loaded on.

    Returns:
        The loaded model.
    """
    model = SequentialModel(model_config, input_channels, output_channels)
    try:
        checkpoint = torch.load(
            os.path.join(model_path, "checkpoints", "final.cpl"),
            map_location=torch.device(device),
        )
    except Exception:
        checkpoint = torch.load(
            os.path.join(model_path, "checkpoints", "final.cpl"),
            map_location=torch.device(device),
            weights_only=False,
        )
    model.load_state_dict(checkpoint)

    return model.to(device)


class SequentialModel(torch.nn.Sequential):
    """A sequential model that is defined by a list of layers."""

    def __init__(
        self,
        architecture_config: ArchitectureConfig,
        input_channels: PositiveInt,
        output_channels: PositiveInt = None,
        dropout: float = 0.0,
    ):
        """
        This function initializes the sequential model.
        Args:
            architecture_config: The config of the architecture.
            input_channels: The number of input channels of the model.
            output_channels: The number of output channels of the model.
        """
        if architecture_config.activation == "LeakyReLU":
            activation = torch.nn.LeakyReLU

        elif architecture_config.activation == "ReLU":
            activation = torch.nn.ReLU

        elif architecture_config.activation == "Softplus":
            activation = torch.nn.Softplus

        layers = []
        num_neurons_previous = input_channels
        for layer_config in architecture_config.layers:
            if isinstance(layer_config, ResnetConfig):
                layers.append(
                    create_cnn_layer(
                        ResnetBlock, layer_config, num_neurons_previous, activation
                    )
                )
                num_neurons_previous = layer_config.num_neurons
                tensor_dim = layer_config.tensor_dim

            elif isinstance(layer_config, VGGConfig):
                layers.append(
                    create_cnn_layer(
                        VGGBlock, layer_config, num_neurons_previous, activation
                    )
                )
                num_neurons_previous = layer_config.num_neurons
                tensor_dim = layer_config.tensor_dim

            elif isinstance(layer_config, FCConfig):
                layers.append(FCBlock(layer_config, num_neurons_previous, activation))
                num_neurons_previous = layer_config.num_neurons
                tensor_dim = layer_config.tensor_dim

            elif isinstance(layer_config, TransformerConfig):
                layers.append(
                    TransformerBlock(layer_config, num_neurons_previous, activation)
                )
                num_neurons_previous = layer_config.num_neurons
                tensor_dim = layer_config.tensor_dim

            elif isinstance(layer_config, str) and layer_config == "mean":
                layers.append(Mean())
                tensor_dim = 0

            else:
                raise ValueError("Unknown layer config: {}".format(layer_config))

        if not dropout == 0.0:
            layers.append(torch.nn.Dropout(dropout))

        if not output_channels is None:
            last_layer_config = FCConfig(
                num_neurons=output_channels, tensor_dim=tensor_dim
            )
            layers.append(
                FCBlock(last_layer_config, num_neurons_previous)
            )  # , activation))
            num_neurons_previous = output_channels

        self.output_channels = num_neurons_previous

        super(SequentialModel, self).__init__(*layers)


class TorchvisionModel(torch.nn.Module):
    """Pretrained backbone plus a fresh linear head, behind one interface.

    The ``model`` string selects the backbone and decides where the
    classification head lives: for the torchvision ResNets and ViTs the head
    inside the backbone is replaced, for the feature-extractor backbones
    (DINOv2/DINOv3, OpenCLIP, UNI) a separate ``self.fc`` is added on top of
    the frozen-format embedding. All variants are reachable through
    :meth:`feature_extractor`, :meth:`get_last_layer` and :meth:`forward`,
    which is what the PEAL adaptors rely on.

    Recognised ``model`` values
    ---------------------------
    ``resnet18``, ``resnet50``
        Pretrained torchvision ResNet with a bias-free ``fc``.
    ``resnet_loaded``
        A pickled model given by ``config.base_model``, whose inner ``fc`` is
        replaced.
    ``dino_v2``/``dino_v2_small``/``dino_v2_base``
        Hugging Face DINOv2 large/small/base plus its image processor.
    ``dino_v3``/``dino_v3_small``/``dino_v3_base``/``dino_v3:<repo>``
        DINOv3 from Hugging Face, falling back to the public timm weights
        when the repo is gated (``self.is_timm`` records which path was
        taken).
    ``open_clip:<model>:<pretrained>``
        OpenCLIP image tower; the normalisation stats and input size are read
        back out of the generated preprocessing transform.
    ``PLIP``, ``UNI``
        Pathology foundation models.
    ``vit_b_16``
        torchvision ViT; the positional embedding is truncated or re-inited
        when ``input_size`` differs from 224.

    Parameters
    ----------
    model : str
        Backbone selector, see above.
    num_classes : int
        Number of output logits of the classification head.
    input_size : int, optional
        Square input resolution; only used to adapt the ViT positional
        embedding.
    config : object, optional
        Predictor config; only ``config.base_model`` is read, for
        ``resnet_loaded``.

    Raises
    ------
    ValueError
        If ``model`` is not one of the supported names.
    """

    def __init__(self, model, num_classes, input_size=None, config=None):
        """Build the selected backbone and attach the classification head."""
        super(TorchvisionModel, self).__init__()
        self.config = config
        self.model_type = model

        # --- Standard ResNet ---
        if model == "resnet18":
            self.model = torchvision.models.resnet18(pretrained=True)
            self.model.fc = torch.nn.Linear(
                self.model.fc.in_features, num_classes, bias=False
            )

        elif model == "resnet50":
            self.model = torchvision.models.resnet50(pretrained=True)
            self.model.fc = torch.nn.Linear(
                self.model.fc.in_features, num_classes, bias=False
            )

        elif model == "resnet_loaded":
            try:
                self.model = torch.load(self.config.base_model, map_location="cpu")
            except Exception:
                self.model = torch.load(
                    self.config.base_model, map_location="cpu", weights_only=False
                )
            self.model.model.fc = torch.nn.Linear(
                self.model.model.fc.in_features, num_classes, bias=False
            )
        elif model == "PLIP":
            from transformers import CLIPModel

            foundation_model = CLIPModel.from_pretrained("vincentqb/PLIP")
            self.plip_mean = (0.48145466, 0.4578275, 0.40821073)
            self.plip_std = (0.26862954, 0.26130258, 0.27577711)
            self.plip_size = (224, 224)

        # --- DINOv2 ---
        elif model == "dino_v2_small":
            from transformers import AutoImageProcessor, AutoModel

            self.model = AutoModel.from_pretrained("facebook/dinov2-small")
            self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
            self.fc = torch.nn.Linear(384, num_classes, bias=False)

        elif model == "dino_v2_base":
            from transformers import AutoImageProcessor, AutoModel

            self.model = AutoModel.from_pretrained("facebook/dinov2-base")
            self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            self.fc = torch.nn.Linear(768, num_classes, bias=False)

        elif model == "dino_v2":
            from transformers import AutoImageProcessor, AutoModel

            self.model = AutoModel.from_pretrained("facebook/dinov2-large")
            self.fc = torch.nn.Linear(1024, num_classes, bias=False)
            self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")

        # --- DINOv3 ---
        elif model.startswith("dino_v3"):
            if ":" in model:
                repo_id = model.split(":", 1)[1]
                timm_name = repo_id
            elif model == "dino_v3_small":
                repo_id = "facebook/dinov3-vits16-pretrain-lvd1689m"
                timm_name = "vit_small_patch16_dinov3"
            elif model == "dino_v3_base":
                repo_id = "facebook/dinov3-vitb16-pretrain-lvd1689m"
                timm_name = "vit_base_patch16_dinov3"
            else:
                repo_id = "facebook/dinov3-vitl16-pretrain-lvd1689m"
                timm_name = "vit_large_patch16_dinov3"

            self.is_timm = False
            try:
                from transformers import AutoImageProcessor, AutoModel

                self.model = AutoModel.from_pretrained(repo_id)
                self.processor = AutoImageProcessor.from_pretrained(repo_id)
            except Exception:
                # If Hugging Face AutoModel is restricted (gated repo), fallback to timm public weights
                import timm

                try:
                    self.model = timm.create_model(timm_name, pretrained=True)
                    self.is_timm = True
                except Exception:
                    self.model = timm.create_model(f"hf-hub:{repo_id}", pretrained=True)
                    self.is_timm = True

            if getattr(self, "is_timm", False):
                embed_dim = getattr(self.model, "num_features", 1024)
            elif hasattr(self.model.config, "hidden_size"):
                embed_dim = self.model.config.hidden_size
            elif hasattr(self.model.config, "embed_dim"):
                embed_dim = self.model.config.embed_dim
            else:
                embed_dim = 1024

            self.fc = torch.nn.Linear(embed_dim, num_classes, bias=False)

        # --- OpenCLIP Integration ---
        elif model.startswith("open_clip"):
            import open_clip

            # Expected format: "open_clip:<model_name>:<pretrained>"
            # Example: "open_clip:ViT-B-32:laion2b_s34b_b79k"
            parts = model.split(":")
            if len(parts) >= 3:
                model_name = parts[1]
                pretrained = parts[2]
            else:
                # Default fallback if args not provided
                model_name = "ViT-B-32"
                pretrained = "laion2b_s34b_b79k"

            # Load model and transforms
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )

            # Determine embedding dimension and input size
            # OpenCLIP visual models usually have an 'output_dim' or we can infer it
            if hasattr(self.model.visual, "output_dim"):
                embed_dim = self.model.visual.output_dim
            else:
                # Fallback: simple inference to get shape
                with torch.no_grad():
                    # Default to 224 if unknown, though many are different
                    dummy = torch.zeros(1, 3, 224, 224)
                    embed_dim = self.model.encode_image(dummy).shape[1]

            # Try to extract input size and normalization stats from the generated transform
            # This is cleaner than hardcoding
            try:
                # Standard OpenCLIP transform is Compose -> [Resize, CenterCrop, ..., Normalize]
                # We try to extract the Normalize mean/std and Resize size
                self.clip_mean = None
                self.clip_std = None
                self.clip_size = (224, 224)

                for t in self.preprocess.transforms:
                    if isinstance(t, torchvision.transforms.Normalize):
                        self.clip_mean = t.mean
                        self.clip_std = t.std
                    if isinstance(
                        t,
                        (
                            torchvision.transforms.Resize,
                            torchvision.transforms.CenterCrop,
                            torchvision.transforms.RandomResizedCrop,
                        ),
                    ):
                        if isinstance(t.size, int):
                            self.clip_size = (t.size, t.size)
                        else:
                            self.clip_size = t.size

                # Fallback if extraction fails
                if self.clip_mean is None:
                    self.clip_mean = (0.48145466, 0.4578275, 0.40821073)
                    self.clip_std = (0.26862954, 0.26130258, 0.27577711)

            except Exception:
                self.clip_mean = (0.48145466, 0.4578275, 0.40821073)
                self.clip_std = (0.26862954, 0.26130258, 0.27577711)
                self.clip_size = (224, 224)

            self.fc = torch.nn.Linear(embed_dim, num_classes, bias=False)

        # --- UNI ---
        elif model == "UNI":
            import timm
            from timm.data import resolve_data_config
            from timm.data.transforms_factory import create_transform

            # login() # Assuming login is handled elsewhere or env vars

            self.model = timm.create_model(
                "hf-hub:MahmoodLab/uni",
                pretrained=True,
                init_values=1e-5,
                dynamic_img_size=True,
            )
            self.transform = create_transform(
                **resolve_data_config(self.model.pretrained_cfg, model=self.model)
            )
            self.fc = torch.nn.Linear(1024, num_classes, bias=False)

        # --- ViT ---
        elif model == "vit_b_16":
            self.model = torchvision.models.vit_b_16()
            kernel_size = 16

            if input_size and not input_size == 224:
                num_patches = (input_size // kernel_size) ** 2 + 1
                if num_patches < self.model.encoder.pos_embedding.shape[1]:
                    self.model.encoder.pos_embedding = torch.nn.Parameter(
                        self.model.encoder.pos_embedding[:, :num_patches]
                    )
                else:
                    self.model.encoder.pos_embedding = torch.nn.Parameter(
                        torch.zeros(
                            1, num_patches, self.model.encoder.pos_embedding.shape[2]
                        )
                    )
                    torch.nn.init.trunc_normal_(
                        self.model.encoder.pos_embedding, std=0.02
                    )

            if num_classes != 1000:
                self.model.heads.head = torch.nn.Linear(
                    self.model.heads.head.in_features, num_classes, bias=False
                )

        else:
            raise ValueError("Unknown model: {}".format(model))

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.model.patch_size
        n_h = h // p
        n_w = w // p

        x = self.model.conv_proj(x)
        x = x.reshape(n, self.model.hidden_dim, n_h * n_w)
        x = x.permute(0, 2, 1)

        return x

    def feature_extractor(self, x):
        """Embed a batch of images with the backbone, without the head.

        The path taken depends on ``self.model_type``:

        * ResNets (and models predating the ``model_type`` attribute): every
          child module but the last is re-assembled into a ``Sequential``, so
          the result still carries the trailing spatial dimensions
          ``(B, C, 1, 1)``.
        * DINOv2/DINOv3: the batch is resized to the processor's crop size
          (224 for the timm fallback), normalised with the processor's
          ``image_mean``/``image_std`` and reduced to the CLS token.
        * OpenCLIP and PLIP: resized to ``self.clip_size``, normalised with
          the CLIP stats and passed through the image tower.
        * UNI: the timm transform is applied and the model called directly.
        * otherwise (torchvision ViT): patch embedding, CLS token, encoder,
          CLS output.

        Parameters
        ----------
        x : torch.Tensor
            Images of shape ``(B, C, H, W)`` in ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            Latent codes, ``(B, D)`` for the transformer backbones and
            ``(B, D, 1, 1)`` for the ResNets.
        """
        if (
            not hasattr(self, "model_type")
            or self.model_type[: len("resnet")] == "resnet"
        ):
            submodules = list(self.children())
            while len(submodules) == 1:
                submodules = list(submodules[0].children())

            feature_extractor = torch.nn.Sequential(*submodules[:-1])
            return feature_extractor(x)

        elif self.model_type.startswith("dino_v2") or self.model_type.startswith(
            "dino_v3"
        ):
            if getattr(self, "is_timm", False):
                x_resized = torchvision.transforms.Resize([224, 224])(x)

                def pv(v):
                    """Reshape a per-channel mean/std to ``(C, 1, 1)``."""
                    v = torch.tensor(v, device=x.device, dtype=x.dtype)[:, None, None]
                    return v

                image_mean = (0.485, 0.456, 0.406)
                image_std = (0.229, 0.224, 0.225)
                x_processed = (x_resized - pv(image_mean)) / pv(image_std)
                feat = self.model.forward_features(x_processed)
                if hasattr(feat, "shape") and len(feat.shape) == 3:
                    latent_code = feat[:, 0]
                else:
                    latent_code = feat
                return latent_code

            cs = getattr(self.processor, "crop_size", None) or getattr(
                self.processor, "size", None
            )
            if isinstance(cs, int):
                height, width = cs, cs
            elif isinstance(cs, dict) or hasattr(cs, "__getitem__"):
                height = (
                    cs["height"] if "height" in cs else cs.get("shortest_edge", 224)
                )
                width = cs["width"] if "width" in cs else cs.get("shortest_edge", 224)
            elif hasattr(cs, "height") and hasattr(cs, "width"):
                height, width = cs.height, cs.width
            else:
                height, width = 224, 224

            x_resized = torchvision.transforms.Resize([height, width])(x)

            def pv(v):
                """Broadcast a per-channel mean/std to ``(C, height, width)``."""
                v = torch.tensor(v, device=x.device, dtype=x.dtype)[:, None, None]
                return torch.tile(v, [1, height, width])

            image_mean = getattr(self.processor, "image_mean", (0.485, 0.456, 0.406))
            image_std = getattr(self.processor, "image_std", (0.229, 0.224, 0.225))

            x_processed = (x_resized - pv(image_mean)) / pv(image_std)
            outputs = self.model(x_processed)
            if hasattr(outputs, "last_hidden_state"):
                latent_code = outputs.last_hidden_state[:, 0]
            elif isinstance(outputs, torch.Tensor):
                latent_code = outputs
            else:
                latent_code = outputs[0][:, 0]
            return latent_code

        # --- OpenCLIP Feature Extractor ---
        elif self.model_type.startswith("open_clip"):
            # Resize tensor to expected CLIP size (extracted in __init__)
            x_resized = torchvision.transforms.Resize(self.clip_size)(x)

            # Manual normalization for tensors (avoiding PIL transforms)
            def pv(v):
                """Reshape a per-channel mean/std to ``(C, 1, 1)``."""
                # Helper to reshape mean/std for broadcasting (C, 1, 1)
                v = torch.tensor(v, device=x.device, dtype=x.dtype)[:, None, None]
                return v

            # Normalize: (x - mean) / std
            x_processed = (x_resized - pv(self.clip_mean)) / pv(self.clip_std)

            # Encode image
            latent_code = self.model.encode_image(x_processed)
            return latent_code
        elif self.model_type == "PLIP":

            x_resized = torchvision.transforms.Resize(self.clip_size)(x)

            def pv(v):
                """Reshape a per-channel mean/std to ``(C, 1, 1)``."""
                # Helper to reshape mean/std for broadcasting (C, 1, 1)
                v = torch.tensor(v, device=x.device, dtype=x.dtype)[:, None, None]
                return v

            x_norm = (x_resized - pv(self.clip_mean)) / pv(self.clip_std)
            latent_code = self.model.get_image_features(pixel_values=x_norm)

        elif self.model_type == "UNI":
            x_processed = self.transform(x)
            latent_code = self.model(x_processed)
            return latent_code

        else:
            try:
                x = self._process_input(x)
            except:
                raise
            n = x.shape[0]

            batch_class_token = self.model.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            x = self.model.encoder(x)
            x = x[:, 0]
            return x

    def get_last_layer(self):
        """Return the final linear layer, i.e. the classification head.

        This is ``self.model.fc`` for the ResNets and the separate
        ``self.fc`` for the DINO, OpenCLIP and UNI backbones. Adaptors such as
        CLARC and the projection adaptor edit the weights of this layer.

        Returns
        -------
        torch.nn.Linear or None
            ``None`` for backbones whose head is not exposed this way (for
            example the torchvision ViT).
        """
        if (
            not hasattr(self, "model_type")
            or self.model_type[: len("resnet")] == "resnet"
        ):
            return self.model.fc

        elif (
            self.model_type.startswith("open_clip")
            or self.model_type.startswith("dino_v2")
            or self.model_type.startswith("dino_v3")
            or self.model_type == "UNI"
        ):
            return self.fc

    def forward(self, x: torch.Tensor, return_latents: bool = False):
        """Classify a batch, optionally also returning the latent code.

        Parameters
        ----------
        x : torch.Tensor
            Images of shape ``(B, C, H, W)``.
        return_latents : bool, optional
            Return ``(latent_code, logits)`` instead of just the logits. For
            the ResNets the latent code is squeezed to ``(B, D)``; for the
            torchvision ViT branch the logits are returned twice.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
            Logits of shape ``(B, num_classes)``, or the pair described above.
        """
        if (
            not hasattr(self, "model_type")
            or self.model_type[: len("resnet")] == "resnet"
        ):
            if return_latents:
                latent_code = self.feature_extractor(x)
                latent_code = latent_code.squeeze(-1).squeeze(-1)
                x_out = self.model.fc(latent_code)
                return latent_code, x_out
            else:
                return self.model(x)

        elif (
            self.model_type.startswith("open_clip")
            or self.model_type.startswith("dino_v2")
            or self.model_type.startswith("dino_v3")
            or self.model_type == "UNI"
        ):
            latent_code = self.feature_extractor(x)
            x_out = self.fc(latent_code)
            if return_latents:
                return latent_code, x_out
            else:
                return x_out

        else:
            x = self._process_input(x)
            n = x.shape[0]
            batch_class_token = self.model.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            x = self.model.encoder(x)
            x = x[:, 0]
            x = self.model.heads(x)
            if return_latents:
                return x, x
            else:
                return x
        # --- ViT ---

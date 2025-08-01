import torch
import os

import torchvision
from pydantic import PositiveInt
from transformers import AutoModel, AutoImageProcessor

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
    if isinstance(predictor, torch.nn.Module) or callable(predictor):
        return predictor, None

    elif isinstance(predictor, str):
        if predictor[-4:] == ".cpl":
            return torch.load(predictor, map_location=device), None

        elif predictor[-5:] == ".onnx":
            import onnxruntime as ort

            # Load the ONNX model
            session = ort.InferenceSession(predictor, providers=["CUDAExecutionProvider"])

            # Get input name for the model
            input_name = session.get_inputs()[0].name

            # Run inference
            def onnx_model(input_data):
                session_output = session.run(None, {input_name: input_data.cpu().numpy()})
                return torch.from_numpy(session_output[0]).to(device)

            # onnx_model = lambda input_data: torch.randint(0, 1, (input_data.shape[0])).to(device)
            return onnx_model, None

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
    checkpoint = torch.load(
        os.path.join(model_path, "checkpoints", "final.cpl"),
        map_location=torch.device(device),
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
                layers.append(create_cnn_layer(ResnetBlock, layer_config, num_neurons_previous, activation))
                num_neurons_previous = layer_config.num_neurons
                tensor_dim = layer_config.tensor_dim

            elif isinstance(layer_config, VGGConfig):
                layers.append(create_cnn_layer(VGGBlock, layer_config, num_neurons_previous, activation))
                num_neurons_previous = layer_config.num_neurons
                tensor_dim = layer_config.tensor_dim

            elif isinstance(layer_config, FCConfig):
                layers.append(FCBlock(layer_config, num_neurons_previous, activation))
                num_neurons_previous = layer_config.num_neurons
                tensor_dim = layer_config.tensor_dim

            elif isinstance(layer_config, TransformerConfig):
                layers.append(TransformerBlock(layer_config, num_neurons_previous, activation))
                num_neurons_previous = layer_config.num_neurons
                tensor_dim = layer_config.tensor_dim

            elif isinstance(layer_config, str) and layer_config == "mean":
                layers.append(Mean())
                tensor_dim = 0

            else:
                import pdb

                pdb.set_trace()
                raise ValueError("Unknown layer config: {}".format(layer_config))

        if not dropout == 0.0:
            layers.append(torch.nn.Dropout(dropout))

        if not output_channels is None:
            last_layer_config = FCConfig(num_neurons=output_channels, tensor_dim=tensor_dim)
            layers.append(FCBlock(last_layer_config, num_neurons_previous))  # , activation))
            num_neurons_previous = output_channels

        self.output_channels = num_neurons_previous

        super(SequentialModel, self).__init__(*layers)


class TorchvisionModel(torch.nn.Module):
    def __init__(self, model, num_classes, input_size=None, config=None):
        super(TorchvisionModel, self).__init__()
        self.config = config
        self.model_type = model
        if model == "resnet18":
            self.model = torchvision.models.resnet18(pretrained=True)
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)

        elif model == "resnet50":
            self.model = torchvision.models.resnet50(pretrained=True)
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)

        elif model == "dino_v2":
            self.model = AutoModel.from_pretrained("facebook/dinov2-large")
            self.fc = torch.nn.Linear(1024, num_classes)
            self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")

        elif model == "UNI":
            import timm
            from timm.data import resolve_data_config
            from timm.data.transforms_factory import create_transform
            from huggingface_hub import login

            login()  # login with your User Access Token, found at https://huggingface.co/settings/tokens

            # pretrained=True needed to load UNI weights (and download weights for the first time)
            # init_values need to be passed in to successfully load LayerScale parameters (e.g. - block.0.ls1.gamma)
            self.model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
            self.transform = create_transform(**resolve_data_config(self.model.pretrained_cfg, model=self.model))
            self.fc = torch.nn.Linear(1024, num_classes)

        elif model == "vit_b_16":
            self.model = torchvision.models.vit_b_16()
            # Modify the patch embedding layer
            kernel_size = 16
            """
                kernel_size = min(16, input_size // 8)
                self.model.conv_proj = torch.nn.Conv2d(
                    in_channels=3,
                    out_channels=self.model.conv_proj.out_channels,
                    kernel_size=kernel_size,  # changed from 16 to 8
                    stride=kernel_size,  # changed from 16 to 8
                    padding=0,
                    bias=False,
                )"""

            if input_size and not input_size == 224:
                # Modify the positional embedding
                num_patches = (input_size // kernel_size) ** 2 + 1  # 64 patches + class token
                if num_patches < self.model.encoder.pos_embedding.shape[1]:
                    self.model.encoder.pos_embedding = torch.nn.Parameter(
                        self.model.encoder.pos_embedding[:, :num_patches]
                    )
                else:
                    self.model.encoder.pos_embedding = torch.nn.Parameter(
                        torch.zeros(1, num_patches, self.model.encoder.pos_embedding.shape[2])
                    )
                    # reinitialize the positional embedding to random values.
                    torch.nn.init.trunc_normal_(self.model.encoder.pos_embedding, std=0.02)

            if num_classes != 1000:
                self.model.heads.head = torch.nn.Linear(self.model.heads.head.in_features, num_classes)

        else:
            raise ValueError("Unknown model: {}".format(model))

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.model.patch_size
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.model.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.model.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def feature_extractor(self, x):
        if not hasattr(self, "model_type") or self.model_type[: len("resnet")] == "resnet":
            submodules = list(self.children())
            while len(submodules) == 1:
                submodules = list(submodules[0].children())

            feature_extractor = torch.nn.Sequential(*submodules[:-1])
            return feature_extractor(x)

        elif self.model_type == "dino_v2":
            cs = self.processor.crop_size
            x_resized = torchvision.transforms.Resize([cs['height'],cs['width']])(x)
            def pv(v):
                v = torch.tensor(v).to(x_resized)[:, None, None]
                return torch.tile(v, [1, cs['height'],cs['width']])

            x_processed = (x_resized - pv(self.processor.image_mean)) / pv(self.processor.image_std)
            latent_code = self.model(x_processed)['last_hidden_state'][:,0]
            return latent_code

        elif self.model_type == "UNI":
            x_processed = self.transform(x)
            latent_code = self.model(x_processed)
            return latent_code

        else:
            # Reshape and permute the input tensor
            x = self._process_input(x)
            n = x.shape[0]

            # Expand the class token to the full batch
            batch_class_token = self.model.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            # torch.Size([128, 197, 768])
            x = self.model.encoder(x)

            # Classifier "token" as used by standard language architectures
            x = x[:, 0]

            return x

    def get_last_layer(self):
        if not hasattr(self, "model_type") or self.model_type[: len("resnet")] == "resnet":
            return self.model.fc

        elif self.model_type in ["dino_v2", "UNI"]:
            return self.fc


    def forward(self, x: torch.Tensor, return_latents: bool = False):
        if not hasattr(self, "model_type") or self.model_type[: len("resnet")] == "resnet":
            if return_latents:
                latent_code = self.feature_extractor(x)
                latent_code = latent_code.squeeze(-1).squeeze(-1)
                x_out = self.model.fc(latent_code)
                return latent_code, x_out

            else:
                return self.model(x)

        elif self.model_type in ["dino_v2", "UNI"]:
            # xt = self.processor(x)['pixel_values']
            latent_code = self.feature_extractor(x)
            x_out = self.fc(latent_code)
            if return_latents:
                return latent_code, x_out

            else:
                return x_out

        else:
            # Reshape and permute the input tensor
            x = self._process_input(x)
            n = x.shape[0]

            # Expand the class token to the full batch
            batch_class_token = self.model.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            # torch.Size([128, 197, 768])
            x = self.model.encoder(x)

            # Classifier "token" as used by standard language architectures
            x = x[:, 0]

            x = self.model.heads(x)

            return x

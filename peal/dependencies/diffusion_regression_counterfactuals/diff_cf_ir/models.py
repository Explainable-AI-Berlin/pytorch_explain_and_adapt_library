from typing import Union
import torch
from torch import nn
import torchvision
from torchvision.models.resnet import (
    ResNet18_Weights,
    ResNet50_Weights,
    ResNet152_Weights,
)
from models.normalizer import Normalizer
from models.steex.DecisionDensenetModel import DecisionDensenetModel
import lightning as L


class ResNetRegression(L.LightningModule):
    def __init__(
        self,
        type: str,
        out_features=1,
        learning_rate=1e-5,
        small_images=False,
        **kwargs,
    ):
        super().__init__()
        if type.lower() == "resnet18":
            model = torchvision.models.resnet18()
        elif type.lower() == "resnet50":
            model = torchvision.models.resnet50(ResNet50_Weights.DEFAULT)
        elif type.lower() == "resnet152":
            model = torchvision.models.resnet152(weights=ResNet152_Weights.DEFAULT)
        else:
            raise ValueError(f"ResNet type {type} not supported.")

        # Change the first layer for smaller images (Resnet was trained on 224x224)
        if small_images:
            model.conv1 = nn.Conv2d(
                3, 64, kernel_size=4, stride=1, padding=2, bias=False
            )
        # Change the last layer to output continuous values
        fc_in_features = model.fc.in_features
        model.fc = nn.Linear(fc_in_features, out_features=out_features)

        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss()
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)

        # Debug Logs
        tensorboard = self.logger.experiment
        if self.current_epoch % 5 == 0:
            tensorboard.add_text(
                "train_preds",
                str(torch.hstack([y, y_hat]).detach().cpu()),
                self.current_epoch,
            )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

        mae = torch.nn.functional.l1_loss(y_hat, y)
        self.log("val_mae", mae, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        # From ResNet Paper?
        # optimizer = torch.optim.SGD(
        #     self.parameters(),
        #     lr=self.learning_rate,
        #     weight_decay=self.weight_decay,
        #     momentum=0.9,
        # )
        # Choose AdamW for faster convgernce. Not relevant for CFs.
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


def load_resnet(path):
    model = ResNetRegression.load_from_checkpoint(path)
    return model.model.eval()


class DenseNetRegression(L.LightningModule):
    def __init__(self, weights_path: Union[str, None] = None, full_finetune=False):
        """Creates a DenseNet model for regression.

        weights_path can be provided to load the pretrained classifier from the STEEX paper.

        Parameters
        ----------
        weights_path : Union[str, None], optional
            Path to the pretrained STEEX classifier, by default None
        full_finetune : bool, optional
            Whether to do a full finetune, instead of only the last linear layer, by default False
        """
        super().__init__()
        ql = 39  # Placeholder label for young/old label from CelebAHQ
        densenet = DecisionDensenetModel(3, pretrained=False, query_label=ql)

        if weights_path is not None:
            densenet.load_state_dict(
                torch.load(weights_path, map_location="cpu")["model_state_dict"]
            )
            if not full_finetune:
                # Freeze the DenseNet feature extractor weights
                for param in densenet.feat_extract.parameters():
                    param.requires_grad = False

        # Change the last layer to output continuous values
        densenet.classifier = nn.Linear(densenet.feat_extract.output_size, 1)

        self.model = nn.Sequential(densenet.feat_extract, densenet.classifier)
        self.model = Normalizer(self.model, [0.5] * 3, [0.5] * 3)

        self.learning_rate = 1e-4
        self.criterion = nn.MSELoss()
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)

        # Debug Logs
        tensorboard = self.logger.experiment
        if self.current_epoch % 5 == 0:
            tensorboard.add_text(
                "train_preds",
                str(torch.hstack([y, y_hat]).detach().cpu()),
                self.current_epoch,
            )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

        mae = torch.nn.functional.l1_loss(y_hat, y)
        self.log("val_mae", mae, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        # Choose AdamW for faster convgernce. Not relevant for CFs.
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


def load_model(path):
    try:
        model = ResNetRegression.load_from_checkpoint(path)
        return model.model.eval()
    except Exception as e:
        try:
            print("Could not load ResNet model. Trying DenseNet model.")
            model = DenseNetRegression.load_from_checkpoint(path)
            return model.model.eval()
        except Exception as e:
            print("Error while loading either ResNet or DenseNet model.")
            raise e

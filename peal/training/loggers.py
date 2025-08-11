import torch
import os
import math
import torchvision
import torch.nn as nn

from peal.data.dataloaders import DataloaderMixer
from peal.generators.interfaces import InvertibleGenerator


class Logger:
    """ """

    def __init__(
        self,
        config,
        model,
        optimizer,
        base_dir,
        criterions,
        val_dataloader,
        attacker=None,
        writer=None,
    ):
        """ """
        #
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.base_dir = base_dir
        self.criterions = criterions
        self.attacker = attacker
        self.writer = writer
        self.val_dataloader = val_dataloader

        if not self.config.task.output_channels is None:
            self.output_channels = self.config.task.output_channels

        else:
            self.output_channels = self.config.data.output_size[0]

        self.device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
        #
        if "ce" in config.task.criterions.keys():
            #
            # self.test_X, self.test_y = create_class_ordered_batch(val_dataloader.dataset, config)
            self.test_X, self.test_y = next(iter(val_dataloader))

        else:
            #
            self.test_X, self.test_y = next(iter(val_dataloader))

        if isinstance(self.model, InvertibleGenerator):
            self.latent_code = self.model.sample_z()

        # temporary variables
        self.losses = []
        #
        if self.config.data.output_type in ["singleclass", "multiclass", "mixed"]:
            self.predictions = []
            self.targets = []
            self.predicted_classes = []
            self.correct = []

    def log_step(self, mode, pred, y, loss_logs):
        """ """
        for criterion in loss_logs:
            self.writer.add_scalar(
                mode + "_" + criterion,
                loss_logs[criterion],
                self.config.training.global_train_step,
            )

        self.losses.append(loss_logs["loss"])

        if len(
            set(["ce", "bce"]).intersection(self.config.task.criterions.keys())
        ) >= 1 and not isinstance(self.model, InvertibleGenerator):
            #
            self.targets.append(y.detach())
            self.predictions.append(pred.detach())
            #
            if "ce" in self.config.task.criterions.keys():
                class_prediction = pred.detach().argmax(-1)

            elif "bce" in self.config.task.criterions.keys():
                class_prediction = nn.Sigmoid()(pred) >= 0.5

            self.correct.append(
                (class_prediction == y.to(self.device)).type(torch.float)
            )
            self.predicted_classes.append(class_prediction.detach().to(torch.float32))

        #
        if hasattr(
            self.config.training,
            "global_" + mode + "_step",
        ):
            setattr(
                self.config.training,
                "global_" + mode + "_step",
                getattr(self.config.training, "global_" + mode + "_step") + 1,
            )

    def log_epoch(self, mode, pbar=None):
        """ """
        #
        loss_accumulated = torch.mean(torch.tensor(self.losses))
        self.writer.add_scalar(
            "epoch_" + mode + "_loss_accumulated",
            loss_accumulated.item(),
            self.config.training.epoch,
        )
        if not pbar is None:
            pbar.stored_values[mode + "_loss_accumulated"] = loss_accumulated.item()

        if "ce" in self.config.task.criterions.keys() and not isinstance(
            self.model, InvertibleGenerator
        ):
            try:
                targets_one_hot = torch.nn.functional.one_hot(
                    torch.cat(self.targets).to(torch.int64), self.output_channels
                ).to(torch.float32)
            except Exception:
                import pdb

                pdb.set_trace()

            predictions_one_hot = torch.nn.functional.one_hot(
                torch.cat(self.predicted_classes).to(torch.int64), self.output_channels
            ).to(torch.float32)

        if "bce" in self.config.task.criterions.keys() and not isinstance(
            self.model, InvertibleGenerator
        ):
            targets_one_hot = torch.cat(self.targets)
            predictions_one_hot = torch.cat(self.predicted_classes)
            correct_per_class = torch.cat(self.correct).mean(0)
            if not pbar is None:
                pbar.stored_values["correct_per_class"] = correct_per_class

            self.writer.add_histogram(
                "epoch_" + mode + "_correct_per_class",
                correct_per_class,
                self.config.training.epoch,
            )

        if len(
            set(["ce", "bce"]).intersection(self.config.task.criterions.keys())
        ) >= 1 and not isinstance(self.model, InvertibleGenerator):
            try:
                accuracy = torch.cat(self.correct).mean().item()

            except Exception:
                import pdb

                pdb.set_trace()
            self.writer.add_scalar(
                "epoch_" + mode + "_accuracy",
                accuracy,
                self.config.training.epoch,
            )
            self.writer.add_histogram(
                "epoch_" + mode + "_predicted_classes",
                predictions_one_hot.mean(0),
                self.config.training.epoch,
            )
            self.writer.add_histogram(
                "epoch_" + mode + "_targets",
                targets_one_hot.mean(0),
                self.config.training.epoch,
            )
            self.writer.add_histogram(
                mode + "_classes_difference",
                predictions_one_hot.mean(0).cpu() - targets_one_hot.mean(0).cpu(),
                self.config.training.epoch,
            )
            if not pbar is None:
                pbar.stored_values[mode + "_accuracy"] = accuracy
                pbar.stored_values[
                    mode + "_predicted_classes"
                ] = predictions_one_hot.mean(0).cpu()[:2]
                pbar.stored_values[mode + "_targets"] = targets_one_hot.mean(0).cpu()[
                    :2
                ]
                pbar.stored_values[mode + "_classes_difference"] = (
                    predictions_one_hot.mean(0).cpu() - targets_one_hot.mean(0).cpu()
                )[:2]
            #
            self.predictions = []
            self.targets = []
            self.predicted_classes = []
            self.correct = []

        else:
            accuracy = torch.exp(-loss_accumulated).item()

        if (
            isinstance(self.model, InvertibleGenerator)
            and self.config.data.input_type == "image"
        ):
            reconstructed_images = self.model.decode(
                self.model.encode(self.test_X.to(self.device))
            )
            reconstruction_collage = torch.cat(
                [reconstructed_images, self.test_X.to(self.device)]
            )
            torchvision.utils.save_image(
                reconstruction_collage.cpu().data,
                os.path.join(
                    self.base_dir,
                    "outputs",
                    "reconstruction_" + str(self.config.training.epoch) + ".png",
                ),
                normalize=True,
                nrow=self.config.training.val_batch_size,
                range=(-0.5, 0.5),
            )
            if hasattr(self.val_dataloader.dataset, "project_to_pytorch_default"):
                reconstruction_collage = (
                    self.val_dataloader.dataset.project_to_pytorch_default(
                        reconstruction_collage
                    )
                )

            else:
                print(
                    "Warning! If your dataloader uses another normalization than the PyTorch default [0,1] range data might be visualized incorrect!"
                    + "In that case add function project_to_pytorch_default() to your underlying dataset to correct visualization!"
                )

            self.writer.add_image(
                "reconstructions",
                torchvision.utils.make_grid(
                    reconstruction_collage,
                    nrow=self.config.training.val_batch_size,
                ),
                self.config.training.epoch,
            )

            #
            sample_images = self.model.decode(self.latent_code, reconstruct=False)
            torchvision.utils.save_image(
                sample_images.cpu().data,
                os.path.join(
                    self.base_dir,
                    "outputs",
                    "samples_" + str(self.config.training.epoch) + ".png",
                ),
                normalize=True,
                nrow=int(math.sqrt(self.config.training.val_batch_size)),
                range=(-0.5, 0.5),
            )
            if hasattr(self.val_dataloader.dataset, "project_to_pytorch_default"):
                sample_images = self.val_dataloader.dataset.project_to_pytorch_default(
                    sample_images
                )

            else:
                print(
                    "Warning! If your dataloader uses another normalization than the PyTorch default [0,1] range data might be visualized incorrect!"
                    + "In that case add function project_to_pytorch_default() to your underlying dataset to correct visualization!"
                )

            self.writer.add_image(
                "samples",
                torchvision.utils.make_grid(
                    sample_images,
                    nrow=int(math.sqrt(self.config.training.val_batch_size)),
                ),
                self.config.training.epoch,
            )

        self.losses = []

        return accuracy


def log_images_to_writer(dataloader, writer, tag="train"):
    dataloader_mixer_treatment = isinstance(dataloader, DataloaderMixer)
    if dataloader_mixer_treatment:
        dataloader_mixer_treatment &= (
            not hasattr(dataloader.train_config, "concatenate_batches")
            or not dataloader.train_config.concatenate_batches
        )

    if dataloader_mixer_treatment:
        iterator = iter(dataloader.dataloaders[0])

    else:
        iterator = iter(dataloader)

    for i in range(3):
        if i == 1:
            if dataloader_mixer_treatment:
                iterator = iter(dataloader.dataloaders[0])

            else:
                iterator = iter(dataloader)

        if i == 2 and dataloader_mixer_treatment and len(dataloader.dataloaders) > 1:
            iterator = iter(dataloader.dataloaders[1])

        sample_train_imgs, sample_train_y = next(iterator)

        if isinstance(sample_train_imgs, list):
            sample_train_imgs, sample_train_y = sample_train_imgs

        if hasattr(dataloader.dataset, "project_to_pytorch_default"):
            sample_train_imgs = dataloader.dataset.project_to_pytorch_default(
                sample_train_imgs
            )

        else:
            print(
                "Warning! If your dataloader uses another normalization than the PyTorch default [0,1]"
                + "range data might be visualized incorrect!"
                + "In that case add function project_to_pytorch_default() to your underlying dataset to correct visualization!"
            )

        sample_batch_label_str = "sample_" + tag + "_batch" + str(i) + "_"
        if isinstance(sample_train_y, torch.Tensor) and len(sample_train_y.shape) == 1:
            sample_batch_label_str += "_" + str(
                list(map(lambda x: int(x), list(sample_train_y)))
            )

        elif isinstance(sample_train_y, list) and len(sample_train_y) == 1:
            sample_batch_label_str += "_" + str(
                list(map(lambda x: int(x), list(sample_train_y[0])[:128]))
            )

        writer.add_image(
            sample_batch_label_str,
            torchvision.utils.make_grid(sample_train_imgs[:128], 128),
        )

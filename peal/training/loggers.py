import torch
import os
import math
import torchvision
import torch.nn as nn

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

        if "output_size" in self.config["task"].keys():
            self.output_size = self.config["task"]["output_size"]

        else:
            self.output_size = self.config["data"]["output_size"]

        self.device = "cuda" if next(
            self.model.parameters()).is_cuda else "cpu"
        #
        if "ce" in config["task"]["criterions"].keys():
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
        if self.config["data"]["output_type"] in ["singleclass", "multiclass"]:
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
                self.config["training"]["global_train_step"],
            )

        self.losses.append(loss_logs["loss"])

        if (
            len(
                set(["ce", "bce"]).intersection(
                    self.config["task"]["criterions"].keys()
                )
            )
            >= 1 and not isinstance(self.model, InvertibleGenerator)
        ):
            #
            self.targets.append(y.detach())
            self.predictions.append(pred.detach())
            #
            if "ce" in self.config["task"]["criterions"].keys():
                class_prediction = pred.detach().argmax(-1)

            elif "bce" in self.config["task"]["criterions"].keys():
                class_prediction = nn.Sigmoid()(pred) >= 0.5

            self.correct.append(
                (class_prediction == y.to(self.device)).type(torch.float)
            )
            self.predicted_classes.append(
                class_prediction.detach().to(torch.float32))

        #
        if not "global_" + mode + "_step" in self.config["training"].keys():
            self.config["training"]["global_" + mode + "_step"] = 0

        self.config["training"]["global_" + mode + "_step"] += 1

    def log_epoch(self, mode):
        """ """
        #
        loss_accumulated = torch.mean(torch.tensor(self.losses))
        self.writer.add_scalar(
            "epoch_" + mode + "_loss_accumulated",
            loss_accumulated.item(),
            self.config["training"]["epoch"],
        )
        print("")
        print("epoch_" + mode + "_loss_accumulated",
              str(loss_accumulated.item()))

        if "ce" in self.config["task"]["criterions"].keys() and not isinstance(self.model, InvertibleGenerator):
            targets_one_hot = torch.nn.functional.one_hot(
                torch.cat(self.targets).to(torch.int64), self.output_size
            ).to(torch.float32)
            predictions_one_hot = torch.nn.functional.one_hot(
                torch.cat(self.predicted_classes).to(
                    torch.int64), self.output_size
            ).to(torch.float32)

        if "bce" in self.config["task"]["criterions"].keys() and not isinstance(self.model, InvertibleGenerator):
            targets_one_hot = torch.cat(self.targets)
            predictions_one_hot = torch.cat(self.predicted_classes)
            correct_per_class = torch.cat(self.correct).mean(0)
            print("correct_per_class: " + str(correct_per_class))
            self.writer.add_histogram(
                "epoch_" + mode + "_correct_per_class",
                correct_per_class,
                self.config["training"]["epoch"],
            )

        if (
            len(
                set(["ce", "bce"]).intersection(
                    self.config["task"]["criterions"].keys()
                )
            )
            >= 1 and not isinstance(self.model, InvertibleGenerator)
        ):
            accuracy = torch.cat(self.correct).mean().item()
            self.writer.add_scalar(
                "epoch_" + mode + "_accuracy",
                accuracy,
                self.config["training"]["epoch"],
            )
            self.writer.add_histogram(
                "epoch_" + mode + "_predicted_classes",
                predictions_one_hot.mean(0),
                self.config["training"]["epoch"],
            )
            self.writer.add_histogram(
                "epoch_" + mode + "_targets",
                targets_one_hot.mean(0),
                self.config["training"]["epoch"],
            )
            self.writer.add_histogram(
                mode + "_classes_difference",
                predictions_one_hot.mean(0).cpu() -
                targets_one_hot.mean(0).cpu(),
                self.config["training"]["epoch"],
            )
            print("epoch_" + mode + "_accuracy", str(accuracy))
            print(
                "epoch_"
                + mode
                + "_predicted_classes: "
                + str(predictions_one_hot.mean(0)[:3])
            )
            print("epoch_" + mode + "_targets: " +
                  str(targets_one_hot.mean(0)[:3]))
            print(
                "epoch_"
                + mode
                + "_classes_difference: "
                + str(
                    predictions_one_hot.mean(0).cpu()[:3]
                    - targets_one_hot.mean(0).cpu()[:3]
                )
            )
            #
            self.predictions = []
            self.targets = []
            self.predicted_classes = []
            self.correct = []

        else:
            accuracy = torch.exp(-loss_accumulated).item()

        if (
            isinstance(self.model, InvertibleGenerator)
            and self.config["data"]["input_type"] == "image"
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
                    "reconstruction_" +
                    str(self.config["training"]["epoch"]) + ".png",
                ),
                normalize=True,
                nrow=self.config["training"]["val_batch_size"],
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
                    nrow=self.config["training"]["val_batch_size"],
                ),
                self.config["training"]["epoch"],
            )

            #
            sample_images = self.model.decode(
                self.latent_code, reconstruct=False)
            torchvision.utils.save_image(
                sample_images.cpu().data,
                os.path.join(
                    self.base_dir,
                    "outputs",
                    "samples_" +
                    str(self.config["training"]["epoch"]) + ".png",
                ),
                normalize=True,
                nrow=int(math.sqrt(self.config["training"]["val_batch_size"])),
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
                    nrow=int(
                        math.sqrt(self.config["training"]["val_batch_size"])),
                ),
                self.config["training"]["epoch"],
            )

        self.losses = []

        return accuracy

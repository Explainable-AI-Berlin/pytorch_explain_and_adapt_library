import torch
import math
import numpy as np
import torch.nn.functional as F

from torch import nn

from peal.global_utils import onehot


def cross_entropy_loss(input, target, size_average=True, latent_code=None):
    input = F.log_softmax(input, dim=1)
    loss = -torch.sum(input * target)
    if size_average:
        return loss / input.size(0)
    else:
        return loss


class OnehotCrossEntropyLoss(object):
    def __init__(self, size_average=True):
        self.size_average = size_average

    def __call__(self, input, target):
        return cross_entropy_loss(input, target, self.size_average)


def orthogonality_criterion(model, pred, y, latent_code=None):
    """ """
    loss = torch.tensor(0.0).to(next(model.parameters()).device)
    for parameter_idx, parameter in enumerate(model.parameters()):
        if len(parameter.shape) == 1:
            continue
        elif len(parameter.shape) == 4:
            parameter_reshaped = torch.reshape(
                parameter,
                [
                    parameter.shape[0],
                    parameter.shape[1] * parameter.shape[2] * parameter.shape[3],
                ],
            )
        else:
            parameter_reshaped = parameter

        loss += torch.linalg.matrix_norm(
            torch.matmul(parameter_reshaped, parameter_reshaped.t())
            - torch.eye(parameter_reshaped.shape[0]).to(next(model.parameters()).device)
        )
    return loss


def l1_criterion(model, pred, y, latent_code=None):
    """ """
    loss = torch.tensor(0.0).to(next(model.parameters()).device)
    num_weights = 0
    for parameter_idx, parameter in enumerate(model.parameters()):
        loss += torch.sum(torch.abs(parameter))
        num_weights += int(np.prod(list(parameter.shape)))

    return loss / num_weights


def l2_criterion(model, pred, y, latent_code=None):
    """ """
    loss = torch.tensor(0.0).to(next(model.parameters()).device)
    num_weights = 0
    for parameter_idx, parameter in enumerate(model.parameters()):
        loss += torch.sum(torch.square(parameter))
        num_weights += int(np.prod(list(parameter.shape)))

    return loss / num_weights


def mixed_bce_mse_criterion(model, y_pred, y_target, latent_code=None):
    loss_discrete = torch.nn.BCEWithLogitsLoss()(
        y_pred[: model.config.data.output_split],
        y_target[: model.config.data.output_split],
    )
    loss_continuous = torch.nn.MSELoss()(
        y_pred[model.config.data.output_split :],
        y_target[model.config.data.output_split :],
    )
    return loss_discrete + loss_continuous


#
class LogDiscriminantJacobianCriterion(nn.Module):
    def __init__(self, config, writer, device):
        super().__init__()
        self.config = config
        self.writer = writer
        self.device = device

    def forward(self, model, y_pred, y_target, latent_code=None):
        n_pixel = np.prod(self.config.data.input_size)
        # return torch.mean(torch.norm(y_pred[2])) / (math.log(2) * n_pixel)
        loss = y_pred[2]
        return -torch.mean(loss / (math.log(2) * n_pixel))


#
class LogProbCriterion(nn.Module):
    def __init__(self, config, writer, device):
        super().__init__()
        self.config = config
        self.writer = writer
        self.device = device

    def forward(self, model, y_pred, y_target):
        n_pixel = np.prod(self.config.data.input_size)
        loss = y_pred[1]
        return -torch.mean(loss / (math.log(2) * n_pixel))


#
class ConstantCriterion(nn.Module):
    def __init__(self, config, writer, device):
        super().__init__()
        self.config = config
        self.writer = writer
        self.device = device

    def forward(self, model, y_pred, y_target):
        n_pixel = np.prod(self.config.data.input_size)
        # just for sake that loss does not get negative
        loss = -self.config.architecture.n_bits * n_pixel
        return -torch.tensor(loss / (math.log(2) * n_pixel)).to(self.device)


#
def isotropic_likelihood_criterion(model, y_pred, y_target):
    if isinstance(y_pred[1], list):
        loss = torch.tensor(0.0).to(next(model.parameters()).device)
        for y_pred_ in y_pred[1]:
            loss += torch.mean(torch.square(y_pred_))

        return loss

    else:
        return torch.mean(torch.square(y_pred[1]))


#
def reconstruction_criterion(model, y_pred, y_target):
    return torch.nn.MSELoss()(y_pred[0], y_target)


#
def supervised_criterion(model, y_pred, y_target):
    pass


#
class AdversarialCriterion(nn.Module):
    """ """

    def __init__(self, config, writer, device):
        """ """
        super().__init__()
        self.config = config
        self.writer = writer
        self.device = device
        params = {
            "n_layer": 3,
            "gan_type": "nsgan",
            "dim": 32,
            "norm": "bn",
            "activ": "lrelu",
            "num_scales": 3,
            "pad_type": "zero",
        }
        self.discriminator = MsImageDis(
            input_dim=self.config.data.input_size[0], params=params, device=device
        ).to(self.device)

        if self.config.training.optimizer == "sgd":
            self.dis_optimizer = torch.optim.SGD(
                self.discriminator.parameters(), lr=config.training.learning_rate
            )

        elif self.config.training.optimizer == "adam":
            self.dis_optimizer = torch.optim.Adam(
                self.discriminator.parameters(), lr=config.training.learning_rate
            )

    def forward(self, model, y_pred, y_target):
        """ """
        if model.training:
            self.discriminator.train()
            self.dis_optimizer.zero_grad()
            dis_loss = self.discriminator.calc_dis_loss(
                torch.clone(y_pred[0]).detach(), y_target.to(self.device)
            )
            self.writer.add_scalar(
                "train_adversarial_dis",
                dis_loss.detach().cpu().item(),
                self.config.training.global_train_step,
            )
            dis_loss.backward()
            self.dis_optimizer.step()
            self.discriminator.eval()

        return self.discriminator.calc_gen_loss(y_pred[0])


def cross_entropy_criterion(model, y_pred, y_target, latent_code=None):
    if y_pred.shape == y_target.shape:
        return OnehotCrossEntropyLoss()(y_pred, y_target)

    if isinstance(y_pred, tuple):
        y_pred = y_pred[0]

    y_pred = y_pred.reshape([int(np.prod(y_pred.shape[:-1])), y_pred.shape[-1]])
    y_target = y_target.flatten().to(torch.int64)
    return nn.CrossEntropyLoss()(y_pred, y_target)


def latent_convexity_criterion(model, y_pred, y_target, latent_code=None):
    indices = torch.randperm(latent_code.size(0))
    data2 = latent_code[indices].to(latent_code)
    targets2 = y_target[indices].to(latent_code)

    targets_onehot = onehot(y_target.to(latent_code), model.config.task.output_channels)
    targets2_onehot = onehot(targets2,model.config.task.output_channels)

    #lam = torch.FloatTensor([np.random.beta(model.config.training.alpha, model.config.training.alpha)]).to(latent_code)
    lam = torch.rand(y_target.shape).to(latent_code)[:,None]
    data = latent_code * lam + data2 * (1 - lam)
    targets_new = targets_onehot * lam + targets2_onehot * (1 - lam)
    logits = model.get_last_layer()(data)
    return cross_entropy_criterion(model, logits, targets_new)


available_criterions = {
    "ce": cross_entropy_criterion,
    "bce": lambda model, y_pred, y_target: nn.BCEWithLogitsLoss()(y_pred, y_target),
    "mse": lambda model, y_pred, y_target: nn.MSELoss()(y_pred, y_target),
    "mae": lambda model, y_pred, y_target: nn.L1Loss()(y_pred, y_target),
    "mixed": mixed_bce_mse_criterion,
    "orthogonality": orthogonality_criterion,
    "l1": l1_criterion,
    "l2": l2_criterion,
    "ldj": LogDiscriminantJacobianCriterion,
    "logprob": LogProbCriterion,
    "constant": ConstantCriterion,
    "likelihood": isotropic_likelihood_criterion,
    "reconstruction": reconstruction_criterion,
    # 'adversarial'    : AdversarialCriterion,
    "supervised": supervised_criterion,
    "lc": latent_convexity_criterion,
}


def get_criterions(config):
    #
    criterions = {}
    for criterion in config.task.criterions.keys():
        criterions[criterion] = available_criterions[criterion]

    return criterions

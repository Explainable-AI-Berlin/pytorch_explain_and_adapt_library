"""Loss functions ("criterions") used by PEAL's predictor trainers.

A criterion is a callable ``criterion(model, y_pred, y_target, latent_code)``
returning a scalar tensor. ``model`` is passed so a criterion can read the
training config (``model.config.task`` / ``model.config.data``) or, for the
weight regularisers, iterate over the parameters. ``ModelTrainer.run_epoch``
looks every key of ``config.task.criterions`` up in ``available_criterions``
and sums the results weighted by the configured factors. ``l1``, ``l2`` and
``orthogonality`` are weight regularisers; the others are data losses.
"""

import torch
import numpy as np
import torch.nn.functional as F

from torch import nn

from peal.global_utils import onehot


def cross_entropy_loss(input, target, size_average=True, latent_code=None):
    """Cross-entropy between logits and a soft (one-hot or mixed) target.

    Parameters
    ----------
    input : torch.Tensor
        Logits of shape ``[N, C]``; log-softmax is applied here.
    target : torch.Tensor
        Target distribution of the same shape (one-hot or a mixup of them).
    size_average : bool, optional
        Divide the summed loss by the batch size ``N`` (default) or return
        the plain sum.
    latent_code : optional
        Ignored; present for the common criterion signature.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    input = F.log_softmax(input, dim=1)
    loss = -torch.sum(input * target)
    if size_average:
        return loss / input.size(0)
    else:
        return loss


class OnehotCrossEntropyLoss(object):
    """Callable wrapper around :func:`cross_entropy_loss`.

    Parameters
    ----------
    size_average : bool, optional
        Forwarded to :func:`cross_entropy_loss`.
    """

    def __init__(self, size_average=True):
        """Store the ``size_average`` flag used on every call."""
        self.size_average = size_average

    def __call__(self, input, target, latent_code=None):
        """Soft-target cross-entropy of ``input`` and ``target``; ``latent_code`` is ignored."""
        return cross_entropy_loss(input, target, self.size_average)


def orthogonality_criterion(model, pred, y, latent_code=None):
    """Penalise non-orthogonal weight matrices of ``model``.

    Every parameter with two or more dimensions is reshaped to ``[out, in]``
    (conv kernels are flattened over ``in * kh * kw``) and the Frobenius norm
    of ``W W^T - I`` is summed. Vectors (biases, norm weights) are skipped.
    ``pred``, ``y`` and ``latent_code`` are ignored.

    Returns
    -------
    torch.Tensor
        Scalar penalty on the model's device.
    """
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
    """Mean absolute value over all parameters of ``model`` (L1 weight decay).

    ``pred``, ``y`` and ``latent_code`` are ignored; the sum is divided by the
    total number of weights so the value is comparable across architectures.
    """
    loss = torch.tensor(0.0).to(next(model.parameters()).device)
    num_weights = 0
    for parameter_idx, parameter in enumerate(model.parameters()):
        loss += torch.sum(torch.abs(parameter))
        num_weights += int(np.prod(list(parameter.shape)))

    return loss / num_weights


def l2_criterion(model, pred, y, latent_code=None):
    """Mean squared value over all parameters of ``model`` (L2 weight decay).

    ``pred``, ``y`` and ``latent_code`` are ignored; the sum is divided by the
    total number of weights so the value is comparable across architectures.
    """
    loss = torch.tensor(0.0).to(next(model.parameters()).device)
    num_weights = 0
    for parameter_idx, parameter in enumerate(model.parameters()):
        loss += torch.sum(torch.square(parameter))
        num_weights += int(np.prod(list(parameter.shape)))

    return loss / num_weights


def mixed_bce_mse_criterion(model, y_pred, y_target, latent_code=None):
    """BCE on the discrete outputs plus weighted MSE on the continuous ones.

    Output columns before ``model.config.data.output_split`` are logits of
    binary attributes and are scored with
    ``BCEWithLogitsLoss(pos_weight=model.config.task.mixed_bce_pos_weight)``;
    the remaining columns are regression targets scored with MSE and weighted
    by ``model.config.task.bce_mse_mix``.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    pos_weight = torch.tensor([model.config.task.mixed_bce_pos_weight]).to(
        y_pred.device
    )
    loss_discrete = torch.nn.BCEWithLogitsLoss(pos_weight)(
        y_pred[:, : model.config.data.output_split],
        y_target[:, : model.config.data.output_split],
    )
    loss_continuous = torch.nn.MSELoss()(
        y_pred[:, model.config.data.output_split :],
        y_target[:, model.config.data.output_split :],
    )
    return loss_discrete + model.config.task.bce_mse_mix * loss_continuous


def cross_entropy_criterion(model, y_pred, y_target, latent_code=None):
    """Cross-entropy for hard or soft targets.

    If ``y_pred`` and ``y_target`` have the same shape the target is taken as
    a distribution and :class:`OnehotCrossEntropyLoss` is used. Otherwise the
    logits are flattened to ``[N * ..., C]`` (a tuple output is reduced to its
    first element), the targets to ``int64`` class indices, and
    ``nn.CrossEntropyLoss`` is applied.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    if y_pred.shape == y_target.shape:
        return OnehotCrossEntropyLoss()(y_pred, y_target)

    if isinstance(y_pred, tuple):
        y_pred = y_pred[0]

    y_pred = y_pred.reshape([int(np.prod(y_pred.shape[:-1])), y_pred.shape[-1]])
    y_target = y_target.flatten().to(torch.int64)
    return nn.CrossEntropyLoss()(y_pred, y_target)


def latent_convexity_criterion(model, y_pred, y_target, latent_code=None):
    """Mixup in the penultimate feature space (latent-convexity regulariser).

    Pairs every latent code with a randomly permuted partner, interpolates the
    codes and the one-hot targets with a uniform ``lam`` per sample, pushes
    the mixed codes through ``model.get_last_layer()`` and scores the result
    with :func:`cross_entropy_criterion`. ``y_pred`` is unused;
    ``latent_code`` must be the ``[N, D]`` activations the trainer extracted
    for the batch.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    indices = torch.randperm(latent_code.size(0))
    data2 = latent_code[indices].to(latent_code)
    targets2 = y_target[indices].to(latent_code)

    targets_onehot = onehot(y_target.to(latent_code), model.config.task.output_channels)
    targets2_onehot = onehot(targets2, model.config.task.output_channels)

    # lam = torch.FloatTensor([np.random.beta(model.config.training.alpha, model.config.training.alpha)]).to(latent_code)
    lam = torch.rand(y_target.shape).to(latent_code)[:, None]
    data = latent_code * lam + data2 * (1 - lam)
    targets_new = targets_onehot * lam + targets2_onehot * (1 - lam)
    logits = model.get_last_layer()(data)
    return cross_entropy_criterion(model, logits, targets_new)


class FocalLoss(nn.Module):
    """Binary focal loss on logits (Lin et al., 2017).

    Parameters
    ----------
    alpha : float, optional
        Constant weight of every term (default 0.25).
    gamma : float, optional
        Focusing exponent; ``0`` recovers plain BCE (default 1).
    reduction : {"mean", "sum", "none"}, optional
        How the per-element losses are aggregated.
    """

    def __init__(self, alpha=0.25, gamma=1, reduction="mean"):
        """Store the focal-loss hyperparameters."""
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """Return ``alpha * (1 - p_t) ** gamma * BCE`` reduced as configured."""
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == "mean":
            return torch.mean(F_loss)
        elif self.reduction == "sum":
            return torch.sum(F_loss)
        else:
            return F_loss


def mixed_focal_mse_criterion(model, y_pred, y_target, latent_code=None):
    """Focal loss on the discrete outputs plus weighted MSE on the continuous.

    Same column split as :func:`mixed_bce_mse_criterion`, but with
    :class:`FocalLoss` (``focal_alpha``/``focal_gamma`` from
    ``model.config.task``) and mixing factor ``model.config.task.focal_mse_mix``.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    loss_discrete = FocalLoss(
        model.config.task.focal_alpha, model.config.task.focal_gamma
    )(
        y_pred[:, : model.config.data.output_split],
        y_target[:, : model.config.data.output_split],
    )
    loss_continuous = torch.nn.MSELoss()(
        y_pred[:, model.config.data.output_split :],
        y_target[:, model.config.data.output_split :],
    )
    return loss_discrete + model.config.task.focal_mse_mix * loss_continuous


def focal_criterion(model, y_pred, y_target, latent_code=None):
    """Focal loss on every output column.

    ``alpha``/``gamma`` come from ``model.config.task.focal_alpha`` and
    ``focal_gamma``.
    """
    return FocalLoss(model.config.task.focal_alpha, model.config.task.focal_gamma)(
        y_pred, y_target
    )


# Registry read by get_criterions(); the lambdas are plain torch losses.
available_criterions = {
    "ce": cross_entropy_criterion,
    "bce": lambda model, y_pred, y_target, latent_code: nn.BCEWithLogitsLoss()(
        y_pred, y_target.float()
    ),
    "mse": lambda model, y_pred, y_target, latent_code: nn.MSELoss()(
        y_pred[:, model.config.data.output_split :],
        y_target[:, model.config.data.output_split :],
    ),
    "mae": lambda model, y_pred, y_target, latent_code: nn.L1Loss()(y_pred, y_target),
    "mixed": mixed_bce_mse_criterion,
    "focal_mixed": mixed_focal_mse_criterion,
    "focal": focal_criterion,
    "orthogonality": orthogonality_criterion,
    "l1": l1_criterion,
    "l2": l2_criterion,
    "lc": latent_convexity_criterion,
}


def get_criterions(config):
    """Select the criterions named in ``config.task.criterions``.

    Parameters
    ----------
    config : PredictorConfig
        Config whose ``task.criterions`` dict maps criterion names to weights.

    Returns
    -------
    dict
        Name -> callable from ``available_criterions`` in config order; an
        unknown name raises ``KeyError``.
    """
    #
    criterions = {}
    for criterion in config.task.criterions.keys():
        criterions[criterion] = available_criterions[criterion]

    return criterions

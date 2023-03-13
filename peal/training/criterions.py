import torch
import math
import numpy as np

from torch import nn

def orthogonality_criterion(model, pred, y):
    '''

    '''
    loss = torch.tensor(0.0).to(next(model.parameters()).device)
    for parameter_idx, parameter in enumerate(model.parameters()):
        if len(parameter.shape) == 1:
            continue
        elif len(parameter.shape) == 4:
            parameter_reshaped = torch.reshape(parameter, [parameter.shape[0], parameter.shape[1] * parameter.shape[2] * parameter.shape[3]])
        else:
            parameter_reshaped = parameter
        loss += torch.linalg.matrix_norm(torch.matmul(
            parameter_reshaped, parameter_reshaped.t()) - torch.eye(parameter_reshaped.shape[0]).to(next(model.parameters()).device)
        )
    return loss


def l1_criterion(model, pred, y):
    '''

    '''
    loss = torch.tensor(0.0).to(next(model.parameters()).device)
    num_weights = 0
    for parameter_idx, parameter in enumerate(model.parameters()):
        loss += torch.sum(torch.abs(parameter))
        num_weights += int(np.prod(list(parameter.shape)))

    return loss / num_weights


def l2_criterion(model, pred, y):
    '''

    '''
    loss = torch.tensor(0.0).to(next(model.parameters()).device)
    num_weights = 0
    for parameter_idx, parameter in enumerate(model.parameters()):
        loss += torch.sum(torch.square(parameter))
        num_weights += int(np.prod(list(parameter.shape)))

    return loss / num_weights


#
def mixed_bce_mse_criterion(model, y_pred, y_target):
    loss_discrete = torch.nn.BCEWithLogitsLoss()(y_pred[:config['data']['output_split']], y_target[:config['data']['output_split']])
    loss_continuous = torch.nn.MSELoss()(y_pred[config['data']['output_split']:], y_target[config['data']['output_split']:])
    return loss_discrete + loss_continuous


#
class LogDiscriminantJacobianCriterion(nn.Module):
    def __init__(self, config, writer, device):
        super().__init__()
        self.config = config
        self.writer = writer
        self.device = device

    def forward(self, model, y_pred, y_target):
        n_pixel = np.prod(self.config['data']['input_size'])
        #return torch.mean(torch.norm(y_pred[2])) / (math.log(2) * n_pixel)
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
        n_pixel = np.prod(self.config['data']['input_size'])
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
        n_pixel = np.prod(self.config['data']['input_size'])
        # just for sake that loss does not get negative
        loss = - self.config['architecture']['n_bits'] * n_pixel
        return - torch.tensor(loss / (math.log(2) * n_pixel)).to(self.device)


#
def isotropic_likelihood_criterion(model, y_pred, y_target):
    #return torch.mean(torch.square(y_pred[1]))
    p = torch.distributions.normal.Normal(loc=0.0, scale=1.0)
    return - p.log_prob(y_pred[1]).mean() #.mean(list(range(len(y_pred[1].shape)))[1:])


#
def reconstruction_criterion(model, y_pred, y_target):
    return torch.nn.MSELoss()(y_pred[0], y_target)


#
def supervised_criterion(model, y_pred, y_target):
    pass


#
class AdversarialCriterion(nn.Module):
    '''

    '''
    def __init__(self, config, writer, device):
        '''

        '''
        super().__init__()
        self.config = config
        self.writer = writer
        self.device = device
        params = {
            'n_layer' : 3,
            'gan_type' : 'nsgan',
            'dim' : 32,
            'norm' : 'bn',
            'activ' : 'lrelu',
            'num_scales' : 3,
            'pad_type' : 'zero',
        }
        self.discriminator = MsImageDis(
            input_dim = self.config['data']['input_size'][0],
            params = params,
            device = device
        ).to(self.device)

        if self.config['training']['optimizer'] == 'sgd':
            self.dis_optimizer = torch.optim.SGD(self.discriminator.parameters(), lr=config['training']['learning_rate'])

        elif self.config['training']['optimizer'] == 'adam':
            self.dis_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=config['training']['learning_rate'])
    
    def forward(self, model, y_pred, y_target):
        '''

        '''
        if model.training:
            self.discriminator.train()
            self.dis_optimizer.zero_grad()
            dis_loss = self.discriminator.calc_dis_loss(torch.clone(y_pred[0]).detach(), y_target.to(self.device))
            self.writer.add_scalar('train_adversarial_dis', dis_loss.detach().cpu().item(), self.config['training']['global_train_step'])
            dis_loss.backward()
            self.dis_optimizer.step()
            self.discriminator.eval()

        return self.discriminator.calc_gen_loss(y_pred[0])

available_criterions = {
    'ce'             : lambda model, y_pred, y_target: nn.CrossEntropyLoss()(y_pred, y_target),
    'bce'            : lambda model, y_pred, y_target: nn.BCEWithLogitsLoss()(y_pred, y_target),
    'mse'            : lambda model, y_pred, y_target: nn.MSELoss()(y_pred, y_target),
    'mae'            : lambda model, y_pred, y_target: nn.L1Loss()(y_pred, y_target),
    'mixed'          : mixed_bce_mse_criterion,
    'orthogonality'  : orthogonality_criterion,
    'l1'             : l1_criterion,
    'l2'             : l2_criterion,
    'ldj'            : LogDiscriminantJacobianCriterion,
    'logprob'        : LogProbCriterion,
    'constant'       : ConstantCriterion,
    'likelihood'     : isotropic_likelihood_criterion,
    'reconstruction' : reconstruction_criterion,
    #'adversarial'    : AdversarialCriterion,
    'supervised'     : supervised_criterion
}

def get_criterions(config):
    #
    criterions = {}
    for criterion in config['task']['criterions'].keys():
        criterions[criterion] = available_criterions[criterion]

    return criterions
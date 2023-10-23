import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['kl_criterion', 'kl_bern_criterion', 'reconstruction_loss', 'adversarial_loss', 'perceptual_loss']


def kl_criterion(mu, logvar):
    KLD = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return KLD.mean()


def kl_bern_criterion(x):
    KLD = torch.mul(x, torch.log(x + 1e-20) - math.log(0.5)) + torch.mul(1 - x, torch.log(1 - x + 1e-20) - math.log(1 - 0.5))
    return KLD.mean()


def reconstruction_loss(output, target):
    criterion = nn.MSELoss()
    return criterion(output, target)


def adversarial_loss(logits, target_is_real=True):
    if target_is_real:
        targets = torch.ones_like(logits)
    else:
        targets = torch.zeros_like(logits)
    criterion = nn.BCEWithLogitsLoss()
    return criterion(logits, targets)


class PerceptualLoss(nn.Module):
    def __init__(self, vgg_model, layer_weights):
        super(PerceptualLoss, self).__init__()
        self.vgg_model = vgg_model
        self.layer_weights = layer_weights
        self.criterion = nn.L1Loss()

    def forward(self, input, target):
        input_features = self.vgg_model(input)
        target_features = self.vgg_model(target)

        loss = 0
        for i, (weight, input_feature, target_feature) in enumerate(zip(self.layer_weights, input_features, target_features)):
            loss += weight * self.criterion(input_feature, target_feature)

        return loss

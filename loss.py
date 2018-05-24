import torch
from torch import nn
from torch.nn import functional as F

import numpy as np


class LossBinary:
    """
    Loss defined as BCE - log(soft_jaccard)

    Vladimir Iglovikov, Sergey Mushinskiy, Vladimir Osin,
    Satellite Imagery Feature Detection using Deep Convolutional Neural Network: A Kaggle Competition
    arXiv:1706.06169
    """

    def __init__(self, jaccard_weight=0):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        loss = self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            jaccard_target = (targets == 1).float()
            jaccard_output = F.sigmoid(outputs)

            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()

            loss -= self.jaccard_weight * torch.log((intersection + eps) / (union - intersection + eps))
        return loss


class MyLossMulti:
    def __init__(self, jaccard_weight=0, num_classes=1, betas=None, class_weights=None):
        self.nll_loss = nn.NLLLoss(weight=class_weights)
        self.jaccard_weight = jaccard_weight
        self.num_classes=num_classes
        self.betas = betas

    def __call__(self, outputs, targets):
#         loss = self.nll_loss(outputs, targets)
        loss = torch.tensor(0, dtype=torch.float, requires_grad=True)
        if self.jaccard_weight:
            cls_weight = self.jaccard_weight / self.num_classes
            eps = 1e-15
            
            # add f1 scores to loss
            for cls, beta in zip([1,2],[.5, 2]):
                target = (targets == cls).float()
                output = outputs[:, cls].exp()
                correct = (target*output).sum().item()
                prec = correct / (output.sum().item() + eps)
                recall = correct / (target.sum().item() + eps)
                f_score = (1+beta**2)*(prec*recall)/(beta**2*prec+recall)
                loss = loss + (1-f_score)/2
#             loss /= 2
        return loss

class LossMulti:
    def __init__(self, jaccard_weight=0, num_classes=1, betas=None, class_weights=None):
        self.nll_loss = nn.NLLLoss(weight=class_weights)
        self.jaccard_weight = jaccard_weight
        self.num_classes=num_classes
        self.betas = betas

    def __call__(self, outputs, targets):
        loss = self.nll_loss(outputs, targets)
        if self.jaccard_weight:
            cls_weight = self.jaccard_weight / self.num_classes
            eps = 1e-15
            
            for cls in range(self.num_classes):
                jaccard_target = (targets == cls).float()
                jaccard_output = outputs[:, cls].exp()
                intersection = (jaccard_output * jaccard_target).sum()

                union = jaccard_output.sum() + jaccard_target.sum() + eps
                loss += (1 - intersection / (union - intersection)) * cls_weight

            loss /= (1 + self.jaccard_weight)
        return loss

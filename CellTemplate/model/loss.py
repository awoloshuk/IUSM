import torch.nn.functional as F
import torch.nn as nn


def nll_loss(output, target):
    return F.nll_loss(output, target)

def cross_entropy_loss(output, target):
    lossF = nn.CrossEntropyLoss()
    return lossF(output, target)

def bce_loss(output, target):
    lossF = nn.BCELoss()
    return lossF(output,target)

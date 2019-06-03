import torch.nn.functional as F
import torch.nn as nn

def nll_loss():
    return F.nll_loss()

def cross_entropy_loss(weights):
    return nn.CrossEntropyLoss(weight = weights)

def bce_loss(output, target):
    lossF = nn.BCELoss()
    return lossF(output,target)

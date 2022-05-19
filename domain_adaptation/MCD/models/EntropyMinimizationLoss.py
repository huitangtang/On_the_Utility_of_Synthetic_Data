import torch
import torch.nn as nn
import torch.nn.functional as F


class EMLossForTarget(nn.Module):

    def __init__(self):
        super(EMLossForTarget, self).__init__()

    def forward(self, input):
        return - (F.softmax(input, dim=1) * F.log_softmax(input, dim=1)).sum(1).mean()
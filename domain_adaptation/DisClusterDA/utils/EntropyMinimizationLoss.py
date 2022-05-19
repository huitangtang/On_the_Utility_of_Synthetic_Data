import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import ipdb


class AdaptiveFilteringEMLossForTarget(nn.Module):

    def __init__(self, eps):
        super(AdaptiveFilteringEMLossForTarget, self).__init__()
        self.eps = eps

    def forward(self, prob):        
        temp = torch.zeros(prob.size()).cuda(prob.device)
        temp[prob.data == 0] = self.eps 
        temp = Variable(temp)
        
        neg_ent = ((prob * ((prob + temp).log())).sum(1)).exp()
        
        loss = - (((prob * ((prob + temp).log())).sum(1)) * neg_ent).mean()

        return loss

 
class EMLossForTarget(nn.Module):

    def __init__(self, eps):
        super(EMLossForTarget, self).__init__()
        self.eps = eps

    def forward(self, prob):
        temp = torch.zeros(prob.size()).cuda(prob.device)
        temp[prob.data == 0] = self.eps 
        temp = Variable(temp)
        
        loss = - ((prob * ((prob + temp).log())).sum(1)).mean()

        return loss



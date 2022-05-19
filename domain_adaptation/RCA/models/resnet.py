from torchvision import models
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function, Variable


class ResBase(nn.Module):
    def __init__(self, option='resnet50', pret=False, num_classes=1000):
        super(ResBase, self).__init__()
        self.dim = 2048
        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=pret)
            self.dim = 512
        if option == 'resnet34':
            model_ft = models.resnet34(pretrained=pret)
            self.dim = 512
        if option == 'resnet50':
            model_ft = models.resnet50(pretrained=pret)
        if option == 'resnet101':
            model_ft = models.resnet101(pretrained=pret)
        if option == 'resnet152':
            model_ft = models.resnet152(pretrained=pret)
            
        self.fc = nn.Linear(self.dim, num_classes)
        self.fc_joint = nn.Linear(self.dim, num_classes*2)
        mod = list(model_ft.children())
        mod.pop()
        self.features = nn.Sequential(*mod)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), self.dim)
        y1 = self.fc(x)
        y2 = self.fc_joint(x)
        
        return y1, y2
        
        

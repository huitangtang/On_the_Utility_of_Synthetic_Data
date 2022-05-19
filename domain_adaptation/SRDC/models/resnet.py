from torchvision import models
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function, Variable


class ResBase(nn.Module):
    def __init__(self, option='resnet50', pret=False, num_classes=1000, num_neurons=None):
        super(ResBase, self).__init__()
        self.num_neurons = num_neurons
        self.block_expansion = 1
        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=pret)
        if option == 'resnet34':
            model_ft = models.resnet34(pretrained=pret)
        if option == 'resnet50':
            model_ft = models.resnet50(pretrained=pret)
            self.block_expansion = 4
        if option == 'resnet101':
            model_ft = models.resnet101(pretrained=pret)
            self.block_expansion = 4
        if option == 'resnet152':
            model_ft = models.resnet152(pretrained=pret)
            self.block_expansion = 4
        
        self.feat1_dim = 512 * self.block_expansion
        if self.num_neurons:
            self.feat2_dim = self.num_neurons * self.block_expansion
            self.fc1 = nn.Sequential(nn.Linear(self.feat1_dim, self.feat2_dim),
            nn.BatchNorm1d(self.feat2_dim),
            nn.ReLU(inplace=True))
            self.fc2 = nn.Linear(self.feat2_dim, num_classes)
            #self.fc2 = nn.Linear(self.feat1_dim, num_classes)
        else:
            self.fc1 = nn.Linear(self.feat1_dim, num_classes)
            
        mod = list(model_ft.children())
        mod.pop()
        self.features = nn.Sequential(*mod)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), self.feat1_dim)
        y1 = self.fc1(x)
        if self.num_neurons:
            y2 = self.fc2(y1)
            #y2 = self.fc2(x)
            return x, y1, y2
        else:
            return x, None, y1
        
        

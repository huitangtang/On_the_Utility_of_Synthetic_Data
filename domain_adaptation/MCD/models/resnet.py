from torchvision import models
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function, Variable


class ResBase(nn.Module):
    def __init__(self, option='resnet50', feat_proj=False, pret=False, num_classes=1000):
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
            
        mod = list(model_ft.children())
        mod.pop()
        self.features = nn.Sequential(*mod)
        if feat_proj:
            self.proj_fc = nn.Linear(self.dim, 256)
            self.dim = 256
        self.feat_proj = feat_proj

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), self.dim)
        if self.feat_proj:
            x = self.proj_fc(x)
            
        return x


class ResClassifier(nn.Module):
    def __init__(self, num_classes=12, num_layer=3, num_unit=2048, prob=0.5, middle=1000):
        super(ResClassifier, self).__init__()

        if num_layer == 1:
            self.classifier = nn.Linear(num_unit, num_classes)
        else:
            layers = []
            # currently 10000 units
            layers.append(nn.Dropout(p=prob))
            layers.append(nn.Linear(num_unit,middle))
            layers.append(nn.BatchNorm1d(middle,affine=True))
            layers.append(nn.ReLU(inplace=True))

            for i in range(num_layer-2):
                layers.append(nn.Dropout(p=prob))
                layers.append(nn.Linear(middle,middle))
                layers.append(nn.BatchNorm1d(middle,affine=True))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(middle,num_classes))
            self.classifier = nn.Sequential(*layers)
        
        self.apply(weights_init)


    def forward(self, x):
        x = self.classifier(x)
        return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)
        
        
        
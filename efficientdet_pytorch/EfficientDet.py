import torch
from efficientnet_pytorch import EfficientNet
from torch import nn
from torch.nn import functional as F

from .BiFPN import BiFPN


class EfficientDet(nn.Module):
    """
    Implementation of EfficientDet
    """

    def __init__(self, compound_coefficient=0):
        super().__init__()
        self.compound_coefficient = compound_coefficient
        self.backbone = self.create_backbone(self.compound_coefficient)
        self.BiFPN = self.create_BiFPN(self.compound_coefficient)
        self.prediction_net = self.create_prediction_net(self.compound_coefficient)

    def create_backbone(self, compound_coefficient):
        #todo : load diffirent EfficientNet based on compound_coefficient
        return EfficientNet()
    
    def create_BiFPN(self, compound_coefficient):
        return BiFPN(compound_coefficient)

    def forward(self, input):
        x = self.backbone.forward(input)
        x = self.BiFPN.forward(x)
        out = self.prediction_net.forward(x)
        return out

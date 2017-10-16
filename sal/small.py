from .datasets import cifar_dataset
from utils.pytorch_fixes import *
from utils.pytorch_trainer import *
import torch
from torch.autograd import Variable
from torch.nn import functional as F
import torch.nn as nn
from torch.nn import Module
import numpy as np
import pycat




class SimpleClassifier(EasyModule):
    '''Relatively fast and works well for 32x32 and 64x64 images. Achieves ~89% on CIFAR10 with base=24.
       Returns all the intermediate features so it can serve as a feature extractor or a base for the U-Net decoder.'''
    def __init__(self, base_channels=24, num_classes=10, image_channels=3):
        super(SimpleClassifier, self).__init__()

        BASE1 = base_channels
        BASE2 = BASE1 * 2
        BASE3 = BASE1 * 4
        BASE4 = BASE1 * 8
        self.scale0 = SimpleCNNBlock(image_channels, BASE1, layers=3)
        self.scale1 = SimpleCNNBlock(BASE1, BASE2, stride=2, layers=3)
        self.scale2 = SimpleCNNBlock(BASE2, BASE3, stride=2, layers=3)
        self.scale3 = SimpleCNNBlock(BASE3, BASE4, stride=2, layers=3)
        self.scaleX = GlobalAvgPool()
        self.scaleC = nn.Linear(BASE4, num_classes)

    def forward(self, _images):
        s0 = self.scale0(_images)
        s1 = self.scale1(s0)
        s2 = self.scale2(s1)
        s3 = self.scale3(s2)
        sX = self.scaleX(s3)
        sC = self.scaleC(sX)
        return s0, s1, s2, s3, sX, sC

    @staticmethod
    def out_to_logits(out):
        s0, s1, s2, s3, sX, sC = out
        return sC







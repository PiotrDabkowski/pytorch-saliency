from sal.datasets import cifar_dataset
from sal.utils.pytorch_fixes import *
from sal.utils.pytorch_trainer import *
import torch
from torch.autograd import Variable
from torch.nn import functional as F
import torch.nn as nn
from torch.nn import Module
import numpy as np
import pycat
from sal.saliency import SimpleClassifier, Saliency
from torchvision.models.resnet import resnet50
from PIL import Image

im = Variable(torch.Tensor(
    np.expand_dims(np.transpose(np.array(Image.open(os.path.join(os.path.dirname(__file__), 'sal/utils/test.jpg'))), (2, 0, 1)),
                   0) / 255.), requires_grad=False)
black_box_model = resnet50(pretrained=True)
black_box_model.train(False)

@ev_batch_to_images_labels
def ev(_images, _labels):
    black_box_model(_images)


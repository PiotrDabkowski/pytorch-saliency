from .datasets import cifar_dataset
from utils.pytorch_fixes import *
from utils.pytorch_trainer import *
import torch
from torch.autograd import Variable
from torch.nn import functional as F
import torch.nn as nn
from torch.nn import Module
from small import SimpleClassifier
import numpy as np
import pycat

class Saliency(Module):
    def __init__(self, encoder, encoder_scales, encoder_base, upsampler_scales, upsampler_base):
        super(Saliency, self).__init__()
        assert upsampler_scales <= encoder_scales

        self.encoder = encoder  # decoder must return at least scale0 to scaleN where N is num_scales
        self.upsampler_scales = upsampler_scales
        self.encoder_scales = encoder_scales

        # now build the decoder for the specified number of scales
        # start with the top scale
        down = self.encoder_scales
        for up in reversed(xrange(self.upsampler_scales)):
            upsampler_chans = upsampler_base * 2**(up+1)
            encoder_chans = encoder_base * 2**down
            self.add_module('up%d'%up,
                            UNetUpsampler(
                                in_channels=upsampler_chans if down!=encoder_scales else encoder_chans,
                                passthrough_channels=encoder_chans/2,
                                out_channels=upsampler_chans/2,
                                follow_up_residual_blocks=1,
                            ))
            down -= 1

        self.to_saliency_chans = nn.Conv2d(upsampler_base, 2, 1)


    def forward(self, _images):
        # forward pass through the encoder
        out = self.encoder(_images)

        down = self.encoder_scales
        main_flow = out[down]
        for up in reversed(xrange(self.upsampler_scales)):
            assert down > 0
            main_flow = self._modules['up%d'%up](main_flow, out[down-1])
            down -= 1
            print main_flow.size()

        # now get the final saliency map (the reslution of the map = resolution_of_the_image / (2**(encoder_scales-upsampler_scales)))
        saliency_chans = self.to_saliency_chans(main_flow)


        a = torch.abs(saliency_chans[:,0,:,:])
        b = torch.abs(saliency_chans[:,1,:,:])
        return torch.unsqueeze(a/(a+b), dim=1)




class SaliencyLoss(Module):
    def __init__(self):
        super(SaliencyLoss, self).__init__()
        pass





import torch
import torch.nn as nn
import torch.optim as torch_optim
import numpy as np
from torch.autograd import Variable
from sal.saliency_model import SaliencyLoss, get_black_box_fn
from sal.utils import pt_store
from saliency_eval import to_batch_variable, load_image_as_variable
PT = pt_store.PTStore()
import pycat
import os


class IterativeSaliency:
    def __init__(self, cuda=True, black_box_fn=None, mask_resolution=32, num_classes=1000, default_iterations=200):
        if black_box_fn is None:
            self.black_box_fn = get_black_box_fn(cuda=cuda)  # defaults to ResNet-50 on ImageNet
        self.default_iterations = default_iterations
        self.mask_resolution = mask_resolution
        self.num_classes = num_classes
        self.saliency_loss_calc = SaliencyLoss(self.black_box_fn, area_loss_coef=11, smoothness_loss_coef=0.5, preserver_loss_coef=0.2)
        self.cuda = cuda

    def get_saliency_maps(self, _images, _targets, iterations=None, show=False):
        ''' returns saliency maps.
         Params
         _images - input images of shape (C, H, W) or (N, C, H, W) if in batch. Can be either a numpy array, a Tensor or a Variable
         _targets - class ids to be masked. Can be either an int or an array with N integers. Again can be either a numpy array, a Tensor or a Variable

         returns a Variable of shape (N, 1, H, W) with one saliency maps for each input image.
         '''
        _images, _targets = to_batch_variable(_images, 4, self.cuda).float(), to_batch_variable(_targets, 1, self.cuda).long()


        if iterations is None:
            iterations = self.default_iterations

        if self.cuda:
            _mask = nn.Parameter(torch.Tensor(_images.size(0), 2, self.mask_resolution, self.mask_resolution).fill_(0.5).cuda())
        else:
            _mask = nn.Parameter(torch.Tensor(_images.size(0), 2, self.mask_resolution, self.mask_resolution).fill_(0.5))
        optim = torch_optim.SGD([_mask], 0.1, 0.9, nesterov=True)
        #optim = torch_optim.Adam([_mask], 0.2)

        for iteration in xrange(iterations):
            #_mask.data.clamp_(0., 1.)
            optim.zero_grad()

            a = torch.abs(_mask[:, 0, :, :])
            b = torch.abs(_mask[:, 1, :, :])
            _mask_ = torch.unsqueeze(a / (a + b+0.001), dim=1)

            total_loss = self.saliency_loss_calc.get_loss(_images, _targets, _mask_, pt_store=PT)

            total_loss.backward()

            optim.step()
            if show:
                pycat.show(PT['masks'][0]*255, auto_normalize=False)
                pycat.show(PT['preserved'][0])
        return PT.masks



def test():
    from PIL import Image
    import time
    print 'We will optimize for the white terrier class...'
    time.sleep(3)

    i = IterativeSaliency()

    ims = load_image_as_variable(os.path.join(os.path.dirname(__file__), 'sal/utils/test.jpg'))
    i.get_saliency_maps(ims, [203], show=True)


if __name__ == '__main__':
    test()

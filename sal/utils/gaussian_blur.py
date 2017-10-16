import numpy as np
from torch.nn import Module, BatchNorm2d
from torch.nn.functional import conv2d
import torch
from torch.autograd import Variable

def _gaussian_kernel(kernel_size, sigma, chans):
    assert kernel_size % 2, 'Kernel size of the gaussian blur must be odd!'
    x = np.expand_dims(np.array(range(-kernel_size/2, -kernel_size/2+kernel_size, 1)), 0)
    vals = np.exp(-np.square(x)/(2.*sigma**2))
    kernel_raw = np.matmul(vals.T, vals)
    kernel = np.reshape(kernel_raw / np.sum(kernel_raw), (1, 1, kernel_size, kernel_size))
    repeated =  np.zeros((chans, 1, kernel_size, kernel_size), dtype=np.float32) + kernel
    return repeated


class GaussianBlur(Module):
    def __init__(self, kernel_size=55, sigma=11, output_channels=3):
        '''remember that kernel size should be larger than ~4*sigma'''
        super(GaussianBlur, self).__init__()
        self.padding = kernel_size/2
        self.kernel = torch.Tensor(_gaussian_kernel(kernel_size=kernel_size, sigma=sigma, chans=output_channels))

    def forward(self, x):
        return conv2d(x, Variable(self.kernel, requires_grad=False), groups=x.size(1), padding=self.padding)


def test():
    from PIL import Image
    import os, pycat
    im = Variable(torch.Tensor(np.expand_dims(np.transpose(np.array(Image.open(os.path.join(os.path.dirname(__file__), 'test.jpg'))), (2, 0, 1)), 0)/255.), requires_grad=True)
    g = GaussianBlur()(im)
    print 'Original'
    pycat.show(im[0].data.numpy())
    print 'Blurred version'
    pycat.show(g[0].data.numpy())
    print 'Image gradient over blurred sum (should be white in the middle + turning darker at the edges)'
    l = torch.sum(g)
    l.backward()
    assert np.mean(im.grad[0].data.numpy()) > 0.9
    pycat.show(im.grad[0].data.numpy())



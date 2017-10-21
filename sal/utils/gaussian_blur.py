import numpy as np
from torch.nn import Module
from torch.nn.functional import conv2d
import torch
from torch.autograd import Variable


def _gaussian_kernels(kernel_size, sigma, chans):
    assert kernel_size % 2, 'Kernel size of the gaussian blur must be odd!'
    x = np.expand_dims(np.array(range(-kernel_size/2, -kernel_size/2+kernel_size, 1)), 0)
    vals = np.exp(-np.square(x)/(2.*sigma**2))
    _kernel = np.reshape(vals / np.sum(vals), (1, 1, kernel_size, 1))
    kernel =  np.zeros((chans, 1, kernel_size, 1), dtype=np.float32) + _kernel
    return kernel, np.transpose(kernel, [0, 1, 3, 2])

def gaussian_blur(_images, kernel_size=55, sigma=11):
    ''' Very fast, linear time gaussian blur, using separable convolution. Operates on batch of images [N, C, H, W].
    Returns blurred images of the same size. Kernel size must be odd.
    Increasing kernel size over 4*simga yields little improvement in quality. So kernel size = 4*sigma is a good choice.'''
    kernel_a, kernel_b = _gaussian_kernels(kernel_size=kernel_size, sigma=sigma, chans=_images.size(1))
    kernel_a = torch.Tensor(kernel_a)
    kernel_b = torch.Tensor(kernel_b)
    if _images.is_cuda:
        kernel_a = kernel_a.cuda()
        kernel_b = kernel_b.cuda()
    _rows = conv2d(_images, Variable(kernel_a, requires_grad=False), groups=_images.size(1), padding=(kernel_size / 2, 0))
    return conv2d(_rows, Variable(kernel_b, requires_grad=False), groups=_images.size(1), padding=(0, kernel_size / 2))


def test():
    from PIL import Image
    import os, pycat
    im = Variable(torch.Tensor(np.expand_dims(np.transpose(np.array(Image.open(os.path.join(os.path.dirname(__file__), 'test.jpg'))), (2, 0, 1)), 0)/255.), requires_grad=True)
    g = gaussian_blur(im)
    print 'Original'
    pycat.show(im[0].data.numpy())
    print 'Blurred version'
    pycat.show(g[0].data.numpy())
    print 'Image gradient over blurred sum (should be white in the middle + turning darker at the edges)'
    l = torch.sum(g)
    l.backward()
    gr = im.grad[0].data.numpy()
    assert np.mean(gr) > 0.9 and np.mean(np.flip(gr, 1)-gr) < 1e-6 and  np.mean(np.flip(gr, 2)-gr) < 1e-6
    pycat.show(gr)



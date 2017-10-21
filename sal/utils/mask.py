import torch
from torch.autograd import Variable
import gaussian_blur

def calc_smoothness_loss(mask, power=2, border_penalty=0.3):
    ''' For a given image this loss should be more or less invariant to image resize when using power=2...
        let L be the length of a side
        EdgesLength ~ L
        EdgesSharpness ~ 1/L, easy to see if you imagine just a single vertical edge in the whole image'''
    x_loss = torch.sum((torch.abs(mask[:,:,1:,:] - mask[:,:,:-1,:]))**power)
    y_loss = torch.sum((torch.abs(mask[:,:,:,1:] - mask[:,:,:,:-1]))**power)
    if border_penalty>0:
        border = float(border_penalty)*torch.sum(mask[:,:,-1,:]**power + mask[:,:,0,:]**power + mask[:,:,:,-1]**power + mask[:,:,:,0]**power)
    else:
        border = 0.
    return (x_loss + y_loss + border) / float(power * mask.size(0))  # watch out, normalised by the batch size!



def calc_area_loss(mask, power=1.):
    if power != 1:
        mask = (mask+0.0005)**power # prevent nan (derivative of sqrt at 0 is inf)
    return torch.mean(mask)


def tensor_like(x):
    if x.is_cuda:
        return torch.Tensor(*x.size()).cuda()
    else:
        return torch.Tensor(*x.size())

def apply_mask(images, mask, noise=True, random_colors=True, blurred_version_prob=0.5, noise_std=0.11,
               color_range=0.66, blur_kernel_size=55, blur_sigma=11,
               bypass=0., boolean=False, preserved_imgs_noise_std=0.03):
    images = images.clone()
    cuda = images.is_cuda

    if boolean:
        # remember its just for validation!
        return (mask > 0.5).float() *images

    assert 0. <= bypass < 0.9
    n, c, _, _ = images.size()
    if preserved_imgs_noise_std > 0:
        images = images + Variable(tensor_like(images).normal_(std=preserved_imgs_noise_std), requires_grad=False)
    if bypass > 0:
        mask = (1.-bypass)*mask + bypass
    if noise and noise_std:
        alt = tensor_like(images).normal_(std=noise_std)
    else:
        alt = tensor_like(images).zero_()
    if random_colors:
        if cuda:
            alt += torch.Tensor(n, c, 1, 1).cuda().uniform_(-color_range/2., color_range/2.)
        else:
            alt += torch.Tensor(n, c, 1, 1).uniform_(-color_range/2., color_range/2.)

    alt = Variable(alt, requires_grad=False)

    if blurred_version_prob > 0.: # <- it can be a scalar between 0 and 1
        cand = gaussian_blur.gaussian_blur(images, kernel_size=blur_kernel_size, sigma=blur_sigma)
        if cuda:
            when = Variable((torch.Tensor(n, 1, 1, 1).cuda().uniform_(0., 1.) < blurred_version_prob).float(), requires_grad=False)
        else:
            when = Variable((torch.Tensor(n, 1, 1, 1).uniform_(0., 1.) < blurred_version_prob).float(), requires_grad=False)
        alt = alt*(1.-when) + cand*when

    return (mask*images.detach()) + (1. - mask)*alt.detach()




def test():
    from PIL import Image
    import numpy as np
    import os, pycat
    im = Variable(torch.Tensor(np.expand_dims(np.transpose(np.array(Image.open(os.path.join(os.path.dirname(__file__), 'test.jpg'))), (2, 0, 1)), 0)/255.*2-1.), requires_grad=False)
    print 'Original'
    pycat.show(im[0].data.numpy())
    print
    for pres in [1., 0.5, 0.1]:
        print 'Mask strength =', pres
        for e in xrange(5):
            m = Variable(torch.Tensor(1, 3, im.size(2), im.size(3)).fill_(pres), requires_grad=True)
            res = apply_mask(im, m)
            pycat.show(res[0].data.numpy())
    s = torch.sum(res)
    s.backward()
    print torch.sum(m.grad)



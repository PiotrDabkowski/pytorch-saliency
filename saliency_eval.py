from sal.saliency_model import SaliencyModel
from sal.utils.resnet_encoder import resnet50encoder
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import os


def get_pretrained_saliency_fn(cuda=True, return_classification_logits=False):
    ''' returns a saliency function that takes images and class selectors as inputs. If cuda=True then places the model on a GPU.
    You can also specify model_confidence - smaller values (~0) will show any object in the image that even slightly resembles the specified class
    while higher values (~5) will show only the most salient parts.
    Params of the saliency function:
    images - input images of shape (C, H, W) or (N, C, H, W) if in batch. Can be either a numpy array, a Tensor or a Variable
    selectors - class ids to be masked. Can be either an int or an array with N integers. Again can be either a numpy array, a Tensor or a Variable
    model_confidence - a float, 6 by default, you may want to decrease this value to obtain more complete saliency maps.

    returns a Variable of shape (N, 1, H, W) with one saliency maps for each input image.
    '''
    saliency = SaliencyModel(resnet50encoder(pretrained=True), 5, 64, 3, 64, fix_encoder=True, use_simple_activation=False, allow_selector=True)
    saliency.minimialistic_restore(os.path.join(os.path.dirname(__file__), 'minsaliency'))
    saliency.train(False)
    if cuda:
        saliency = saliency.cuda()
    def fn(images, selectors, model_confidence=6):
        _images, _selectors = to_batch_variable(images, 4, cuda).float(), to_batch_variable(selectors, 1, cuda).long()
        masks, _, cls_logits = saliency(_images*2, _selectors, model_confidence=model_confidence)
        sal_map = F.upsample(masks, (_images.size(2), _images.size(3)), mode='bilinear')
        if not return_classification_logits:
            return sal_map
        return sal_map, cls_logits
    return fn


def to_batch_variable(x, required_rank, cuda=False):
    if isinstance(x, Variable):
        if cuda and not x.is_cuda:
            return x.cuda()
        if not cuda and x.is_cuda:
            return x.cpu()
        else:
            return x
    if isinstance(x, (float, long, int)):
        assert required_rank == 1
        return to_batch_variable(np.array([x]), required_rank, cuda)
    if isinstance(x, (list, tuple)):
        return to_batch_variable(np.array(x), required_rank, cuda)
    if isinstance(x, np.ndarray):
        c = len(x.shape)
        if c==required_rank:
            return to_batch_variable(torch.from_numpy(x), required_rank, cuda)
        elif c+1==required_rank:
            return to_batch_variable(torch.unsqueeze(torch.from_numpy(x), dim=0), required_rank, cuda)
        else:
            raise ValueError()
    if cuda:
        return Variable(x).cuda()
    else:
        return Variable(x)


def load_image_as_variable(path, cuda=False):
    ''' Loads an image and returns a pytorch Variable of shape (1, 3, H, W). Image will be normalised between -1 and 1.'''
    return to_batch_variable(np.expand_dims(np.transpose(np.array(Image.open(path)), (2, 0, 1)), 0)/255.*2-1., 4, cuda=cuda).float()


def test(cuda=True):
    f = get_pretrained_saliency_fn(cuda=cuda)
    import os
    import pycat
    import time
    from sal.utils.mask import apply_mask
    # simply load an image
    ims = load_image_as_variable(os.path.join(os.path.dirname(__file__), 'sal/utils/test2.jpg'), cuda=cuda)

    zebra_mask = f(ims, 340)  # 340 is a zebra
    elefant_mask = f(ims, [386])  # 386 is an elefant (check sal/datasets/imagenet_synset.py for more)


    print 'You should see a zebra'
    pycat.show(apply_mask(ims, zebra_mask, boolean=False).cpu()[0].data.numpy()*128+128, auto_normalize=False)
    print 'You should see an elefant'
    pycat.show(apply_mask(ims, elefant_mask, boolean=False).cpu()[0].data.numpy()*128+128, auto_normalize=False)

    print 'Testing speed with CUDA_ENABLED =', cuda
    print 'Please wait...'
    t = time.time()
    for e in xrange(20 if cuda else 2):
        f(np.random.randn(32, 3, 224, 224), np.random.uniform(0, 100, size=(32,)).astype(np.int), 6)
    print 'Images per second:', 32. * (20 if cuda else 2) / (time.time()-t)
    print 'You should expect ~200 images per second on a GPU (Titan XP) and 2.5 images per second on a CPU. '


if __name__ == '__main__':
    test()
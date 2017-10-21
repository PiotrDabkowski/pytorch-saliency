__all__ = ['RandomSizedCrop2', 'STD_NORMALIZE', 'ShapeLog', 'AssertSize', 'GlobalAvgPool', 'SimpleCNNBlock', 'SimpleLinearBlock',
           'SimpleExtractor', 'SimpleGenerator', 'DiscreteNeuron', 'chain', 'EasyModule', 'UNetUpsampler',
           'SimpleUpsamplerSubpixel', 'CustomModule', 'BottleneckBlock', 'losses', 'F', 'torch_optim', 'Variable',
           'one_hot', 'cw_loss', 'adapt_to_image_domain', 'nn', 'MultiModulator']



from torchvision.transforms import *
from torch.nn import *
import torch.nn as nn
import torch.nn.modules.loss as losses
from torch.nn import functional as F
from itertools import chain
import torch.optim as torch_optim
from torch.autograd import Variable
import signal
import sys

INFO_TEMPLATE = '\033[38;5;2mINFO: %s\033[0m\n'
WARN_TEMPLATE = '\033[38;5;1mWARNING: %s\033[0m\n'


def signal_handler(signal, frame):
        print('Finishing...')
        sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)



STD_NORMALIZE = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


class DiscreteNeuron(Module):
    def forward(self, x):
        return discrete_neuron_func()(x)

class discrete_neuron_func(torch.autograd.Function):
    def forward(self, x):
        self.save_for_backward(x)
        x = x.clone()
        x[x>0] = 1.
        x[x<0] = -1.
        return x

    def backward(self, grad_out):
        x, = self.saved_tensors
        grad_input = grad_out.clone()
        this_grad = torch.exp(-(x**2)/2.) / (3.14/2.)**0.5
        return this_grad*grad_input

class RandomSizedCrop2(object):
    """Crop the given PIL.Image to random size and aspect ratio.

    A crop of random size of (0.08 to 1.0) of the original size and a random
    aspect ratio of 3/4 to 4/3 of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: size of the smaller edge
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, min_area=0.3, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.min_area = min_area

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(self.min_area, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))

                return img.resize((self.size, self.size), self.interpolation)

        # Fallback
        scale = Scale(self.size, interpolation=self.interpolation)
        crop = CenterCrop(self.size)
        return crop(scale(img))


class AssertSize(Module):
    def __init__(self, expected_dim=None):
        super(AssertSize, self).__init__()
        self.expected_dim = expected_dim

    def forward(self, x):
        if self.expected_dim is not None:
            assert self.expected_dim==x.size(2), 'expected %d got %d' % (self.expected_dim, x.size(2))
        return x


class ShapeLog(Module):
    def forward(self, x):
        print x.size()
        return x

class PixelShuffleBlock(Module):
    def forward(self, x):
        return F.pixel_shuffle(x, 2)


class GlobalAvgPool(Module):
    def forward(self, x):
        x = F.avg_pool2d(x, x.size(2), stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True)
        return x.view(x.size(0), -1)



def SimpleCNNBlock(in_channels, out_channels,
                 kernel_size=3, layers=1, stride=1,
                 follow_with_bn=True, activation_fn=lambda: ReLU(True), affine=True):
        assert layers > 0 and kernel_size%2 and stride>0
        current_channels = in_channels
        _modules = []
        for layer in range(layers):
            _modules.append(Conv2d(current_channels, out_channels, kernel_size, stride=stride if layer==0 else 1, padding=kernel_size/2, bias=not follow_with_bn))
            current_channels = out_channels
            if follow_with_bn:
                _modules.append(BatchNorm2d(current_channels, affine=affine))
            if activation_fn is not None:
                _modules.append(activation_fn())
        return Sequential(*_modules)


def ReducedCNNBlock(in_channels, out_channels,
                   kernel_size=3, layers=1, stride=1, activation_fn=lambda: ReLU(True), reducer_family='all'):
    assert layers > 0 and kernel_size % 2 and stride > 0
    current_channels = in_channels
    follow_with_bn = True
    _modules = []
    for layer in range(layers):
        _modules.append(Conv2d(current_channels, out_channels, kernel_size, stride=stride if layer == layers - 1 else 1,
                               padding=kernel_size / 2, bias=not follow_with_bn))
        current_channels = out_channels
        if follow_with_bn:
            _modules.append(Reducer4D(current_channels, family=reducer_family))
        if activation_fn is not None:
            _modules.append(activation_fn())
    return Sequential(*_modules)


def SimpleLinearBlock(in_channels, out_channels, layers=1, follow_with_bn=True, activation_fn=lambda: ReLU(inplace=False), affine=True):
    assert layers > 0
    current_channels = in_channels
    _modules = []
    for layer in range(layers):
        _modules.append(Linear(current_channels, out_channels, bias=not follow_with_bn))
        current_channels = out_channels
        if follow_with_bn:
            _modules.append(BatchNorm1d(current_channels, affine=affine))
        if activation_fn is not None:
            _modules.append(activation_fn())
    return Sequential(*_modules)



def ReducedExtractor(base_channels, downsampling_blocks, extra_modules=(), activation_fn=lambda: torch.nn.ReLU(inplace=False)):
    # final_dimension is an extra layer of protection so that we have the dimensions right
    current_channels = 3
    _modules = [BatchNorm2d(current_channels)]
    for layers in downsampling_blocks:
        if layers-1>0:
            _modules.append(ReducedCNNBlock(current_channels, base_channels, layers=layers-1, activation_fn=activation_fn))
            current_channels = base_channels
        base_channels *= 2
        _modules.append(ReducedCNNBlock(current_channels, base_channels, stride=2, activation_fn=activation_fn))
        current_channels = base_channels
    _modules.extend(extra_modules)
    return Sequential(*_modules)



def SimpleExtractor(base_channels, downsampling_blocks, extra_modules=(), affine=True, activation_fn=lambda: torch.nn.ReLU(inplace=False)):
    # final_dimension is an extra layer of protection so that we have the dimensions right
    current_channels = 3
    _modules = [BatchNorm2d(current_channels)]
    for layers in downsampling_blocks:
        if layers-1>0:
            _modules.append(SimpleCNNBlock(current_channels, base_channels, layers=layers-1, activation_fn=activation_fn))
            current_channels = base_channels
        base_channels *= 2
        _modules.append(SimpleCNNBlock(current_channels, base_channels, stride=2, affine=affine, activation_fn=activation_fn))
        current_channels = base_channels
    _modules.extend(extra_modules)
    return Sequential(*_modules)


def SimpleUpsamplerSubpixel(in_channels, out_channels, kernel_size=3, activation_fn=lambda: torch.nn.ReLU(inplace=False), follow_with_bn=True):
    _modules = [
        SimpleCNNBlock(in_channels, out_channels * 4, kernel_size=kernel_size, follow_with_bn=follow_with_bn),
        PixelShuffleBlock(),
        activation_fn(),
    ]
    return Sequential(*_modules)


class UNetUpsampler(Module):
    def __init__(self, in_channels, out_channels, passthrough_channels, follow_up_residual_blocks=1, upsampler_block=SimpleUpsamplerSubpixel,
                 upsampler_kernel_size=3, activation_fn=lambda: torch.nn.ReLU(inplace=False)):
        super(UNetUpsampler, self).__init__()
        assert follow_up_residual_blocks >= 1, 'You must follow up with residuals when using unet!'
        assert passthrough_channels >= 1, 'You must use passthrough with unet'
        self.upsampler = upsampler_block(in_channels=in_channels,
                                         out_channels=out_channels, kernel_size=upsampler_kernel_size, activation_fn=activation_fn)

        self.follow_up = BottleneckBlock(out_channels+passthrough_channels, out_channels, layers=follow_up_residual_blocks, activation_fn=activation_fn)

    def forward(self, inp, passthrough):
        upsampled = self.upsampler(inp)
        upsampled = torch.cat((upsampled, passthrough), 1)
        return self.follow_up(upsampled)


class CustomModule(Module):
    def __init__(self, py_func):
        super(CustomModule, self).__init__()
        self.py_func = py_func

    def forward(self, inp):
        return self.py_func(inp)


class MultiModulator(Module):
    def __init__(self, embedding_size, num_classes, modulator_sizes):
        super(MultiModulator, self).__init__()
        self.emb = Embedding(num_classes, embedding_size)
        self.num_modulators = len(modulator_sizes)
        for i, m in enumerate(modulator_sizes):
            self.add_module('m%d'%i, Linear(embedding_size, m))

    def forward(self, selectors):
        ''' class selector must be of shape (BS,)  Returns (BS, MODULATOR_SIZE) for each modulator.'''
        em = torch.squeeze(self.emb(selectors.view(-1, 1)), 1)
        res = []
        for i in xrange(self.num_modulators):
            res.append( self._modules['m%d'%i](em) )
        return tuple(res)


import os
class EasyModule(Module):
    def save(self, save_dir, step=1):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save(self.state_dict(), os.path.join(save_dir, 'model-%d.ckpt'%step))


    def restore(self, save_dir, step=1):
        p = os.path.join(save_dir, 'model-%d.ckpt' % step)
        if not os.path.exists(p):
            print WARN_TEMPLATE % ('Could not find any checkpoint at %s, skipping restore' % p)
            return
        self.load_state_dict(torch.load(p))
Module.save = EasyModule.save.__func__
Module.restore = EasyModule.restore.__func__

def one_hot(labels, depth):
    return Variable(torch.zeros(labels.size(0), depth).cuda().scatter_(1, labels.long().view(-1, 1).data, 1))


def cw_loss(logits, one_hot_labels, targeted=True, t_conf=2, nt_conf=5):
    ''' computes the advantage of the selected label over other highest prob guess.
        In case of the targeted it tries to maximise this advantage to reach desired confidence.
        For example confidence of 3 would mean that the desired label is e^3 (about 20) times more probable than the second top guess.
        In case of non targeted optimisation the case is opposite and we try to minimise this advantage - the probability of the label is
        20 times smaller than the probability of the top guess.

        So for targeted optim a small confidence should be enough (about 2) and for non targeted about 5-6 would work better (assuming 1000 classes so log(no_idea)=6.9)
    '''
    this = torch.sum(logits*one_hot_labels, 1)
    other_best, _ = torch.max(logits*(1.-one_hot_labels) - 12111*one_hot_labels, 1)   # subtracting 12111 from selected labels to make sure that they dont end up a maximum
    t = F.relu(other_best - this + t_conf)
    nt = F.relu(this - other_best + nt_conf)
    if isinstance(targeted, (bool, int)):
        return torch.mean(t) if targeted else torch.mean(nt)
    else:  # must be a byte tensor of zeros and ones

        return torch.mean(t*(targeted>0).float() + nt*(targeted==0).float())

def adapt_to_image_domain(images_plus_minus_one, desired_domain):
    if desired_domain == (-1., 1.):
        return images_plus_minus_one
    return images_plus_minus_one * (desired_domain[1] - desired_domain[0]) / 2. + (desired_domain[0] + desired_domain[1]) / 2.

class Bottleneck(Module):
    def __init__(self, in_channels, out_channels, stride=1, bottleneck_ratio=4, activation_fn=lambda: torch.nn.ReLU(inplace=False)):
        super(Bottleneck, self).__init__()
        bottleneck_channels = out_channels/bottleneck_ratio
        self.conv1 = Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(bottleneck_channels)
        self.conv2 = Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(bottleneck_channels)
        self.conv3 = Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(out_channels)
        self.activation_fn = activation_fn()

        if stride != 1 or in_channels != out_channels :
            self.residual_transformer = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True)
        else:
            self.residual_transformer = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation_fn(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation_fn(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.residual_transformer is not None:
            residual = self.residual_transformer(residual)
        out += residual

        out = self.activation_fn(out)
        return out

def BottleneckBlock(in_channels, out_channels, stride=1, layers=1, activation_fn=lambda: torch.nn.ReLU(inplace=False)):
    assert layers > 0 and stride > 0
    current_channels = in_channels
    _modules = []
    for layer in range(layers):
        _modules.append(Bottleneck(current_channels, out_channels, stride=stride if layer==0 else 1, activation_fn=activation_fn))
        current_channels = out_channels
    return Sequential(*_modules) if len(_modules)>1 else _modules[0]

def SimpleGenerator(in_channels, base_channels, upsampling_blocks=lambda: torch.nn.ReLU(inplace=False)):
    _modules = []
    current_channels = in_channels
    base_channels = base_channels * 2**len(upsampling_blocks)
    for layers in upsampling_blocks:
        if layers-1>0:
            _modules.append(SimpleCNNBlock(current_channels, base_channels, layers=layers-1))
            current_channels = base_channels
        _modules.append(SimpleCNNBlock(current_channels, base_channels*2))
        _modules.append(PixelShuffleBlock())
        base_channels /= 2
        current_channels = base_channels
    _modules.append(Conv2d(base_channels, 3, 1))
    _modules.append(Tanh())
    return Sequential(*_modules)


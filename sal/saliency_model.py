from utils.pytorch_fixes import *
from torch.nn import functional as F
import torch.nn as nn
from torch.nn import Module
from sal.utils.mask import *
from torchvision.models.resnet import resnet50
import os

def get_black_box_fn(model_zoo_model=resnet50, image_domain=(-2., 2.)):
    ''' You can try any model from the pytorch model zoo (torchvision.models)
        eg. VGG, inception, mobilenet, alexnet...
    '''
    black_box_model = model_zoo_model(pretrained=True)

    black_box_model.train(False)
    black_box_model = torch.nn.DataParallel(black_box_model).cuda()

    def black_box_fn(_images):
        return black_box_model(adapt_to_image_domain(_images, image_domain))
    return black_box_fn



class SaliencyModel(Module):
    def __init__(self, encoder, encoder_scales, encoder_base, upsampler_scales, upsampler_base, fix_encoder=True,
                 use_simple_activation=False, allow_selector=False, num_classes=1000):
        super(SaliencyModel, self).__init__()
        assert upsampler_scales <= encoder_scales

        self.encoder = encoder  # decoder must return at least scale0 to scaleN where N is num_scales
        self.upsampler_scales = upsampler_scales
        self.encoder_scales = encoder_scales
        self.fix_encoder = fix_encoder
        self.use_simple_activation = use_simple_activation

        # now build the decoder for the specified number of scales
        # start with the top scale
        down = self.encoder_scales
        modulator_sizes = []
        for up in reversed(xrange(self.upsampler_scales)):
            upsampler_chans = upsampler_base * 2**(up+1)
            encoder_chans = encoder_base * 2**down
            inc = upsampler_chans if down!=encoder_scales else encoder_chans
            modulator_sizes.append(inc)
            self.add_module('up%d'%up,
                            UNetUpsampler(
                                in_channels=inc,
                                passthrough_channels=encoder_chans/2,
                                out_channels=upsampler_chans/2,
                                follow_up_residual_blocks=1,
                                activation_fn=lambda: nn.ReLU(),
                            ))
            down -= 1

        self.to_saliency_chans = nn.Conv2d(upsampler_base, 2, 1)

        self.allow_selector = allow_selector

        if self.allow_selector:
            s = encoder_base*2**encoder_scales
            self.selector_module = nn.Embedding(num_classes, s)
            self.selector_module.weight.data.normal_(0, 1./s**0.5)


    def minimialistic_restore(self, save_dir):
        assert self.fix_encoder, 'You should not use this function if you are not using a pre-trained encoder like resnet'

        p = os.path.join(save_dir, 'model-%d.ckpt' % 1)
        if not os.path.exists(p):
            print 'Could not find any checkpoint at %s, skipping restore' % p
            return
        for name, data in torch.load(p, map_location=lambda storage, loc: storage).items():
            self._modules[name].load_state_dict(data)

    def minimalistic_save(self, save_dir):
        assert self.fix_encoder, 'You should not use this function if you are not using a pre-trained encoder like resnet'
        data = {}
        for name, module in self._modules.items():
            if module is self.encoder:  # we do not want to restore the encoder as it should have its own restore function
                continue
            data[name] = module.state_dict()
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save(data, os.path.join(save_dir, 'model-%d.ckpt' % 1))


    def get_trainable_parameters(self):
        all_params = self.parameters()
        if not self.fix_encoder: return set(all_params)
        unwanted = self.encoder.parameters()
        return set(all_params) - set(unwanted) - (set(self.selector_module.parameters()) if self.allow_selector else set([]))

    def forward(self, _images, _selectors=None, pt_store=None, model_confidence=0.):
        # forward pass through the encoder
        out = self.encoder(_images)
        if self.fix_encoder:
            out = [e.detach() for e in out]

        down = self.encoder_scales
        main_flow = out[down]

        if self.allow_selector:
            assert _selectors is not None
            em = torch.squeeze(self.selector_module(_selectors.view(-1, 1)), 1)
            act = torch.sum(main_flow*em.view(-1, 2048, 1, 1), 1, keepdim=True)
            th = torch.sigmoid(act-model_confidence)
            main_flow = main_flow*th

            ex = torch.mean(torch.mean(act, 3), 2)
            exists_logits = torch.cat((-ex / 2., ex / 2.), 1)
        else:
            exists_logits = None

        for up in reversed(xrange(self.upsampler_scales)):
            assert down > 0
            main_flow = self._modules['up%d'%up](main_flow, out[down-1])
            down -= 1
        # now get the final saliency map (the reslution of the map = resolution_of_the_image / (2**(encoder_scales-upsampler_scales)))
        saliency_chans = self.to_saliency_chans(main_flow)

        if self.use_simple_activation:
            return torch.unsqueeze(torch.sigmoid(saliency_chans[:,0,:,:]/2), dim=1), exists_logits


        a = torch.abs(saliency_chans[:,0,:,:])
        b = torch.abs(saliency_chans[:,1,:,:])
        return torch.unsqueeze(a/(a+b), dim=1), exists_logits



class SaliencyLoss:
    def __init__(self, black_box_fn, area_loss_coef=8, smoothness_loss_coef=0.5, preserver_loss_coef=0.3,
                 num_classes=1000, area_loss_power=0.3, preserver_confidence=1, destroyer_confidence=5, **apply_mask_kwargs):
        self.black_box_fn = black_box_fn
        self.area_loss_coef = area_loss_coef
        self.smoothness_loss_coef = smoothness_loss_coef
        self.preserver_loss_coef = preserver_loss_coef
        self.num_classes = num_classes
        self.area_loss_power =area_loss_power
        self.preserver_confidence = preserver_confidence
        self.destroyer_confidence = destroyer_confidence
        self.apply_mask_kwargs = apply_mask_kwargs

    def get_loss(self, _images, _targets, _masks, _is_real_target=None, pt_store=None):
        ''' masks must be already in the range 0,1 and of shape:  (B, 1, ?, ?)'''
        if _masks.size()[-2:] != _images.size()[-2:]:
            _masks = F.upsample(_masks, (_images.size(2), _images.size(3)), mode='bilinear')

        if _is_real_target is None:
            _is_real_target = Variable(torch.zeros((_targets.size(0),)).cuda(), requires_grad=False)

        destroyed_images = apply_mask(_images, 1.-_masks, **self.apply_mask_kwargs)
        destroyed_logits = self.black_box_fn(destroyed_images)

        preserved_images = apply_mask(_images, _masks, **self.apply_mask_kwargs)
        preserved_logits = self.black_box_fn(preserved_images)

        _one_hot_targets = one_hot(_targets, self.num_classes)
        preserver_loss = cw_loss(preserved_logits, _one_hot_targets, targeted=_is_real_target == 1, t_conf=self.preserver_confidence, nt_conf=1.)
        destroyer_loss = cw_loss(destroyed_logits, _one_hot_targets, targeted=_is_real_target == 0, t_conf=1., nt_conf=self.destroyer_confidence)
        area_loss = calc_area_loss(_masks, self.area_loss_power)
        smoothness_loss = calc_smoothness_loss(_masks)

        total_loss = destroyer_loss + self.area_loss_coef*area_loss + self.smoothness_loss_coef*smoothness_loss + self.preserver_loss_coef*preserver_loss

        if pt_store is not None:
            # add variables to the pt_store
            pt_store(masks=_masks)
            pt_store(destroyed=destroyed_images)
            pt_store(preserved=preserved_images)
            pt_store(area_loss=area_loss)
            pt_store(smoothness_loss=smoothness_loss)
            pt_store(destroyer_loss=destroyer_loss)
            pt_store(preserver_loss=preserver_loss)
            pt_store(preserved_logits=preserved_logits)
            pt_store(destroyed_logits=destroyed_logits)
        return total_loss




import torch
from sal.utils.pytorch_fixes import *
from sal.utils.pytorch_trainer import *
from sal.saliency_model import SaliencyModel, SaliencyLoss, get_black_box_fn
from sal.datasets import imagenet_dataset
from sal.utils.resnet_encoder import resnet50encoder
from torchvision.models.resnet import resnet50
import pycat



# ---- config ----
# You can choose your own dataset and a black box classifier as long as they are compatible with the ones below.
# The training code does not need to be changed and the default values should work well for high resolution ~300x300 real-world images.
# By default we train on 224x224 resolution ImageNet images with a resnet50 black box classifier.
dts = imagenet_dataset
black_box_fn = get_black_box_fn(model_zoo_model=resnet50)
# ----------------



train_dts = dts.get_train_dataset()
val_dts = dts.get_val_dataset()

# Default saliency model with pretrained resnet50 feature extractor, produces saliency maps which have resolution 4 times lower than the input image.
saliency = SaliencyModel(resnet50encoder(pretrained=True), 5, 64, 3, 64, fix_encoder=True, use_simple_activation=False, allow_selector=True)

saliency_p = nn.DataParallel(saliency).cuda()
saliency_loss_calc = SaliencyLoss(black_box_fn, smoothness_loss_coef=0.005) # model based saliency requires very small smoothness loss and therefore can produce very sharp masks
optim_phase1 = torch_optim.Adam(saliency.selector_module.parameters(), 0.001, weight_decay=0.0001)
optim_phase2 = torch_optim.Adam(saliency.get_trainable_parameters(), 0.001, weight_decay=0.0001)

@TrainStepEvent()
@EveryNthEvent(2000)
def lr_step_phase1(s):
    print
    print GREEN_STR % 'Reducing lr by a factor of 10'
    for param_group in optim_phase1.param_groups:
        param_group['lr'] = param_group['lr'] / 10.


@ev_batch_to_images_labels
def ev_phase1(_images, _labels):
    __fakes = Variable(torch.Tensor(_images.size(0)).uniform_(0, 1).cuda()<FAKE_PROB)
    _targets = (_labels + Variable(torch.Tensor(_images.size(0)).uniform_(1, 999).cuda()).long()*__fakes.long())%1000
    _is_real_label = PT(is_real_label=(_targets == _labels).long())
    _masks, _exists_logits = saliency_p(_images, _targets)
    PT(exists_logits=_exists_logits)
    exists_loss = F.cross_entropy(_exists_logits, _is_real_label)
    loss = PT(loss=exists_loss)


@ev_batch_to_images_labels
def ev_phase2(_images, _labels):
    __fakes = Variable(torch.Tensor(_images.size(0)).uniform_(0, 1).cuda()<FAKE_PROB)
    _targets = PT(targets=(_labels + Variable(torch.Tensor(_images.size(0)).uniform_(1, 999).cuda()).long()*__fakes.long())%1000)
    _is_real_label = PT(is_real_label=(_targets == _labels).long())
    _masks, _exists_logits = saliency_p(_images, _targets)
    PT(exists_logits=_exists_logits)
    saliency_loss = saliency_loss_calc.get_loss(_images, _labels, _masks, _is_real_target=_is_real_label,  pt_store=PT)
    loss = PT(loss=saliency_loss)


@TimeEvent(period=5)
def phase2_visualise(s):
    pt = s.pt_store
    orig = auto_norm(pt['images'][0])
    mask = auto_norm(pt['masks'][0]*255, auto_normalize=False)
    preserved = auto_norm(pt['preserved'][0])
    destroyed = auto_norm(pt['destroyed'][0])
    print
    print 'Target (%s) = %s' % (GREEN_STR%'REAL' if pt['is_real_label'][0] else RED_STR%'FAKE!' , dts.CLASS_ID_TO_NAME[pt['targets'][0]])
    final = np.concatenate((orig, mask, preserved, destroyed), axis=1)
    pycat.show(final)



nt_phase1 = NiceTrainer(ev_phase1, dts.get_loader(train_dts, batch_size=128), optim_phase1,
                 val_dts=dts.get_loader(val_dts, batch_size=128),
                 modules=[saliency],
                 printable_vars=['loss', 'exists_accuracy'],
                 events=[lr_step_phase1,],
                 computed_variables={'exists_accuracy': accuracy_calc_op('exists_logits', 'is_real_label')})
FAKE_PROB = .5
nt_phase1.train(50)

print GREEN_STR % 'Finished phase 1 of training, waiting until the dataloading workers shut down...'

nt_phase2 = NiceTrainer(ev_phase2, dts.get_loader(train_dts, batch_size=64), optim_phase2,
                 val_dts=dts.get_loader(val_dts, batch_size=64),
                 modules=[saliency],
                 printable_vars=['loss', 'exists_accuracy'],
                 events=[phase2_visualise,],
                 computed_variables={'exists_accuracy': accuracy_calc_op('exists_logits', 'is_real_label')})
FAKE_PROB = .3
nt_phase2.train(3000)
saliency.minimalistic_save('yoursaliencymodel')  # later to restore just use saliency.minimalistic_restore methdod.
from sal.utils.pytorch_trainer import *
from sal.utils.pytorch_fixes import *
from sal.utils.mask import *

from torchvision.models.resnet import resnet50
from sal.datasets import imagenet_dataset



def get_black_box_fn(model_zoo_model=resnet50, image_domain=(-2., 2.)):
    black_box_model = model_zoo_model(pretrained=True)

    black_box_model.train(False)
    black_box_model = torch.nn.DataParallel(black_box_model).cuda()

    def black_box_fn(_images):
        return black_box_model(adapt_to_image_domain(_images, image_domain))
    return black_box_fn




class IterativeSaliency:
    def __init__(self, black_box_fn=None, mask_resolution=15, num_classes=1000, default_iterations=50):
        if black_box_fn is None:
            self.black_box_fn = get_black_box_fn()  # defaults to ResNet-50 on ImageNet
        self.default_iterations = default_iterations
        self.mask_resolution = mask_resolution
        self.num_classes = num_classes

    def get_saliency_maps(self, _images, _targets, iterations=None):
        if iterations is None:
            iterations = self.default_iterations

        _one_hot_targets = one_hot(_targets, self.num_classes)
        _mask = nn.Parameter(torch.Tensor(_images.size(0), 1, self.mask_resolution, self.mask_resolution).fill_(0.))
        for iteration in xrange(iterations):
            mask = F.upsample(F.sigmoid(_mask), (_images.size(2), _images.size(3)), mode='bilinear')

            destroyed_images = apply_mask(_images, 1.-mask)
            destroyed_logits = self.black_box_fn(destroyed_images)

            destroyer_loss = cw_loss(destroyed_logits, _one_hot_targets, targeted=False)
            area_loss = calc_area_loss(mask)
            smoothness_loss = calc_smoothness_loss(mask)

            total_loss = destroyer_loss + 2.*area_loss + 0.01*smoothness_loss




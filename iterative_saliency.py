from sal.utils.pytorch_trainer import *
from sal.utils.pytorch_fixes import *
from sal.saliency_model import SaliencyLoss, get_black_box_fn



class IterativeSaliency:
    def __init__(self, black_box_fn=None, mask_resolution=56, num_classes=1000, default_iterations=200, **loss_kwargs):
        if black_box_fn is None:
            self.black_box_fn = get_black_box_fn()  # defaults to ResNet-50 on ImageNet
        self.default_iterations = default_iterations
        self.mask_resolution = mask_resolution
        self.num_classes = num_classes
        self.saliency_loss_calc = SaliencyLoss(self.black_box_fn, **loss_kwargs)

    def get_saliency_maps(self, _images, _targets, iterations=None):
        if iterations is None:
            iterations = self.default_iterations

        _mask = nn.Parameter(torch.Tensor(_images.size(0), 2, self.mask_resolution, self.mask_resolution).fill_(0.5).cuda())
        optim = torch_optim.SGD([_mask], 0.3, 0.9, nesterov=True)
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
            m = PT['masks'][0]
            pycat.show(np.concatenate((np.zeros_like(m), np.zeros_like(m), m), axis=0))
            pycat.show(PT['destroyed'][0])
        return _mask_







from PIL import Image
ims = Variable(torch.Tensor(np.expand_dims(np.transpose(np.array(Image.open(os.path.join(os.path.dirname(__file__), 'sal/utils/test2.jpg'))), (2, 0, 1)), 0)/255.*2-1.), requires_grad=False).cuda()
labels = Variable(torch.Tensor([340]), requires_grad=False).cuda()


i = IterativeSaliency()
i.get_saliency_maps(ims, labels)
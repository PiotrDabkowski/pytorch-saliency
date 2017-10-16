from sal.utils.pytorch_trainer import *
from sal.utils.pytorch_fixes import adapt_to_image_domain

from torchvision.models.resnet import resnet50
from sal.datasets import imagenet_dataset


# ---- config ----
batch_size = 256
dts = imagenet_dataset
val_dts = dts.get_val_dataset(size=224)
num_validation_examples = 1e10  # tests on up to specified number of examples, set to 1e10 to test on the whole val dataset.

black_box_model = resnet50(pretrained=True)
image_domain = -2., 2. # image domain of the black box. If images were scaled to have zero mean and unit variance then it will be approx -2, 2.
# ----------------

black_box_model.train(False)
print 'Moving the model to the GPUs'
black_box_model = torch.nn.DataParallel(black_box_model).cuda()

@ev_batch_to_images_labels
def ev(_images, _labels):
    _, guesses = torch.max(black_box_model(adapt_to_image_domain(_images, image_domain)), 1)
    return torch.mean((guesses==_labels).float()).data[0]

print 'Please wait, validating, it can take a few minutes...'
scores = []
i = 0
for batch in dts.get_loader(val_dts, batch_size=batch_size):
    scores.append(ev(batch))
    i += batch_size
    if i > num_validation_examples:
        break
print 'Top 1 accuracy:', np.mean(scores)
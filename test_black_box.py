from sal.utils.pytorch_trainer import *
from sal.datasets import imagenet_dataset
from sal.saliency_model import get_black_box_fn
from sal.utils.resnet_encoder import get_resnet50encoder_black_box_fn

# examle model choices
from torchvision.models.alexnet import alexnet
from torchvision.models.resnet import resnet50
from torchvision.models.squeezenet import squeezenet1_1

# ---- config ----
batch_size = 128
dts = imagenet_dataset
val_dts = dts.get_val_dataset(size=224)
num_validation_examples = 5000  # tests on up to specified number of examples, set to 1e10 to test on the whole val dataset.
black_box_fn = get_black_box_fn(model_zoo_model=resnet50)
# ----------------


@ev_batch_to_images_labels
def ev(_images, _labels):
    _, guesses = torch.max(black_box_fn(_images), 1)
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
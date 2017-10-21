from torchvision.transforms import *
from torchvision.datasets import ImageFolder
from torch.utils.data import dataloader
import pycat, time, random
from ..utils.pytorch_fixes import *
import os

# Images must be segregated in folders by class! So both train and val folders should contain 1000 folders, one for each class.
# PLEASE EDIT THESE 2 LINES:
IMAGE_NET_TRAIN_PATH = '/home/piter/ImageNetFull/train/'
IMAGE_NET_VAL_PATH = '/home/piter/ImageNetFull/val/'


#-----------------------------------------------------

SUGGESTED_BS = 128
NUM_CLASSES = 1000
SUGGESTED_EPOCHS_PER_STEP = 11
SUGGESTED_BASE = 64


def get_train_dataset(size=224):
    if not (os.path.exists(IMAGE_NET_TRAIN_PATH) and os.path.exists(IMAGE_NET_VAL_PATH)):
        raise ValueError(
            'Please make sure that you specify a path to the ImageNet dataset folder in sal/datasets/imagenet_dataset.py file!')
    return ImageFolder(IMAGE_NET_TRAIN_PATH, transform=Compose([
        RandomSizedCrop2(size, min_area=0.3),
        RandomHorizontalFlip(),
        ToTensor(),
        STD_NORMALIZE,  # Images will be in range -1 to 1
    ]))


def get_val_dataset(size=224):
    if not (os.path.exists(IMAGE_NET_TRAIN_PATH) and os.path.exists(IMAGE_NET_VAL_PATH)):
        raise ValueError(
            'Please make sure that you specify a path to the ImageNet dataset folder in sal/datasets/imagenet_dataset.py file!')
    return ImageFolder(IMAGE_NET_VAL_PATH, transform=Compose([
        Scale(224),
        CenterCrop(size),
        ToTensor(),
        STD_NORMALIZE,
    ]))

def get_loader(dataset, batch_size=64, pin_memory=True):
    return dataloader.DataLoader(dataset=dataset, batch_size=batch_size,
                                 shuffle=True, drop_last=True, num_workers=8, pin_memory=pin_memory)


def test():
    BS = 64
    SAMP = 20
    dts = get_val_dataset()
    loader = get_loader(dts, batch_size=BS)
    i = 0
    t = time.time()
    for ims, labs in loader:
        i+=1
        if not i%20:
            print 'min', torch.min(ims),'max', torch.max(ims), 'var', torch.var(ims), 'mean', torch.mean(ims)
            print "Images per second:", SAMP*BS/(time.time()-t)
            pycat.show(ims[0].numpy())
            t = time.time()
        if i==100:
            break



from imagenet_synset import synset
SYNSET_TO_NAME= dict((e[:9], e[10:]) for e in synset.splitlines())
SYNSET_TO_CLASS_ID = dict((e[:9], i) for i, e in enumerate(synset.splitlines()))

CLASS_ID_TO_SYNSET = {v:k for k,v in SYNSET_TO_CLASS_ID.items()}
CLASS_ID_TO_NAME = {i:SYNSET_TO_NAME[CLASS_ID_TO_SYNSET[i]] for i in CLASS_ID_TO_SYNSET}
CLASS_NAME_TO_ID = {v:k for k, v in CLASS_ID_TO_NAME.items()}









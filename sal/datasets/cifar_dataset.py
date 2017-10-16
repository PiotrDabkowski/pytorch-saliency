import torch
from torchvision.transforms import *
from torchvision.datasets import CIFAR10
from torch.utils.data import dataloader
import pycat, time, random
from ..utils.pytorch_fixes import *



SUGGESTED_BS = 512
SUGGESTED_EPOCHS_PER_STEP = 22
SUGGESTED_BASE = 32
NUM_CLASSES = 10
CLASSES = '''airplane airplane
automobile automobile
bird bird
cat cat
deer deer
dog dog
frog frog
horse horse
ship ship
truck truck'''.splitlines()

def get_train_dataset(size=32):
    return  CIFAR10('/home/piter/CIFAR10Dataset', train=True, transform=Compose(
        [Scale(size), RandomSizedCrop2(size, min_area=0.5), RandomHorizontalFlip(), ToTensor(), STD_NORMALIZE]), download=True)



def get_val_dataset(size=32):
    return CIFAR10('/home/piter/CIFAR10Dataset', train=False, transform=Compose(
        [Scale(size), CenterCrop(size), ToTensor(), STD_NORMALIZE]), download=True)


def get_loader(dataset, batch_size=64, pin_memory=True):
    return dataloader.DataLoader(dataset=dataset, batch_size=batch_size,
                                 shuffle=True, drop_last=True, num_workers=8, pin_memory=True)


def test():
    BS = 64
    SAMP = 20
    dts = get_train_dataset()
    loader = get_loader(dts, batch_size=BS)
    i = 0
    t = time.time()
    for ims, labs in loader:
        i+=1
        if not i%20:
            print "Images per second:", SAMP*BS/(time.time()-t)
            pycat.show(ims[0].numpy())
            t = time.time()
        if i==100:
            break
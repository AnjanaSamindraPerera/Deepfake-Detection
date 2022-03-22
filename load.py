import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import os
import cv2
from dataset.transform import xception_default_data_transforms

train_dataset =  torchvision.datasets.ImageFolder("/hdd/2017CS128/Research/FFc40-split/train", transform=xception_default_data_transforms['train'])
val_dataset =  torchvision.datasets.ImageFolder("/hdd/2017CS128/Research/FFc40-split/test", transform=xception_default_data_transforms['val'])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=False, num_workers=8)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True, drop_last=False, num_workers=8)
train_dataset_size = len(train_dataset)
val_dataset_size = len(val_dataset)

print(train_dataset.class_to_idx)
print(val_dataset.class_to_idx)
print(train_loader)
print(val_loader)
print(train_dataset_size)
print(val_dataset_size)

next(iter(train_loader))

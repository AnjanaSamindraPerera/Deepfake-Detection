import seaborn as sns
import os
from functools import reduce
import argparse
from os.path import join
import cv2
import dlib
import torch
import torch.nn as nn
from PIL import Image as pil_image
from tqdm import tqdm
import torchvision
from network.classifier import *
from network.transform import mesonet_data_transforms
from sklearn.metrics import confusion_matrix,classification_report,confusion_matrix,accuracy_score
import statistics
import numpy as np
from sklearn.metrics import plot_roc_curve, roc_auc_score,roc_curve  
import matplotlib.pyplot as plt

#creat train and val dataloader
test_dataset = torchvision.datasets.ImageFolder("/hdd/2017CS128/Research/FFc40-RESRGAN/test", transform=mesonet_data_transforms['test'])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True, drop_last=False, num_workers=8)

#print(test_dataset.class_to_idx)

#Creat the model
model = MesoInception4()
model.load_state_dict(torch.load("/hdd/2017CS128/Research/MesoNet-Pytorch/output/Mesonet_Inception_RESRGAN3/29_Mesonet_Inception.pkl"))
if isinstance(model, torch.nn.DataParallel):
		model = model.module
model = model.cuda()
model.eval()


y_pred_list = []
y_true_list = []

with torch.no_grad():
    for x_batch, y_batch in tqdm(test_loader):
        x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
        y_test_pred = model(x_batch)
        #_, y_pred_tag = torch.max(y_test_pred, dim = 1)
        y_pred_list.append(y_test_pred[:,1].cpu().numpy())
        y_true_list.append(y_batch.cpu().numpy())

y_pred_proba = np.concatenate(y_pred_list).ravel()
Y_test = np.concatenate(y_true_list).ravel()


# calculate the false positive rate (for) and true positive rate (tpr) - already calculated
fpr, tpr, thresholds = roc_curve(Y_test, y_pred_proba)

sns.set() # apply the default default seaborn theme

plt.plot(fpr, tpr)

plt.plot(fpr, fpr, linestyle = '--', color = 'k')

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

AUROC = np.round(roc_auc_score(Y_test, y_pred_proba), 2)

plt.title(f'MesoInception Model with RESRGAN ROC curve; AUC: {AUROC}');
plt.savefig('MesoInception_AUC_RESRGAN.png')
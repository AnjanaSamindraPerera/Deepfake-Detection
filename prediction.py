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
from network.models import model_selection
from dataset.transform import xception_default_data_transforms
from sklearn.metrics import confusion_matrix,classification_report,confusion_matrix,accuracy_score
import statistics
import numpy as np

#creat train and val dataloader
test_dataset = torchvision.datasets.ImageFolder("/hdd/2017CS128/Research/FFc40-RESRGAN/test", transform=xception_default_data_transforms['test'])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True, drop_last=False, num_workers=8)

#print(test_dataset.class_to_idx)

#Creat the model
model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
model.load_state_dict(torch.load("/hdd/2017CS128/Research/Deepfake-Detection/output/xception_RESRGAN_LAST/28_fs_c0_299.pkl"))
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
        _, y_pred_tag = torch.max(y_test_pred, dim = 1)
        y_pred_list.append(y_pred_tag.cpu().numpy())
        y_true_list.append(y_batch.cpu().numpy())

y_pred_list = np.concatenate(y_pred_list).ravel()
y_true_list = np.concatenate(y_true_list).ravel()

print('\n')
print("Testing accuracy with Xception:",accuracy_score(y_true_list, y_pred_list))
print(classification_report(y_true_list, y_pred_list))
print('\n')

cm1=confusion_matrix(y_true_list, y_pred_list)
print(cm1)

tn, fp, fn, tp = confusion_matrix(y_true_list, y_pred_list).ravel()
specificity = tn / (tn+fp)
recall = tp/(tp+fn)
print('\n')
print('Sensitivity : ', recall )         # easy way to get matrix cell values
print('Specificity : ', specificity)
print('\n')
print("correct_real:",tp)
print("correct_deepfake:",tn)
print("misclassified_real:",fn)
print("misclassified_deepfake:",fp)
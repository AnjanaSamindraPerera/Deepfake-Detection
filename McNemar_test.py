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
import matplotlib.pyplot as plt

# Libs implementations
from mlxtend.evaluate import mcnemar
from mlxtend.evaluate import mcnemar_table
from scipy.stats import norm, chi2
from mlxtend.evaluate import proportion_difference
from mlxtend.plotting import checkerboard_plot

# Without SR

# Creat train and val dataloader
test_dataset = torchvision.datasets.ImageFolder("/hdd/2017CS128/Research/FFc40-split2/test", transform=xception_default_data_transforms['test'])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=False, num_workers=8)

# Create the model
model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
model.load_state_dict(torch.load("/hdd/2017CS128/Research/Deepfake-Detection/output/xception-normal2/14_fs_c0_299.pkl"))
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

y_pred_list_without_SR = np.concatenate(y_pred_list).ravel()
y_true_list = np.concatenate(y_true_list).ravel()


# With SR

# Creat train and val dataloader
test_dataset = torchvision.datasets.ImageFolder("/hdd/2017CS128/Research/FFc40-RESRGAN/test", transform=xception_default_data_transforms['test'])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=False, num_workers=8)

# Create the model
model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
model.load_state_dict(torch.load("/hdd/2017CS128/Research/Deepfake-Detection/output/xception_RESRGAN_LAST/29_fs_c0_299.pkl"))
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

y_pred_list_with_SR = np.concatenate(y_pred_list).ravel()
y_true_list = np.concatenate(y_true_list).ravel()

# Calculate the accuracy
acc1 = accuracy_score(y_true_list, y_pred_list_without_SR)
acc2 = accuracy_score(y_true_list, y_pred_list_with_SR)
print("\nWithout SR accuracy : ",acc1)
print("With SR accuracy : ",acc2)

# McNemar's test
print("\nMcNemar's test")
table = mcnemar_table(y_target=y_true_list, y_model1=y_pred_list_with_SR, y_model2=y_pred_list_without_SR)
print(table)
chi2_, p = mcnemar(ary=table, corrected=True)
print(f"\nchi² statistic: {chi2_}, p-value: {p}\n")

# Interpret the p-value
alpha = 0.05
if p > alpha:
	print('Same proportions of errors (fail to reject H0)')
else:
	print('Different proportions of errors (reject H0)')

def mcnemar_test2(y_true, y_1, y_2):
    b = sum(np.logical_and((y_pred_list_with_SR != y_true_list),(y_pred_list_without_SR == y_true_list)))
    c = sum(np.logical_and((y_pred_list_with_SR == y_true_list),(y_pred_list_without_SR != y_true_list)))
    
    c_ = (np.abs(b - c) - 1)**2 / (b + c)
    
    p_value = chi2.sf(c_, 1)
    return c_, p_value

print("\nMcNemar's test")
chi2_, p = mcnemar_test2(y_true_list, y_pred_list_with_SR, y_pred_list_without_SR)
print(f"chi² statistic: {chi2_}, p-value: {p}\n")


brd = checkerboard_plot(table,
                        figsize=(6, 6),
                        fmt='%d',
                        cell_colors=['purple', 'whitesmoke'],
                        col_labels=['without SR (Hit)', 'without SR (Miss)'],
                        font_colors=['white', 'purple'],
                        row_labels=['with SR (Hit)', 'with SR (Miss)'])

plt.savefig("McNemar's test.png")


# Run the T test
print("Proportions Z-Test")
z, p = proportion_difference(acc1, acc2, n_1=len(y_true_list))
print(f"z statistic: {z}, p-value: {p}\n")



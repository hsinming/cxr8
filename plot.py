import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as tv_utils
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torchvision
import torch.autograd as autograd
from PIL import Image
import imp
import os
import sys
import math
import time
import random
import shutil
import cv2
import scipy.misc
from glob import glob
import sklearn
import logging

from tqdm import tqdm
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

def norm_meanstd(arr, mean, std):
    return (arr - mean) / std

def denorm_meanstd(arr, mean, std):
    return (arr * std) + mean

def norm255_tensor(arr):
    """Given a color image/where max pixel value in each channel is 255
    returns normalized tensor or array with all values between 0 and 1"""
    return arr / 255.

def denorm255_tensor(arr):
    return arr * 255.

def load_img_as_pil(img_path):
    return Image.open(img_path)

def load_img_as_np_arr(img_path):
    return scipy.misc.imread(img_path) #scipy

def load_img_as_tensor(img_path):
    pil_image = Image.open(img_path)
    return transforms.ToTensor()(pil_image)

def save_tensor_img(tns, fpath):
    tv_utils.save_image(tns, fpath)

def save_pil_img(pil_img, fpath):
    pil_img.save(fpath)

def save_numpy_img(np_arr, fpath):
    scipy.misc.imsave(fpath, np_arr)

def plot_np_array(arr_img, fs=(3,3)):
    plt.figure(figsize=fs)
    plt.imshow(arr_img.astype('uint8'))
    plt.show()

def plot_tensor(tns_img, fs=(3,3)):
    "Takes a normalized tensor [0,1] and plots PIL image"
    pil_from_tns = transforms.ToPILImage()(tns_img)
    plt.figure(figsize=fs)
    plt.imshow(pil_from_tns)
    plt.show()

def plot_pil(pil_img, fs=(3,3)):
    plt.figure(figsize=fs)
    plt.imshow(pil_img)
    plt.show()

def imshow(inp, mean_arr, std_arr, title=None):
    # Input is normalized Tensor or Numpy Arr
    if inp.size(0) == 1:
        inp = np.squeeze(inp.numpy())
        kwargs = {'cmap':'gray'}
    else:
        inp = inp.numpy().transpose((1, 2, 0))
        kwargs = {}
    inp = std_arr * inp + mean_arr
    plt.imshow(inp, **kwargs)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def visualize_preds(model, data_loader, class_names, mean_arr, std_arr, num_images=6):
    images_so_far = 0
    fig = plt.figure()

    for i, data in enumerate(data_loader):
        inputs, labels = data
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        preds = preds.cpu().numpy()
        labels = labels.data.cpu().numpy()
        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('P: {}, A:{}'.format(class_names[preds[j][0]],
                                              class_names[labels[j]]))
            imshow(inputs.cpu().data[j], mean_arr, std_arr)

            if images_so_far == num_images:
                return
        plt.tight_layout()

def plot_bw_samples(arr, dim=(4,4), figsize=(6,6)):
    if type(arr) is not np.ndarray:
        arr = arr.numpy()
    bs = arr.shape[0]
    arr = arr.reshape(bs, 28, 28)
    plt.figure(figsize=figsize)
    for i,img in enumerate(arr):
        plt.subplot(*dim, i+1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.tight_layout()

def plot_samples(arr, mean, std, dim=(4,4), figsize=(6,6)):
    if type(arr) is not np.ndarray:
        arr = arr.numpy().transpose((0, 2, 3, 1))
    arr = denorm_meanstd(arr, mean, std)
    plt.figure(figsize=figsize)
    for i,img in enumerate(arr):
        plt.subplot(*dim, i+1)
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()

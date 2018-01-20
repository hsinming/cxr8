import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as tv_utils

from torch.utils.data import DataLoader, Dataset
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
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

import numpy as np
import pandas as pd
import os
import torch
import visdom
import shutil
import sys
from pathlib import Path
from functools import partial


use_gpu = torch.cuda.is_available
data_dir = "/data/CXR8/images"
save_dir = "./savedModels"
label_path = {'train':"./Train_Label_simple.csv", 'val':"./Val_Label_simple.csv", 'test':"Test_Label_simple.csv"}



class Experiment():
    def __init__(self, name, root, logger=None):
        self.name = name
        self.root = os.path.join(root, name)
        self.logger = logger
        self.epoch = 1
        self.best_val_loss = sys.maxsize
        self.best_val_loss_epoch = 1
        self.weights_dir = os.path.join(self.root, 'weights')
        self.history_dir = os.path.join(self.root, 'history')
        self.results_dir = os.path.join(self.root, 'results')
        self.latest_weights = os.path.join(self.weights_dir, 'latest_weights.pth')
        self.latest_optimizer = os.path.join(self.weights_dir, 'latest_optim.pth')
        self.best_weights_path = self.latest_weights
        self.best_optimizer_path = self.latest_optimizer
        self.train_history_fpath = os.path.join(self.history_dir, 'train.csv')
        self.val_history_fpath = os.path.join(self.history_dir, 'val.csv')
        self.test_history_fpath = os.path.join(self.history_dir, 'test.csv')
        self.loss_history = {
            'train': np.array([]),
            'val': np.array([]),
            'test': np.array([])
        }
        self.acc_history = {
            'train': np.array([]),
            'val': np.array([]),
            'test': np.array([])
        }
        self.viz = visdom.Visdom()
        self.visdom_plots = self.init_visdom_plots()

    def log(self, msg):
        if self.logger:
            self.logger.info(msg)

    def init(self):
        self.log("Creating new experiment")
        self.init_dirs()
        self.init_history_files()

    def resume(self, model, optim, weights_fpath=None, optim_path=None):
        self.log("Resuming existing experiment")
        if weights_fpath is None:
            weights_fpath = self.latest_weights
        if optim_path is None:
            optim_path = self.latest_optimizer

        model, state = self.load_weights(model, weights_fpath)
        optim = self.load_optimizer(optim, optim_path)

        self.best_val_loss = state['best_val_loss']
        self.best_val_loss_epoch = state['best_val_loss_epoch']
        self.epoch = state['last_epoch'] + 1
        self.load_history_from_file('train')
        self.load_history_from_file('val')

        return model, optim

    def init_dirs(self):
        try:
            os.makedirs(self.weights_dir)
        except FileExistsError:
            pass
        try:
            os.makedirs(self.history_dir)
        except FileExistsError:
            pass
        try:
            os.makedirs(self.results_dir)
        except FileExistsError:
            pass

    def init_history_files(self):
        Path(self.train_history_fpath).touch()
        Path(self.val_history_fpath).touch()
        Path(self.test_history_fpath).touch()

    def init_visdom_plots(self):
        loss = self.init_viz_train_plot('loss')
        accuracy = self.init_viz_train_plot('accuracy')
        summary = self.init_viz_txt_plot('summary')
        return {
            'loss': loss,
            'accuracy': accuracy,
            'summary': summary
        }

    def init_viz_train_plot(self, title):
        return self.viz.line(
            X=np.array([1]),
            Y=np.array([[1, 1]]),
            opts=dict(
                xlabel='epoch',
                ylabel=title,
                title=self.name + ' ' + title,
                legend=['Train', 'Validation']
            ),
            env=self.name
        )

    def init_viz_txt_plot(self, title):
        return self.viz.text(
            "Initializing.. " + title,
            env=self.name
        )

    def viz_epochs(self):
        epochs = np.arange(1, self.epoch + 1)
        return np.stack([epochs, epochs], 1)

    def update_viz_loss_plot(self):
        loss = np.stack([self.loss_history['train'],
                         self.loss_history['val']], 1)
        window = self.visdom_plots['loss']
        return self.viz.line(
            X=self.viz_epochs(),
            Y=loss,
            win=window,
            env=self.name,
            opts=dict(
                xlabel='epoch',
                ylabel='loss',
                title=self.name + ' ' + 'loss',
                legend=['Train', 'Validation']
            ),
        )

    def update_viz_acc_plot(self):
        acc = np.stack([self.acc_history['train'],
                        self.acc_history['val']], 1)
        window = self.visdom_plots['accuracy']
        return self.viz.line(
            X=self.viz_epochs(),
            Y=acc,
            win=window,
            env=self.name,
            opts=dict(
                xlabel='epoch',
                ylabel='accuracy',
                title=self.name + ' ' + 'accuracy',
                legend=['Train', 'Validation']
            )
        )

    def update_viz_summary_plot(self):
        trn_loss = self.loss_history['train'][-1]
        val_loss = self.loss_history['val'][-1]
        trn_acc = self.acc_history['train'][-1]
        val_acc = self.acc_history['val'][-1]
        txt = ("""Epoch: %d
            Train - Loss: %.3f Acc: %.3f
            Test - Loss: %.3f Acc: %.3f""" % (self.epoch,
                                              trn_loss, trn_acc, val_loss, val_acc))
        window = self.visdom_plots['summary']
        return self.viz.text(
            txt,
            win=window,
            env=self.name
        )

    def load_history_from_file(self, dset_type):
        fpath = os.path.join(self.history_dir, dset_type + '.csv')
        data = np.loadtxt(fpath, delimiter=',').reshape(-1, 3)
        self.loss_history[dset_type] = data[:, 1]
        self.acc_history[dset_type] = data[:, 2]

    def append_history_to_file(self, dset_type, loss, acc):
        fpath = os.path.join(self.history_dir, dset_type + '.csv')
        with open(fpath, 'a') as f:
            f.write('{},{},{}\n'.format(self.epoch, loss, acc))

    def save_history(self, dset_type, loss, acc):
        self.loss_history[dset_type] = np.append(
            self.loss_history[dset_type], loss)
        self.acc_history[dset_type] = np.append(
            self.acc_history[dset_type], acc)
        self.append_history_to_file(dset_type, loss, acc)

        if dset_type == 'val' and self.is_best_loss(loss):
            self.best_val_loss = loss
            self.best_val_loss_epoch = self.epoch

    def is_best_loss(self, loss):
        return loss < self.best_val_loss

    def save_weights(self, model, trn_loss, val_loss, trn_acc, val_acc):
        weights_fname = self.name + '-weights-%d-%.3f-%.3f-%.3f-%.3f.pth' % (
            self.epoch, trn_loss, trn_acc, val_loss, val_acc)
        weights_fpath = os.path.join(self.weights_dir, weights_fname)
        torch.save({
            'last_epoch': self.epoch,
            'trn_loss': trn_loss,
            'val_loss': val_loss,
            'trn_acc': trn_acc,
            'val_acc': val_acc,
            'best_val_loss': self.best_val_loss,
            'best_val_loss_epoch': self.best_val_loss_epoch,
            'experiment': self.name,
            'state_dict': model.state_dict()
        }, weights_fpath)
        shutil.copyfile(weights_fpath, self.latest_weights)
        if self.is_best_loss(val_loss):
            self.best_weights_path = weights_fpath

    def load_weights(self, model, fpath):
        self.log("loading weights '{}'".format(fpath))
        state = torch.load(fpath)
        model.load_state_dict(state['state_dict'])
        self.log(
            "loaded weights from experiment %s (last_epoch %d, trn_loss %s, trn_acc %s, val_loss %s, val_acc %s)" % (
                self.name, state['last_epoch'], state['trn_loss'],
                state['trn_acc'], state['val_loss'], state['val_acc']))
        return model, state

    def save_optimizer(self, optimizer, val_loss):
        optim_fname = self.name + '-optim-%d.pth' % (self.epoch)
        optim_fpath = os.path.join(self.weights_dir, optim_fname)
        torch.save({
            'last_epoch': self.epoch,
            'experiment': self.name,
            'state_dict': optimizer.state_dict()
        }, optim_fpath)
        shutil.copyfile(optim_fpath, self.latest_optimizer)
        if self.is_best_loss(val_loss):
            self.best_optimizer_path = optim_fpath

    def load_optimizer(self, optimizer, fpath):
        self.log("loading optimizer '{}'".format(fpath))
        optim = torch.load(fpath)
        optimizer.load_state_dict(optim['state_dict'])
        self.log("loaded optimizer from session {}, last_epoch {}"
                 .format(optim['experiment'], optim['last_epoch']))
        return optimizer

    def plot_and_save_history(self):
        trn_data = np.loadtxt(self.train_history_fpath, delimiter=',').reshape(-1, 3)
        val_data = np.loadtxt(self.val_history_fpath, delimiter=',').reshape(-1, 3)

        trn_epoch, trn_loss, trn_acc = np.split(trn_data, [1, 2], axis=1)
        val_epoch, val_loss, val_acc = np.split(val_data, [1, 2], axis=1)

        # Loss
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plt.plot(trn_epoch, trn_loss, label='Train')
        plt.plot(val_epoch, val_loss, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        ax.set_yscale('log')
        loss_fname = os.path.join(self.history_dir, 'loss.png')
        plt.savefig(loss_fname)

        # Accuracy
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plt.plot(trn_epoch, trn_acc, label='Train')
        plt.plot(val_epoch, val_acc, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        ax.set_yscale('log')
        plt.legend()
        acc_fname = os.path.join(self.history_dir, 'accuracy.png')
        plt.savefig(acc_fname)

        # Combined View - loss-accuracy.png
        loss_acc_fname = os.path.join(self.history_dir, 'loss-acc.png')
        os.system('convert +append {} {} {}'.format(loss_fname, acc_fname, loss_acc_fname))


class ResNet50Modified(nn.Module):
    def __init__(self, logger=None):
        super(ResNet50Modified, self).__init__()
        self.logger=logger
        # Get number of classes
        classes = pd.read_csv(label_path['train'], header=None, nrows=1).ix[0, :].as_matrix()
        classes = classes[1:]
        self.n_class = len(classes)

        # Get features from resnet50
        self.model_ft = models.resnet50(pretrained=True)

        for param in self.model_ft.parameters():
            param.requires_grad = False

        self.transition = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=3, padding=1, stride=1, bias=False),
        )
        self.globalPool = nn.Sequential(
            nn.MaxPool2d(32)
        )
        self.prediction = nn.Sequential(
            nn.Linear(2048, self.n_class),
            nn.Sigmoid()
        )

    def log(self, msg):
        if self.logger:
            self.logger.debug(msg)

    def forward(self, x):
        x = self.model_ft.conv1(x)
        x = self.model_ft.bn1(x)
        x = self.model_ft.relu(x)
        x = self.model_ft.maxpool(x)

        x = self.model_ft.layer1(x)
        x = self.model_ft.layer2(x)
        x = self.model_ft.layer3(x)
        x = self.model_ft.layer4(x)  # out: bs*2048*32*32

        x = self.transition(x)  # out: bs*2048*32*32
        x = self.globalPool(x)  # out: bs*2048*1*1
        x = x.view(x.size(0), -1)  # out: bs*2048
        x = self.prediction(x)
        return x


class CXRDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform = None):
        self.labels= pd.read_csv(csv_file, header=0)
        self.root_dir = root_dir
        self.transform = transform
        self.classes = pd.read_csv(csv_file, header=None,nrows=1).ix[0, :].as_matrix()
        self.classes = self.classes[1:]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels.ix[idx, 0])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels.ix[idx, 1:].as_matrix().astype('float')
        label = torch.from_numpy(label).type(torch.FloatTensor)
        return image, label


def weighted_BCELoss(output, target, weights=None):
    if weights is not None:
        assert len(weights) == 2
        loss = -weights[0] * (target * torch.log(output)) - weights[1] * ((1 - target) * torch.log(1 - output))
    else:
        loss = -target * torch.log(output) - (1 - target) * torch.log(1 - output)
    return torch.sum(loss)


def get_logger(ch_log_level=logging.ERROR,
               fh_log_level=logging.INFO):
    logging.shutdown()
    imp.reload(logging)
    logger = logging.getLogger("resnet50")
    logger.setLevel(logging.DEBUG)

    # Console Handler
    if ch_log_level:
        ch = logging.StreamHandler()
        ch.setLevel(ch_log_level)
        ch.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(ch)

    # File Handler
    if fh_log_level:
        fh = logging.FileHandler('resnet50.log')
        fh.setLevel(fh_log_level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def train(net, dataloader, criterion, optimizer, epoch=1):
    net.train()
    n_batches = len(dataloader)
    total_loss = 0
    total_acc = 0
    for inputs, targets in dataloader:
        weights = get_weight(targets)
        inputs = Variable(inputs.cuda())
        targets = Variable(targets.cuda())

        ## Forward Pass
        output = net(inputs)

        ## Clear Gradients
        net.zero_grad()

        loss = criterion(output, targets, weights)

        ## Backprop
        loss.backward()
        optimizer.step()

        preds = get_predictions(output)
        accuracy = get_accuracy(preds, targets.data.cpu().numpy())

        total_loss += loss.data[0]
        total_acc += accuracy

    mean_loss = total_loss / n_batches
    mean_acc = total_acc / n_batches
    return mean_loss, mean_acc


def get_weight(labels):
    P, N, BP, BN = 0, 0, 0, 0
    label_array = labels.numpy()
    number, count = np.unique(label_array, return_counts=True)
    frequency = dict(zip(number, count))
    P = frequency[1]
    N = frequency[0]

    try:
        BP = (P + N) / P
    except:
        BP = 100000

    try:
        BN = (P + N) / N
    except:
        BN = 100000

    weights = [BP, BN]

    if use_gpu:
        weights = torch.FloatTensor(weights).cuda()

    return weights


def get_predictions(model_output):
    # Flatten and Get ArgMax to compute accuracy
    val, idx = torch.max(model_output, dim=1)
    return idx.data.cpu().view(-1).numpy()


def get_accuracy(preds, targets):
    correct = np.sum(preds == targets)
    return correct / len(targets)


def test(net, test_loader, criterion, epoch=1):
    net.eval()
    test_loss = 0
    test_acc = 0
    for data, target in test_loader:
        data = Variable(data.cuda(), volatile=True)
        target = Variable(target.cuda())
        output = net(data)
        test_loss += criterion(output, target).data[0]
        pred = get_predictions(output)
        test_acc += get_accuracy(pred, target.data.cpu().numpy())
    test_loss /= len(test_loader)  # n_batches
    test_acc /= len(test_loader)
    return test_loss, test_acc


def adjust_learning_rate(lr, decay, optimizer, cur_epoch, n_epochs):
    """Sets the learning rate to the initially
        configured `lr` decayed by `decay` every `n_epochs`"""
    new_lr = lr * (decay ** (cur_epoch // n_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def main():
    N_EPOCHS = 5
    MAX_PATIENCE = 50
    LEARNING_RATE = 3e-5
    LR_DECAY = 0.995
    DECAY_LR_EVERY_N_EPOCHS = 1
    EXPERIMENT_NAME = 'resnet50_simple'
    CXR8_PATH = '/data/CXR8'
    BATCH_SIZE = 24
    CXR8_MEAN = np.array([125.867, 125.867, 125.867])
    CXR8_STD = np.array([58.158, 58.158, 58.158])
    normTransform = transforms.Normalize(CXR8_MEAN, CXR8_STD)

    trainTransform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         normTransform])
    valTransform = transforms.Compose([transforms.ToTensor(),
                                        normTransform])

    train_dataset = CXRDataset(label_path['train'], data_dir, transform=trainTransform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_dataset = CXRDataset(label_path['val'], data_dir, transform=valTransform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    logger = get_logger(ch_log_level=logging.INFO, fh_log_level=logging.INFO)
    model = ResNet50Modified(logger).cuda()
    criterion = weighted_BCELoss

    optimizer = optim.Adam([{'params':model.transition.parameters()},
                            {'params':model.globalPool.parameters()},
                            {'params':model.prediction.parameters()}], lr=LEARNING_RATE)

    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    exp = Experiment(EXPERIMENT_NAME, CXR8_PATH, logger)

    # Create New Experiment
    exp.init()

    for epoch in range(exp.epoch, exp.epoch + N_EPOCHS):
        since = time.time()

        ### Train ###
        trn_loss, trn_acc = train(model, train_loader, criterion, optimizer, epoch)
        logger.info('Epoch {:d}: Train - Loss: {:.4f}\tAcc: {:.4f}'.format(epoch, trn_loss, trn_acc))
        time_elapsed = time.time() - since
        logger.info('Train Time {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

        ### Test ###
        val_loss, val_acc = test(model, val_loader, criterion, epoch)
        logger.info('Val - Loss: {:.4f}, Acc: {:.4f}'.format(val_loss, val_acc))
        time_elapsed = time.time() - since
        logger.info('Total Time {:.0f}m {:.0f}s\n'.format(
            time_elapsed // 60, time_elapsed % 60))

        ### Save Metrics ###
        exp.save_history('train', trn_loss, trn_acc)
        exp.save_history('val', val_loss, val_acc)

        ### Checkpoint ###
        exp.save_weights(model, trn_loss, val_loss, trn_acc, val_acc)
        exp.save_optimizer(optimizer, val_loss)

        ### Plot Online ###
        exp.update_viz_loss_plot()
        exp.update_viz_acc_plot()
        exp.update_viz_summary_plot()

        ## Early Stopping ##
        if (epoch - exp.best_val_loss_epoch) > MAX_PATIENCE:
            logger.info(("Early stopping at epoch %d since no "
                         + "better loss found since epoch %.3")
                        % (epoch, exp.best_val_loss))
            break

        ### Adjust Lr ###
        adjust_learning_rate(LEARNING_RATE, LR_DECAY, optimizer,
                             epoch, DECAY_LR_EVERY_N_EPOCHS)

        exp.epoch += 1


if __name__=="__main__":
    status = main()
    sys.exit(status)
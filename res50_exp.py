#!/usr/bin/python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


from PIL import Image
import imp
import time
import logging
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

import numpy as np
import pandas as pd
import os
import visdom
import shutil
import sys
from pathlib import Path
from sklearn.metrics import roc_auc_score
from net_model import SE_ResNet50

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
use_gpu = torch.cuda.is_available
data_dir = "/data/CXR8/images"
save_dir = "./savedModels"
label_path = {'train':"./Train_Label_simple.csv", 'val':"./Val_Label_simple.csv", 'test':"Test_Label_simple.csv"}
N_EPOCHS = 5
MAX_PATIENCE = 2
LEARNING_RATE = 3e-5
LR_DECAY = 0.995
DECAY_LR_EVERY_N_EPOCHS = 1
EXPERIMENT_NAME = 'se_resnet50'
CXR8_PATH = '/data/CXR8'
BATCH_SIZE = 24
CXR8_MEAN = np.array([125.867, 125.867, 125.867])
CXR8_STD = np.array([58.158, 58.158, 58.158])



class Experiment():
    def __init__(self, name, root, logger=None):
        self.name = name
        self.root = os.path.join(root, name)
        self.logger = logger
        self.epoch = 1
        self.best_val_auc = 0
        self.best_val_auc_epoch = 1
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
        self.auc_history = {
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

        self.best_val_auc = state['best_val_auc']
        self.best_val_auc_epoch = state['best_val_auc_epoch']
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
        auc = self.init_viz_txt_plot('auc')
        summary = self.init_viz_txt_plot('summary')
        return {
            'loss': loss,
            'auc': auc,
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

    def update_viz_auc_plot(self):
        auc = np.stack([self.auc_history['train'],
                        self.auc_history['val']], 1)
        window = self.visdom_plots['auc']
        return self.viz.line(
            X=self.viz_epochs(),
            Y=auc,
            win=window,
            env=self.name,
            opts=dict(
                xlabel='epoch',
                ylabel='auc',
                title=self.name + ' ' + 'auc',
                legend=['Train', 'Validation']
            ),
        )

    def update_viz_summary_plot(self):
        trn_loss = self.loss_history['train'][-1]
        val_loss = self.loss_history['val'][-1]
        trn_auc = self.auc_history['train'][-1]
        val_auc = self.auc_history['val'][-1]
        txt = ("""Epoch: %d
            Train - Loss: %.3f Auc: %.3f
            Validate - Loss: %.3f Auc: %.3f""" % (self.epoch,
                                              trn_loss, trn_auc, val_loss, val_auc))
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
        self.auc_history[dset_type] = data[:, 2]

    def append_history_to_file(self, dset_type, loss, auc):
        fpath = os.path.join(self.history_dir, dset_type + '.csv')
        with open(fpath, 'a') as f:
            f.write('{},{},{}\n'.format(self.epoch, loss, auc))

    def save_history(self, dset_type, loss, auc):
        self.loss_history[dset_type] = np.append(
            self.loss_history[dset_type], loss)
        self.auc_history[dset_type] = np.append(
            self.auc_history[dset_type], auc)
        self.append_history_to_file(dset_type, loss, auc)

        if dset_type == 'val' and self.is_best_auc(auc):
            self.best_val_auc = auc
            self.best_val_auc_epoch = self.epoch

    def is_best_auc(self, auc):
        return auc > self.best_val_auc

    def save_weights(self, model, trn_loss, val_loss, trn_auc, val_auc, trn_classes_auc, val_classes_auc):
        weights_fname = self.name + '-weights-%d-%.3f-%.3f-%.3f-%.3f.pth' % (
            self.epoch, trn_loss, trn_auc, val_loss, val_auc)
        weights_fpath = os.path.join(self.weights_dir, weights_fname)
        torch.save({
            'last_epoch': self.epoch,
            'trn_loss': trn_loss,
            'val_loss': val_loss,
            'trn_auc': trn_auc,
            'val_auc': val_auc,
            'trn_classes_auc': trn_classes_auc.tolist(),
            'val_classes_auc': val_classes_auc.tolist(),
            'best_val_auc': self.best_val_auc,
            'best_val_auc_epoch': self.best_val_auc_epoch,
            'experiment': self.name,
            'state_dict': model.state_dict()
        }, weights_fpath)
        shutil.copyfile(weights_fpath, self.latest_weights)
        if self.is_best_auc(val_auc):
            self.best_weights_path = weights_fpath

    def load_weights(self, model, fpath):
        self.log("loading weights '{}'".format(fpath))
        state = torch.load(fpath)
        model.load_state_dict(state['state_dict'])
        self.log(
            "loaded weights from experiment %s (last_epoch %d, trn_loss %s, trn_auc %s, val_loss %s, val_auc %s)" % (
                self.name, state['last_epoch'], state['trn_loss'],
                state['trn_auc'], state['val_loss'], state['val_auc']))
        return model, state

    def save_optimizer(self, optimizer, val_auc):
        optim_fname = self.name + '-optim-%d.pth' % (self.epoch)
        optim_fpath = os.path.join(self.weights_dir, optim_fname)
        torch.save({
            'last_epoch': self.epoch,
            'experiment': self.name,
            'state_dict': optimizer.state_dict()
        }, optim_fpath)
        shutil.copyfile(optim_fpath, self.latest_optimizer)
        if self.is_best_auc(val_auc):
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

        trn_epoch, trn_loss, trn_auc = np.split(trn_data, [1, 2], axis=1)
        val_epoch, val_loss, val_auc = np.split(val_data, [1, 2], axis=1)

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

        # AUC-ROC
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plt.plot(trn_epoch, trn_auc, label='Train')
        plt.plot(val_epoch, val_auc, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        ax.set_yscale('log')
        plt.legend()
        auc_fname = os.path.join(self.history_dir, 'auc.png')
        plt.savefig(auc_fname)

        # Combined View - loss-auc.png
        loss_auc_fname = os.path.join(self.history_dir, 'loss-auc.png')
        os.system('convert +append {} {} {}'.format(loss_fname, auc_fname, loss_auc_fname))


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
    output = output.clamp(min = 1e-6, max = 1 - 1e-6)
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


def get_weight(labels):
    label_array = labels.numpy()
    number, count = np.unique(label_array, return_counts=True)
    frequency = dict(zip(number, count))
    P = frequency.get(1, 0)
    N = frequency.get(0, 0)

    if P!=0 and N!=0:
        BP = (P + N)/P
        BN = (P + N)/N
        weights = [BP, BN]
        if use_gpu:
            weights = torch.FloatTensor(weights).cuda()
    else: 
        weights = None 

    return weights


def get_predictions(model_output):
    # Flatten and Get ArgMax to compute accuracy
    val, idx = torch.max(model_output, dim=1)
    return idx.data.cpu().view(-1).numpy()


def get_accuracy(preds, targets):
    correct = np.sum(preds == targets)
    return correct / len(targets)


def get_auc(output, target, average='macro'):
    try:
        auc = roc_auc_score(np.array(target), np.array(output), average=average)
    except:
        auc = -1

    return auc


def train(net, dataloader, criterion, optimizer, epoch=1):
    net.train()
    n_batches = len(dataloader)
    batch_size = dataloader.batch_size
    classes = dataloader.dataset.classes

    total_loss = 0.0
    iterLoss = 0.0
    total_target = []
    total_output = []

    for idx, (inputs, targets) in enumerate(dataloader):
        weights = get_weight(targets)
        inputs = Variable(inputs.cuda(), volatile=False)
        targets = Variable(targets.cuda(), volatile=False)

        ## Forward Pass
        output = net(inputs)

        ## Clear Gradients
        net.zero_grad()

        loss = criterion(output, targets, weights)

        ## Backprop
        loss.backward()
        optimizer.step()

        ## Statics
        iterLoss += loss.data[0]
        total_loss += loss.data[0]

        targets = targets.data.cpu().numpy()
        output = output.data.cpu().numpy()
        for i in range(output.shape[0]):
            total_output.append(output[i].tolist())
            total_target.append(targets[i].tolist())


        if idx % 100 == 0 and idx != 0:
            batch_auc = get_auc(total_output[-100 * batch_size:], total_target[-100 * batch_size:])
            print('Training {:.2f}% Loss: {:.4f} AUC: {:.4f}'.format(100 * idx / n_batches, iterLoss / (100 * batch_size), batch_auc))
            iterLoss = 0

    mean_loss = total_loss / (n_batches * batch_size)
    mean_auc = get_auc(total_output, total_target)

    #Calculate the scores for each class, return a ndarray shape = (n_classes,)
    classes_auc = roc_auc_score(np.array(total_target), np.array(total_output), average=None)

    print('Train Loss: {:.4f} AUC: {:.4f}'.format(mean_loss, mean_auc))
    print()
    for i, c in enumerate(classes):
        print('{}: {:.4f} '.format(c, classes_auc[i]))
    print()

    return mean_loss, mean_auc, classes_auc


def validate(net, dataloader, criterion, epoch=1):
    net.eval()
    val_loss = 0
    iterLoss = 0
    n_batches = len(dataloader)
    classes = dataloader.dataset.classes
    batch_size = dataloader.batch_size

    total_target = []
    total_output = []

    for idx, (inputs, targets) in enumerate(dataloader):
        weights = get_weight(targets)
        inputs = Variable(inputs.cuda(), volatile=True)
        targets = Variable(targets.cuda(), volatile=True)
        output = net(inputs)
        val_loss += criterion(output, targets, weights=weights).data[0]
        iterLoss += criterion(output, targets, weights=weights).data[0]

        targets = targets.data.cpu().numpy()
        output = output.data.cpu().numpy()
        for i in range(output.shape[0]):
            total_output.append(output[i].tolist())
            total_target.append(targets[i].tolist())

        if idx % 100 == 0 and idx != 0:
            batch_auc = get_auc(total_output[-100 * batch_size:], total_target[-100 * batch_size:])
            print('Validation {:.2f}% Loss: {:.4f} AUC: {:.4f}'.format(100 * idx / n_batches, iterLoss / (100 * batch_size), batch_auc))
            iterLoss = 0

    mean_loss = val_loss / (n_batches * batch_size)
    mean_auc = get_auc(total_output, total_target)

    # Calculate the scores for each class, return a ndarray shape = (n_classes,)
    classes_auc = roc_auc_score(np.array(total_target), np.array(total_output), average=None)

    print('Validation Loss: {:.4f} AUC: {:.4f}'.format(mean_loss, mean_auc))
    print()

    for i, c in enumerate(classes):
        print('{}: {:.4f} '.format(c, classes_auc[i]))
    print()

    return mean_loss, mean_auc, classes_auc


def adjust_learning_rate(lr, decay, optimizer, cur_epoch, n_epochs):
    """Sets the learning rate to the initially
        configured `lr` decayed by `decay` every `n_epochs`"""
    new_lr = lr * (decay ** (cur_epoch // n_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def experiment(exp_name, exp_path, exp_model, exp_type):

    normTransform = transforms.Normalize(CXR8_MEAN, CXR8_STD)
    trainTransform = transforms.Compose([transforms.ToTensor()])
    valTransform = transforms.Compose([transforms.ToTensor()])

    train_dataset = CXRDataset(label_path['train'], data_dir, transform=trainTransform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_dataset = CXRDataset(label_path['val'], data_dir, transform=valTransform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    n_classes = len(train_dataset.classes)
    logger = get_logger(ch_log_level=logging.INFO, fh_log_level=logging.INFO)
    model = exp_model(n_classes, logger).cuda()
    criterion = weighted_BCELoss
    optimizer = optim.Adam([{'params': model.transition.parameters()},
                            {'params': model.globalPool.parameters()},
                            {'params': model.prediction.parameters()}], lr=LEARNING_RATE)

    exp = Experiment(exp_name, exp_path, logger)

    assert exp_type in ['new', 'resume']
    if exp_type == 'new':
        # Create New Experiment
        exp.init()
    elif exp_type == 'resume':
        model, optimizer = exp.resume(model, optimizer)


    for epoch in range(exp.epoch, exp.epoch + N_EPOCHS):
        since = time.time()

        ### Train ###
        trn_loss, trn_auc, trn_classes_auc = train(model, train_loader, criterion, optimizer, epoch)
        logger.info('Epoch {:d}: Train - Loss: {:.4f}\tAuc: {:.4f}'.format(epoch, trn_loss, trn_auc))
        time_elapsed = time.time() - since
        logger.info('Train Time {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

        ### Validate ###
        val_loss, val_auc, val_classes_auc = validate(model, val_loader, criterion, epoch)
        logger.info('Epoch {:d}: Validate - Loss: {:.4f}\tAuc: {:.4f}'.format(epoch, val_loss, val_auc))
        time_elapsed = time.time() - since
        logger.info('Total Time {:.0f}m {:.0f}s\n'.format(
            time_elapsed // 60, time_elapsed % 60))

        ### Save Metrics ###
        exp.save_history('train', trn_loss, trn_auc)
        exp.save_history('val', val_loss, val_auc)

        ### Checkpoint ###
        exp.save_weights(model, trn_loss, val_loss, trn_auc, val_auc, trn_classes_auc, val_classes_auc)
        exp.save_optimizer(optimizer, val_loss)

        ### Plot Online ###
        exp.update_viz_loss_plot()
        exp.update_viz_auc_plot()
        exp.update_viz_summary_plot()

        ## Early Stopping ##
        if (epoch - exp.best_val_auc_epoch) > MAX_PATIENCE:
            logger.info(("Early stopping at epoch %d since no "
                         + "better auc found since epoch %.3")
                        % (epoch, exp.best_val_auc_epoch))
            break

        ### Adjust Lr ###
        adjust_learning_rate(LEARNING_RATE, LR_DECAY, optimizer,
                             epoch, DECAY_LR_EVERY_N_EPOCHS)

        exp.epoch += 1



def main():
    experiment(EXPERIMENT_NAME, CXR8_PATH, SE_ResNet50, 'new')


if __name__=="__main__":
    status = main()
    sys.exit(status)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import torch.utils.model_zoo as model_zoo
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from PIL import Image
import time
import os


use_gpu = torch.cuda.is_available
data_dir = "/data/CXR8/images"
save_dir = "./savedModels"
label_path = {'train':"./Train_Label_simple.csv", 'val':"./Val_Label_simple.csv", 'test':"Test_Label_simple.csv"}

class CXRDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform = None):
        self.labels_csv = pd.read_csv(csv_file, header=0)
        self.root_dir = root_dir
        self.transform = transform
        self.classes = pd.read_csv(csv_file, header=None,nrows=1).ix[0, :].as_matrix()
        self.classes = self.classes[1:]

    def __len__(self):
        return len(self.labels_csv)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels_csv.ix[idx, 0])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels_csv.ix[idx, 1:].as_matrix().astype('float')
        label = torch.from_numpy(label).type(torch.FloatTensor)
        sample = {'image': image, 'label': label}

        return sample

def loadData(batch_size):
    trans = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
    image_datasets = {x: CXRDataset(label_path[x], data_dir, transform = trans)for x in ['train', 'val']}
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, num_workers=4)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print('Training data: {}\nValidation data: {}'.format(dataset_sizes['train'], dataset_sizes['val']))

    class_names = image_datasets['train'].classes
    return dataloders, dataset_sizes, class_names

def train_model(model, optimizer, num_epochs=25):
    batch_size = 100
    since = time.time()
    dataloders, dataset_sizes, class_names = loadData(batch_size)
    best_model_wts = model.state_dict()
    best_auc = []
    best_auc_ave = 0


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)


        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            outputList = []
            labelList = []
            logLoss = 0
            # Iterate over data.
            for idx, data in enumerate(dataloders[phase]):
                # get the inputs
                inputs = data['image']
                labels = data['label']

                #calculate weight for loss
                P = 0
                N = 0
                for label in labels:
                    for v in label:
                        if int(v) == 0: N += 1
                        else: P += 1
                try:
                    BP = (P + N)/P
                    BN = (P + N)/N
                    weight = BP/BN
                except:
                    weight = 1.0
                weight = torch.FloatTensor([weight]).cuda()
                #wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                out_data = outputs.data
                criterion = nn.BCELoss(weight=weight)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                #print('running_loss = {}'.format(running_loss))
                labels = labels.data.cpu().numpy()
                out_data = out_data.cpu().numpy()
                for i in range(out_data.shape[0]):
                    outputList.append(out_data[i].tolist())
                    labelList.append(labels[i].tolist())
                
                logLoss += loss.data[0]
                if(idx%(2000/batch_size)==0):
                    try: iterAuc =  roc_auc_score(np.array(labelList[-2000:]),
                                                  np.array(outputList[-2000:]))
                    except: iterAuc = 0
                    print('{} {:.2f}% Loss: {:.4f} AUC: {:.4f}'.format(phase, 100*idx/len(dataloders[phase]), logLoss/((2000/batch_size)*len(data)), iterAuc))
                    logLoss = 0


            #labelList.append([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            #outputList.append([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_auc_ave = roc_auc_score(np.array(labelList), np.array(outputList))
            epoch_auc = roc_auc_score(np.array(labelList), np.array(outputList), average=None)

            print('{} Loss: {:.4f} AUC: {:.4f}'.format(
                phase, epoch_loss, epoch_auc_ave, epoch_auc))
            print()
            for i, c in enumerate(class_names):
                print('{}: {:.4f} '.format(c, epoch_auc[i]))
            print()

            # deep copy the model
            if phase == 'val' and epoch_auc_ave > best_auc_ave:
                best_auc = epoch_auc
                best_auc_ave = epoch_auc_ave
                best_model_wts = model.state_dict()
                saveInfo(model)
           

        print()


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val AUC: {:4f}'.format(best_auc_ave))
    print()
    for i, c in enumerate(class_names):
        print('{}: {:.4f} '.format(c, epoch_auc[i]))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Get number of classes
        train_dataset = CXRDataset(label_path['train'], data_dir, transform=None)
        classes = train_dataset.classes
        self.n_class = len(classes)

        self.model_ft = models.resnet50(pretrained=True)
        self.model_ft = nn.Sequential(*list(self.model_ft.children())[:-2])

        for param in self.model_ft.parameters():
            param.requires_grad = False

        self.transition = nn.Sequential(
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.Conv2d(2048, 2048, kernel_size=1, padding=0, stride=1, bias=False),
        )
        self.globalPool = nn.Sequential(
            nn.AvgPool2d(7)
        )
        self.prediction = nn.Sequential(
            nn.Linear(2048, self.n_class),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # input x: (bs,3,224,224)
        #print('input x size:{}'.format(x.size()))
        x = self.model_ft(x) # (bs,2048,7,7)
        #print('model_ft(x):{}'.format(x.size()))
        x = self.transition(x) # (bs,2048,7,7)
        #print('transition(x):{}'.format(x.size()))
        x = self.globalPool(x) #(bs,2048,1,1)
        #print('globalPool(x):{}'.format(x.size()))
        x = x.view(x.size(0), -1) #(bs,2048)
        #print('x.view:{}'.format(x.size()))
        x = self.prediction(x)   #(bs,9)
        #print('prediction(x):{}'.format(x.size()))
        return x

def saveInfo(model):
    #save model
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model.state_dict(), os.path.join(save_dir, "resnet50_simple.pth"))

if __name__ == '__main__':
    try:
        model = Model()
        model.load_state_dict(torch.load(os.path.join(save_dir, "resnet50_simple.pth")))
        print('\nUsing previous model')
    except:
        model = Model()

    if use_gpu:
        model = model.cuda()

    optimizer = optim.Adam([
            {'params':model.transition.parameters()},
            {'params':model.globalPool.parameters()},
            {'params':model.prediction.parameters()}],
            lr=1e-3)

    model = train_model(model, optimizer, num_epochs = 10)
    saveInfo(model)


# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 01:50:46 2019

@author: VR LAB PC3
"""

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
#import torchvision
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
from ft_ResNet50 import model, random_erasing
#import ft_net, ft_net_dense, PCB
#from random_erasing import RandomErasing
import json
from shutil import copyfile
from config import cfg

import network
import dataset

version =  torch.__version__

#######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='newGAN', type=str, help='output model name')
#parser.add_argument('--data_dir',default='../Market/pytorch',type=str, help='training dir path')
parser.add_argument('--train_all', default=True, action='store_true', help='use all training data' )
parser.add_argument('--color_jitter', default=True, action='store_true', help='use color jitter in training' )
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--erasing_p', default=0.2, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--use_NAS', action='store_true', help='use NAS' )
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--droprate', default=0.3, type=float, help='drop rate')
parser.add_argument('--PCB', action='store_true', help='use PCB+ResNet50' )
parser.add_argument('--fp16', action='store_true', help='use float16 instead of float32, which will save about 50% memory' )
opt = parser.parse_args()

fp16 = opt.fp16
#data_dir = opt.data_dir
data_dir = 'C:\\Users\\VR LAB PC3\\Desktop\\Y\\Datasets\\Market-1501-v15.09.15\\pytorch'
name = opt.name
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >=0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True
#%%#####################################################################
# Load Data
# ---------
#

transform_train_list = [
        #transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((256,128), interpolation=3),
        transforms.Pad(10),
        transforms.RandomCrop((256,128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

transform_val_list = [
        transforms.Resize(size=(256,128),interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

#if opt.PCB:
#    transform_train_list = [
#        transforms.Resize((384,192), interpolation=3),
#        transforms.RandomHorizontalFlip(),
#        transforms.ToTensor(),
#        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#        ]
#    transform_val_list = [
#        transforms.Resize(size=(384,192),interpolation=3), #Image.BICUBIC
#        transforms.ToTensor(),
#        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#        ]

if opt.erasing_p>0:
    transform_train_list = transform_train_list +  [random_erasing.RandomErasing(probability = opt.erasing_p, mean=[0.0, 0.0, 0.0])]

if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list

print(transform_train_list)
data_transforms = {
    'train': transforms.Compose( transform_train_list ),
    'val': transforms.Compose(transform_val_list),
}


train_all = ''
if opt.train_all:
     train_all = '_all'

image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train' + train_all),
                                          data_transforms['train'])
image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                          data_transforms['val'])

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=True, num_workers=0, pin_memory=True) # 8 workers may work faster
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()

since = time.time()
inputs, classes = next(iter(dataloaders['train']))
print(time.time()-since)
######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.

y_loss = {} # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []


#%% Own training code starts here

# Load Network
def load_network(model_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.TEST.GPU_ID
    print ('###################################')
    print ("#####      Load Network      #####")
    print ('###################################')

    nets = []
    netG = network.Res_Generator(cfg.TRAIN.ngf, cfg.TRAIN.num_resblock)
    netG.load_state_dict(torch.load(model_path)['state_dict'])

    nets.append(netG)
    for net in nets:
        net.cuda()

    print ('Finished !')
    return nets


model_path = 'model/GAN/G_16.pkl'
nets = load_network(model_path)
netG = nets[0]
for param in netG.parameters():
    param.requires_grad = False 
netG.eval()


model_1 = model.ft_net(len(class_names), opt.droprate, opt.stride)
model_1.load_state_dict(torch.load('ft_ResNet50/net_last_resnet.pth'))

temp = nn.Sequential(*list(model_1.children())[:-1])
res50_conv = nn.Sequential(*list(temp[0].children())[:-1])
#model to GPU
res50_conv = res50_conv.cuda()
for param in res50_conv.parameters():
    param.requires_grad = False    
res50_conv.eval()


class FusionNet(nn.Module):
    def __init__(self, num_classes):
        super(FusionNet, self).__init__()
        self.fc1 = nn.Linear(2048*9, 2048*4)
        self.fc2 = nn.Linear(2048*4, 2048)
        self.fc3 = nn.Linear(2048, 2048)
        self.fc4 = nn.Linear(2048, num_classes)

    def forward(self, x_q, x_gen):
        x = torch.cat((x_q, x_gen), dim=1)
        x = x.view(-1, 2048*9)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x_q = x_q.view(-1, 2048)
        x = x_q + x
        x_f = self.fc3(x)
        x_c = self.fc4(x_f)
        return x_c

net = FusionNet(len(class_names))
net = net.cuda()

train_transform = transforms.Compose([
        transforms.Resize((256, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

#best_model_wts = model.state_dict()
#best_acc = 0.0
#%%
#def train_model(model, criterion, optimizer, scheduler, num_epochs, train_transform):
#model = net
since = time.time()
pose_path = 'C:\\Users\\VR LAB PC3\\Desktop\\Y\\github_models\\pngan-fork\\PN_GAN\\script\\GAN\\cannonical_poses'
poses = os.listdir(pose_path)

x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")
def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig( os.path.join('./model',name,'train.jpg'))

######################################################################
# Save model
#---------------------------
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join('./model',name,save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda(gpu_ids[0])

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, weight_decay=5e-4, momentum=0.9, nesterov=True)
scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
num_epochs=60

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            scheduler.step()
            net.train(True)  # Set model to training mode
        else:
            net.train(False)  # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0.0
        # Iterate over data.
        for data in dataloaders[phase]:
            # get the inputs
            inputs, labels = data
#            print(inputs.shape)
            now_batch_size,c,h,w = inputs.shape
            if now_batch_size<opt.batchsize: # skip the last batch
                continue
            #print(inputs.shape)
            # wrap them in Variable
            if use_gpu:
                inputs = Variable(inputs.cuda().detach())
                labels = Variable(labels.cuda().detach())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            # if we use low precision, input also need to be fp16
            if fp16:
                inputs = inputs.half()
                
            #% assume fusionnet and gan is imported
            inputs_vec = []
            for input_i in inputs:
                input_i = input_i[None,:,:,:]
#                print('input_i shape:', input_i.shape)
                fake_img = []
                for pose in poses:
                    pose_img = Image.open(os.path.join(pose_path, pose)).convert('RGB')
                    pose = Variable(train_transform(pose_img).cuda().detach())
                    pose = pose[None,:,:,:]
#                    print('pose dim:', pose.shape)
                    fake_img.append(netG(input_i, pose)[-1,:,:,:])
                fake_img_tensor = Variable(torch.stack(fake_img).cuda().detach())
                inputs_vec.append(res50_conv(fake_img_tensor).view(-1))
#            print(inputs_vec.shape)
            input_train = Variable(torch.stack(inputs_vec).cuda().detach())
            inputs = res50_conv(inputs)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            if phase == 'val':
                with torch.no_grad():
                    outputs = net(inputs, input_train[:,:,None,None])
            else:
                outputs = net(inputs, input_train[:,:,None,None])

            if not opt.PCB:
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
            else:
                part = {}
                sm = nn.Softmax(dim=1)
                num_part = 6
                for i in range(num_part):
                    part[i] = outputs[i]

                score = sm(part[0]) + sm(part[1]) +sm(part[2]) + sm(part[3]) +sm(part[4]) +sm(part[5])
                _, preds = torch.max(score.data, 1)

                loss = criterion(part[0], labels)
                for i in range(num_part-1):
                    loss += criterion(part[i+1], labels)

            # backward + optimize only if in training phase
            if phase == 'train':
                if fp16: # we use optimier to backward loss
                    optimizer.backward(loss)
                else:
                    loss.backward()
                optimizer.step()

            # statistics
            if int(version[0])>0 or int(version[2]) > 3: # for the new version like 0.4.0, 0.5.0 and 1.0.0
                running_loss += loss.item() * now_batch_size
            else :  # for the old version like 0.3.0 and 0.3.1
                running_loss += loss.data[0] * now_batch_size
            running_corrects += float(torch.sum(preds == labels.data))

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects / dataset_sizes[phase]
        
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))
        
        y_loss[phase].append(epoch_loss)
        y_err[phase].append(1.0-epoch_acc)            
        # deep copy the model
        if phase == 'val':
            last_model_wts = net.state_dict()
            if epoch%10 == 9:
                save_network(net, epoch)
            draw_curve(epoch)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print()

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
#print('Best val Acc: {:4f}'.format(best_acc))

# load best model weights
net.load_state_dict(last_model_wts)
save_network(net, 'last')
#return model

######################################################################
# Draw Curve
#---------------------------

#model = train_model(net, criterion, optimizer, scheduler, 60, train_transform)

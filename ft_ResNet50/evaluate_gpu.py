# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 01:50:01 2019

@author: VR LAB PC3
"""

import torch
import torch.nn as nn
from torchvision import transforms, utils
import torch.utils.data as Data
from torch.autograd import Variable
from torch.optim import lr_scheduler
#from config import cfg
from tensorboardX import SummaryWriter
#import os, itertools
#import network_1
#import dataset
import time
import os
# import matplotlib.pyplot as plt
from model import ft_net, ft_net_dense # , PCB
import sys
#import logger
import random
import numpy as np
import scipy.io
from PIL import Image
from torchvision import datasets, models, transforms


#def load_network(class_names, droprate, stride):
#    
#    model = ft_net(len(class_names)-1, droprate, stride)
#    model.load_state_dict(torch.load('net_last.pth'))    
#    # model to gpu
##    model = model.cuda()    
#    temp = nn.Sequential(*list(model.children())[:-1])
#    res50_conv = nn.Sequential(*list(temp[0].children())[:-1])    
#    #for param in res50_conv.parameters():
#    #    param.requires_grad = False    
#    res50_conv.eval()
#        
#    return res50_conv

data_transforms = transforms.Compose([
    transforms.Resize((256,128), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#%%
import torch.backends.cudnn as cudnn
cudnn.benchmark = False

multi_query = False
batchsize = 8
PCB = False
    
data_dir = 'C:\\Users\\VR LAB PC3\\Desktop\\Y\\Datasets\\Market-1501-v15.09.15\\pytorch'

if multi_query:
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query','multi-query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batchsize,
                                             shuffle=False, num_workers=0) for x in ['gallery','query','multi-query']}
else:
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batchsize,
                                             shuffle=False, num_workers=0) for x in ['gallery','query']}
class_names = image_datasets['query'].classes
use_gpu = torch.cuda.is_available()

######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        print(count)
        ff = torch.FloatTensor(n,2048).zero_()

        if PCB:
            ff = torch.FloatTensor(n,2048,6).zero_() # we have six parts
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            #if opt.fp16:
            #    input_img = input_img.half()
            outputs = model(input_img) 
            f = outputs.data.cpu().float()
            ff = ff+f
        # norm feature
        if PCB:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6) 
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff), 0)
    return features

def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        #filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels

gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs

gallery_cam,gallery_label = get_id(gallery_path)
query_cam,query_label = get_id(query_path)

if multi_query:
    mquery_path = image_datasets['multi-query'].imgs
    mquery_cam,mquery_label = get_id(mquery_path)
    
#%%
train_all = '_all'
image_datasets_train = {}
image_datasets_train['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train' + train_all), data_transforms)    
class_names_train = image_datasets_train['train'].classes
   
model = ft_net(len(class_names_train), 0.2, 2)
model.load_state_dict(torch.load('net_last.pth'))    
  
temp = nn.Sequential(*list(model.children())[:-1])
res50_conv = nn.Sequential(*list(temp[0].children())[:-1])    
#for param in res50_conv.parameters():
#    param.requires_grad = False    
res50_conv.eval()
model = res50_conv

if use_gpu:
    model = model.cuda()

# Extract feature
with torch.no_grad():
    gallery_feature = extract_feature(model,dataloaders['gallery'])
    query_feature = extract_feature(model,dataloaders['query'])
    if multi_query:
        mquery_feature = extract_feature(model,dataloaders['multi-query'])
    
# Save to Matlab for check
result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,'query_f':query_feature.numpy(),'query_label':query_label,'query_cam':query_cam}
scipy.io.savemat('pytorch_result_resnet50_pretrain.mat',result)
if multi_query:
    result = {'mquery_f':mquery_feature.numpy(),'mquery_label':mquery_label,'mquery_cam':mquery_cam}
    scipy.io.savemat('multi_query.mat',result)

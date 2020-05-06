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
import scipy.io
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
parser.add_argument('--batchsize', default=512, type=int, help='batchsize')
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
    
    
#%%

transform_val_list = [
        transforms.Resize(size=(256,128),interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

data_transforms = transforms.Compose(transform_val_list)

train_all = ''
if opt.train_all:
     train_all = '_all'

image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=0) for x in ['gallery','query']}

class_names = image_datasets['query'].classes
use_gpu = torch.cuda.is_available()

since = time.time()
#inputs, classes = next(iter(dataloaders['train']))
print(time.time()-since)


#%% Own model loading code starts here

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


model_1 = model.ft_net(len(class_names)+1, opt.droprate, opt.stride)
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
        return x_f, x_c

fusionnet = FusionNet(len(class_names)+1)
fusionnet.load_state_dict(torch.load('C:\\Users\\VR LAB PC3\\Desktop\\Y\\github_models\\pngan-fork\\PN_GAN\\script\\GAN\\model\\newGAN\\net_last.pth'))
#fusionnet = nn.Sequential(*list(fusionnet.children())[:-1])
fusionnet = fusionnet.cuda()

since = time.time()


#%% extraction code from layumi baseline
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders, name):
#    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        print(count)
        if opt.use_dense:
            ff = torch.FloatTensor(n,1024).zero_()
        else:
            ff = torch.FloatTensor(n,2048).zero_()
        if opt.PCB:
            ff = torch.FloatTensor(n,2048,6).zero_() # we have six parts
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            outputs = model(input_img) 
            f = outputs.data.cpu()
            ff = ff+f
    
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

#        features = torch.cat((features,ff), 0)
        save_filename = str(count)+'.pth'
        save_path = os.path.join('./features',name,save_filename)
        torch.save(ff, save_path)
    return 0

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

#%%
pose_path = 'C:\\Users\\VR LAB PC3\\Desktop\\Y\\github_models\\pngan-fork\\PN_GAN\\script\\GAN\\cannonical_poses'
poses = os.listdir(pose_path)

with torch.no_grad():
    count = 0
    print(count)
    name = 'gallery'
    for data in dataloaders[name]:
        # get the inputs
        inputs, labels = data
#            print(inputs.shape)
        n,c,h,w = inputs.shape
        if use_gpu:
            inputs = Variable(inputs.cuda().detach())
            labels = Variable(labels.cuda().detach())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        
        ff = torch.FloatTensor(n,2048).zero_()
        
        inputs_vec = []
        for input_i in inputs:
            input_i = input_i[None,:,:,:]
    #                print('input_i shape:', input_i.shape)
            fake_img = []
            for pose in poses:
                pose_img = Image.open(os.path.join(pose_path, pose)).convert('RGB')
                pose = Variable(data_transforms(pose_img).cuda().detach())
                pose = pose[None,:,:,:]
    #                    print('pose dim:', pose.shape)
                fake_img.append(netG(input_i, pose)[-1,:,:,:])
            fake_img_tensor = Variable(torch.stack(fake_img).cuda().detach())
            inputs_vec.append(res50_conv(fake_img_tensor).view(-1))
    #            print(inputs_vec.shape)
        input_train = Variable(torch.stack(inputs_vec).cuda().detach())
        inputs = res50_conv(inputs)
        
        outputs, _ = fusionnet(inputs, input_train[:,:,None,None])
        ff = ff + outputs.data.cpu()
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        
        count += n
        print(count)

#        features = torch.cat((features,ff), 0)
        save_filename = str(count-n)+'to'+str(count)+'.pth'
        save_path = os.path.join('./features',name,save_filename)
        torch.save(ff, save_path)
        
#%%
#import glob
search_dir_query = 'C:\\Users\\VR LAB PC3\\Desktop\\Y\\github_models\\pngan-fork\\PN_GAN\\script\\GAN\\features\\query'
os.chdir(search_dir_query)
files_query = filter(os.path.isfile, os.listdir(search_dir_query))
files_query = [os.path.join(search_dir_query, f) for f in files_query] # add path to each file
files_query.sort(key=lambda x: os.path.getmtime(x))
    
# query feature
query_feature = torch.FloatTensor()
for item in files_query:
    query_feature = torch.cat((query_feature, torch.load(item)),0)

search_dir_gallery = 'C:\\Users\\VR LAB PC3\\Desktop\\Y\\github_models\\pngan-fork\\PN_GAN\\script\\GAN\\features\\gallery'
os.chdir(search_dir_gallery)
files_gallery = filter(os.path.isfile, os.listdir(search_dir_gallery))
files_gallery = [os.path.join(search_dir_gallery, f) for f in files_gallery] # add path to each file
files_gallery.sort(key=lambda x: os.path.getmtime(x))
    
# gallery_feature
gallery_feature = torch.FloatTensor()
for item in files_gallery:
    gallery_feature = torch.cat((gallery_feature, torch.load(item)),0)
    
#    gallery_feature = extract_feature(model,dataloaders['gallery'], name='gallery')
#    query_feature = extract_feature(model,dataloaders['query'], name='query')

# Save to Matlab for check
result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,'query_f':query_feature.numpy(),'query_label':query_label,'query_cam':query_cam}
scipy.io.savemat('pytorch_result_fullmodel.mat',result)
#if opt.multi:
#    result = {'mquery_f':mquery_feature.numpy(),'mquery_label':mquery_label,'mquery_cam':mquery_cam}
#    scipy.io.savemat('multi_query.mat',result)
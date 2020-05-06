# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 17:29:09 2019

@author: VR LAB PC3
"""

import os

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt


num_epochs = 100
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def read_file(path):
    return torch.tensor(np.load(path)/128.0)

train_loader = torch.utils.data.DataLoader(
    datasets.DatasetFolder('C:\\Users\\VR LAB PC3\\Desktop\\Y\\my_model\\pose_train',
                            loader=read_file, extensions=['npy'], transform=None),
    batch_size=batch_size, shuffle=True)


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(50, 32),
            nn.ReLU(True),
            nn.Linear(32, 16),
            nn.ReLU(True), 
            nn.Linear(16, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(True),
            nn.Linear(16, 32),
            nn.ReLU(True),
            nn.Linear(32, 50),
            nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
    for data in train_loader:
        img, _ = data
        img = img.view(img.size(0), -1)
        img = Variable(img).cuda()
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}' .format(epoch + 1, num_epochs, loss.item()))
#    if epoch % 10 == 0:
#        pic = to_img(output.cpu().data)
#        save_image(pic, './results/image_{}.png'.format(epoch))

torch.save(model.state_dict(), './sim_autoencoder.pth')


#%%
def plot_pose(pose_vector):
    if np.max(pose_vector) <= 64 :
        pose_vector = pose_vector*128
    
    pose_coord = []

    for i in range(int(len(pose_vector)/2)):
        x = pose_vector[i*2]
        y = pose_vector[i*2+1]
        pose_coord.append(np.asarray([x,y]))
    #    plt.scatter(x,y)
    
    connections = [[0,15],[0,16],[16,18],[15,17],[0,1],[1,2],[1,5],[5,6],[6,7],[2,3],[3,4],[1,8],[8,9],
                   [9,10],[10,11],[11,24],[11,22],[23,22],[8,12],[12,13],[13,14],[14,21],[14,19],[19,20]]
    
    plt.figure(figsize=(4,8))
    for i, point in enumerate(connections):
    #    print(i,point)
        if np.sum(pose_coord[point[0]])!=0 and np.sum(pose_coord[point[1]])!=0:
            x = [ pose_coord[point[0]][0], pose_coord[point[1]][0] ]
            y = [ 128 - pose_coord[point[0]][1], 128 - pose_coord[point[1]][1] ]
            plt.plot(x,y, 'ro-')    
    plt.xlim((0,64))
    plt.ylim((0,128))
    plt.show()


with torch.no_grad():    
    sample_1 = np.load('C:\\Users\\VR LAB PC3\\Desktop\\Y\\my_model\\pose_train\\all\\0002_c1s1_000551_01.npy')
    print(sample_1.shape)
    plot_pose(sample_1)
    
    sample_1 = torch.Tensor(sample_1).cuda()
#    x1, x2, s1_mu, s1_logvar = model.encode(sample_1.view(-1, 50))
#    std = torch.exp(0.5*s1_logvar)
#    eps = torch.randn_like(std)
#    mu + eps*std
    re_sample_1 = np.asarray(model(sample_1).cpu())
#    re_sample_1 = np.asarray(model.decode(s1_mu+eps*std).cpu())
    print(re_sample_1.shape)
    plot_pose(re_sample_1.T)
    
    print('sample_1\n', sample_1)
    print('re_sample_1\n', 128*re_sample_1)
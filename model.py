# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:52:04 2019

@author: VR LAB PC3
"""

#%%  Loading the pretrained ResNet50 model upto the avgpool layer
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable

import network


def print_networks(model_names, debug):
        print ('---------------- Network initialized ----------------')
        names = ['netG', 'netD']
        for i, net in enumerate(model_names):
            num_params = 0
            for param in net.parameters():
                num_params += param.numel()
            if debug:
                print ('=========== %s ===========' % names[i])
                print (net)
            print ('[Network %s] Total number of parameters: %.3f M' % (names[i], num_params / 1e6))
        print ('-----------------------------------------------------')


netG = network.Res_Generator(ngf=64, nz=(2048+50))

netD = network.DC_Discriminator(ndf=64)

netRN = network.ResNet50()

netE = network.Ensemble(netRN, netG)

nets = []
nets.append(netE)
nets.append(netD)

print_networks(nets, debug=True)

for net in nets:
    net.cuda()

#from torchsummary import summary
#summary(resnet50, input_size=(3, 224, 224))

#for p in resnet50.parameters():
#    p.requires_grad = True
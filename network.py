# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:52:37 2019

@author: VR LAB PC3
"""

import torch
import cv2, math
import torch.nn as nn
from torch.nn import init
import torch.utils.model_zoo as model_zoo
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from collections import OrderedDict                                                                                         


#def weights_init_kaiming(m):
#    classname = m.__class__.__name__
#    # print(classname)
#    if classname.find('Conv') != -1:
#        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
#    elif classname.find('Linear') != -1:
#        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
#        init.constant(m.bias.data, 0.0)
#    elif classname.find('BatchNorm1d') != -1:
#        init.normal(m.weight.data, 1.0, 0.02)
#        init.constant(m.bias.data, 0.0)
#
#
#def weights_init_classifier(m):
#    classname = m.__class__.__name__
#    if classname.find('Linear') != -1:
#        init.normal(m.weight.data, std=0.001)
#        init.constant(m.bias.data, 0.0)


class ResBlock(nn.Module):
    def __init__(self, ncf, use_bias=False):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(ncf, ncf, kernel_size=3, stride=1, padding=1, bias=use_bias)),
            ('bn', nn.InstanceNorm2d(ncf)),
            ('leakyrelu', nn.LeakyReLU(0.1,inplace=True)),
        ]))
        self.conv2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(ncf, ncf, kernel_size=3, stride=1, padding=1, bias=use_bias)),
            ('bn', nn.InstanceNorm2d(ncf)),
        ]))
        self.relu = nn.LeakyReLU(0.1,inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + x
        out = self.relu(out)

        return out


class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x



class Res_Generator(nn.Module):
    def __init__(self, ngf, nz, num_resblock=9):
        super(Res_Generator, self).__init__()
        self.prepare = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 6, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 6),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ngf*6) x 8 x 8
            nn.ConvTranspose2d( ngf * 6, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d( ngf, ngf, 4, 2, 1, bias=False),
            # state size. (ngf) x 128 x 128
        )
        
        self.interpolate = Interpolate(size=(256,128), mode='bilinear')

        self.conv1 = nn.Sequential(OrderedDict([
            ('pad', nn.ReflectionPad2d(3)),
            ('conv', nn.Conv2d(ngf, ngf, kernel_size=7, stride=1, padding=0, bias=True)),
            ('bn', nn.InstanceNorm2d(ngf)),
            ('relu', nn.LeakyReLU(0.1,inplace=True)),
        ]))
        self.conv2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=1, bias=True)),
            ('bn', nn.InstanceNorm2d(ngf*2)),
            ('relu', nn.LeakyReLU(0.1,inplace=True)),
        ]))
        self.conv3 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1, bias=True)),
            ('bn', nn.InstanceNorm2d(ngf*4)),
            ('relu', nn.LeakyReLU(0.1,inplace=True)),
        ]))

        self.num_resblock = num_resblock
        for i in range(num_resblock):
            setattr(self, 'res'+str(i+1), ResBlock(ngf*4, use_bias=True))

        self.deconv3 = nn.Sequential(OrderedDict([
            ('deconv', nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)),
            ('bn', nn.InstanceNorm2d(ngf*2)),
            ('relu', nn.LeakyReLU(0.1, inplace=True))
        ]))
        self.deconv2 = nn.Sequential(OrderedDict([
            ('deconv', nn.ConvTranspose2d(ngf*2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)),
            ('bn', nn.InstanceNorm2d(ngf)),
            ('relu', nn.LeakyReLU(0.1, inplace=True))
        ]))
        self.deconv1 = nn.Sequential(OrderedDict([
            ('pad', nn.ReflectionPad2d(3)),
            ('conv', nn.Conv2d(ngf, 3, kernel_size=7, stride=1, padding=0, bias=False)),
            ('tanh', nn.Tanh())
        ]))

    def forward(self, z):
        x = self.prepare(z)
        x = self.interpolate(x)
#        print('my_layer:',x.size())
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        for i in range(self.num_resblock):
            res = getattr(self, 'res'+str(i+1))
            x = res(x)
        x = self.deconv3(x)
        x = self.deconv2(x)
        x = self.deconv1(x)

        return x


class DC_Discriminator(nn.Module):
    def __init__(self, ndf, num_att=751):
        super(DC_Discriminator, self).__init__()
        self.conv1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(3, ndf, kernel_size=4, stride=2, padding=1, bias=False)),
            ('relu', nn.LeakyReLU(0.2, True))
        ]))
        self.conv2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1, bias=True)),
            ('bn', nn.InstanceNorm2d(ndf*2)),
            ('relu', nn.LeakyReLU(0.2, True))
        ]))
        self.conv3 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1, bias=True)),
            ('bn', nn.InstanceNorm2d(ndf*4)),
            ('relu', nn.LeakyReLU(0.2, True))
        ]))
        self.conv4 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1, bias=True)),
            ('bn', nn.InstanceNorm2d(ndf*8)),
            ('relu', nn.LeakyReLU(0.2, True))
        ]))
        self.conv5 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(ndf*8, ndf*8, kernel_size=4, stride=2, padding=1, bias=True)),
            ('bn', nn.InstanceNorm2d(ndf*8)),
            ('relu', nn.LeakyReLU(0.2, True))
        ]))
        self.conv6 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(ndf*8, ndf*8, kernel_size=4, stride=1, padding=1, bias=True)),
            ('bn', nn.InstanceNorm2d(ndf*8)),
            ('relu', nn.LeakyReLU(0.2, True))
        ]))
        self.dis = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(ndf*8, 1, kernel_size=1, stride=1, padding=0, bias=False)),
#            ('fc1', nn.Linear(3*7*1, 1)),
#            ('drop1', nn.Dropout(0.2)),
#            ('relu', nn.ReLU(True)),
#            ('fc2', nn.Linear(1024, 1))
        ]))
        self.att = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(3*7*ndf*8, 1024)),
            ('relu', nn.ReLU(True)),
            ('fc2', nn.Linear(1024, num_att))
        ]))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        dis = self.dis(x)
#        print (dis.size())
        x = x.view(x.size(0), -1)
#        print (x.size())
        att = self.att(x)

        return dis, att


#class Patch_Discriminator(nn.Module):
#    def __init__(self, ndf):
#        super(Patch_Discriminator, self).__init__()
#        self.conv1 = nn.Sequential(OrderedDict([
#            ('conv', nn.Conv2d(3, ndf, kernel_size=4, stride=2, padding=1, bias=False)),
#            ('relu', nn.LeakyReLU(0.2, True))
#        ]))
#        self.conv2 = nn.Sequential(OrderedDict([
#            ('conv', nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1, bias=True)),
#            ('bn', nn.InstanceNorm2d(ndf*2)),
#            ('relu', nn.LeakyReLU(0.2, True))
#        ]))
#        self.conv3 = nn.Sequential(OrderedDict([
#            ('conv', nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1, bias=True)),
#            ('bn', nn.InstanceNorm2d(ndf*4)),
#            ('relu', nn.LeakyReLU(0.2, True))
#        ]))
#        self.conv4 = nn.Sequential(OrderedDict([
#            ('conv', nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=1, padding=0, bias=True)),
#            ('bn', nn.InstanceNorm2d(ndf*8)),
#            ('relu', nn.LeakyReLU(0.2, True))
#        ]))
#        self.dis = nn.Sequential(OrderedDict([
#            ('conv', nn.Conv2d(ndf*8, 1, kernel_size=4, stride=1, padding=0, bias=False)),
#        ]))
#
#    def forward(self, x):
#        x = self.conv1(x)
#        x = self.conv2(x)
#        x = self.conv3(x)
#        x = self.conv4(x)
#        dis = self.dis(x).squeeze()
#
#        return dis
    
    
    
class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        net = torchvision.models.resnet50(pretrained=True)
        net.avgpool = nn.AdaptiveAvgPool2d((1,1))
        net.fc = nn.Sequential()
        self.net = net
        
    def forward(self, x):
        x = self.net(x)
        fea = x.view(x.size(0), -1)
        return fea
    
    
class Ensemble(nn.Module):
    def __init__(self, ResNet50, Generator):
        super(Ensemble, self).__init__()
        self.ResNet50 = ResNet50
        self.Generator = Generator
        
    def forward(self, x1, x2):
        x1 = self.ResNet50(x1)
        fea2 = x2.view(x2.size(0), -1)
#        print(x1.size())
#        print(fea2.size())
        x = torch.cat((x1, fea2), dim=1)
        x = x[:,:,None,None]
#        print(x.size())
        x = self.Generator(x)
        return x



#class ResNet50(nn.Module):
#    def __init__(self, num_class=751):
#        super(ResNet50, self).__init__()
#        net = torchvision.models.resnet50(pretrained=True)
#        net.avgpool = nn.AdaptiveAvgPool2d((1,1))
#        net.fc = nn.Sequential()
#        self.net = net
#
#        fc = []
#        num_bottleneck = 512
#        fc += [nn.Linear(2048, num_bottleneck)]
#        fc += [nn.BatchNorm1d(num_bottleneck)]
#        fc += [nn.ReLU(inplace=True)]
#        fc += [nn.Dropout(p=0.5)]
#        fc = nn.Sequential(*fc)
#        fc.apply(weights_init_kaiming)
#        self.fc = fc
#
#        classifier = []
#        classifier += [nn.Linear(num_bottleneck, num_class)]
#        classifier = nn.Sequential(*classifier)
#        classifier.apply(weights_init_classifier)
#        self.classifier = classifier
#
#    def forward(self, x, test=False):
#        x = self.net(x)
#        fea = x.view(x.size(0), -1)
#        out = self.fc(fea)
#        out = self.classifier(out)
#
#        if test:
#            return fea
#        else:
#            return out
        
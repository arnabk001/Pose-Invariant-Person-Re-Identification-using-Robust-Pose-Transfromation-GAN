# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:37:41 2019

@author: VR LAB PC3
"""

import os, random
from PIL import Image
import cv2, math
from torchvision import transforms
import numpy as np
import torch
from config import cfg
import matplotlib.pyplot as plt
from itertools import permutations as P
import torchvision.transforms.functional as F


def train_transform():
    img = transforms.Compose([
        transforms.Resize((256, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    return img


def val_transform():
    img = transforms.Compose([
        transforms.Resize((256, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5, 0.5, 0.5))
    ])
    return img

# add random erasing
def train_loader(path):

    img = Image.open(path).convert('RGB')
    
    return img

#    # add random noise
#    if random.uniform(0,1) > 0.5:
#        return img

#    sl = 0.02
#    sh = 0.4
#    r1 = 0.3
#    # mean=[0.4914, 0.4822, 0.4465]
#    mean = [0.0, 0.0, 0.0]
#    img = np.array(img)
#    for attempt in range(100):
#        area = img.shape[0] * img.shape[1]
#
#        target_area = random.uniform(sl, sh) * area
#        aspect_ratio = random.uniform(r1, 1/r1)
#
#        h = int(round(math.sqrt(target_area * aspect_ratio)))
#        w = int(round(math.sqrt(target_area / aspect_ratio)))
#
#        if w < img.shape[1] and h < img.shape[0]:
#            x1 = random.randint(0, img.shape[0] - h)
#            y1 = random.randint(0, img.shape[1] - w)
#            if img.shape[2] == 3:
#                img[0, x1:x1+h, y1:y1+w] = mean[0]
#                img[1, x1:x1+h, y1:y1+w] = mean[1]
#                img[2, x1:x1+h, y1:y1+w] = mean[2]
#            else:
#                img[0, x1:x1+h, y1:y1+w] = mean[0]
#            return Image.fromarray(np.uint8(img))
#
#    return Image.fromarray(np.uint8(img))


def val_loader(path):
    img = Image.open(path).convert('RGB')
#    im4 = img.resize((224, 224), Image.BICUBIC)
    return img

def pose_loader(path):
    pose = np.reshape(np.load(path),(1,1,50))
#    pose = transforms.ToPILImage()(pose)
#    print(pose)
    return pose


class Market_DataLoader():
    def __init__(self, imgs_path, pose_path, idx_path, transform, img_loader, pose_loader):
        # train/test index
        lines = open(idx_path, 'r').readlines()
        idx = [int(line.split()[0]) for line in lines]
        
        classes = idx
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
                
        # images
        images = os.listdir(imgs_path)
        poses = os.listdir(pose_path)
        images = [im for im in images if int(im.split('_')[0]) in idx and im[:-3]+'npy' in poses]

        # pairs
        data = []
        for i in idx:   # i being the class
            tmp = [j for j in images if int(j.split('_')[0]) == i]
            data += list(P(tmp, 2))

        random.shuffle(data)
        # self.data = data[0:5000*cfg.TRAIN.BATCH_SIZE]
        self.data = data
        self.imgs_path = imgs_path
        self.pose_path = pose_path
        self.img_loader = img_loader
        self.pose_loader = pose_loader
        self.transform = transform
        
        self.class_to_idx = class_to_idx

    def __getitem__(self, index):
        src_path, tgt_path = self.data[index]

        src_img = self.img_loader(os.path.join(self.imgs_path, src_path))
        tgt_img = self.img_loader(os.path.join(self.imgs_path, tgt_path))
        pose    = self.pose_loader(os.path.join(self.pose_path, tgt_path[:-3]+'npy'))
        label   = self.class_to_idx[int(src_path.split('_')[0])]
        
        pose_transform = transforms.Compose([
#                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                ])

        src_img, tgt_img, pose = self.transform(src_img), self.transform(tgt_img), pose_transform(pose)
    
        
#        print(pose)

        return src_img, tgt_img, pose, label

    def __len__(self):
        return len(self.data)


class Market_test():
    def __init__(self, imgs_name, imgs_path, pose_path, transform, img_loader, pose_loader):
        # images
        images = imgs_name
        poses = os.listdir(pose_path)

        # pairs
        data = []
        for i in images:
            for j in poses:
                name = i.split('.')[0] + '_to_' + j.split('.')[0]
                data.append([i, j, name])

        self.data = data
        self.imgs_path = imgs_path
        self.pose_path = pose_path
        self.img_loader = img_loader
        self.pose_loader = pose_loader
        self.transform = transform

    def __getitem__(self, index):
        src_path, pose, name = self.data[index]

        src_img = self.img_loader(os.path.join(self.imgs_path, src_path))
        pose    = self.pose_loader(os.path.join(self.pose_path, pose))
        
        pose_transform = transforms.Compose([
#                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                ])

        src_img = self.transform(src_img)
        pose = pose_transform(pose)

        return src_img, pose, name

    def __len__(self):
        return len(self.data)
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 22:25:47 2019

@author: VR LAB PC3
"""

import numpy as np
from numpy import pi, r_
import matplotlib.pyplot as plt
from scipy import optimize
import os
from sklearn.mixture import GaussianMixture

def plot_pose(pose_vector):    
#    save_dir = './pose_img'
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
    
    fig = plt.figure(frameon=False)
    w = 128
    h = 256
    DPI = fig.get_dpi()
    fig.set_size_inches(w/float(DPI),h/float(DPI))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    for i, point in enumerate(connections):
    #    print(i,point)
        if np.sum(pose_coord[point[0]])!=0 and np.sum(pose_coord[point[1]])!=0:
            x = [ pose_coord[point[0]][0], pose_coord[point[1]][0] ]
            y = [ 128 - pose_coord[point[0]][1], 128 - pose_coord[point[1]][1] ]
            ax.plot(x,y, 'ko-')    
    ax.set_xlim((0,64))
    ax.set_ylim((0,128))
#    fig.savefig(os.path.join(save_dir, name[:-3]+'png'), cmap='gray')
    plt.show()

gmm = GaussianMixture(n_components=25, covariance_type='full')


#%%
dataset_folder = 'C:\\Users\\VR LAB PC3\\Desktop\\Y\\my_model\\pose_train\\all'
pose_list = []

for pose in os.listdir(dataset_folder):
    pose_vec = np.load(os.path.join(dataset_folder, pose))
    pose_list.append(pose_vec)
    
gmm.fit(pose_list)
for i in range(25):
    plot_pose(gmm.means_[i])

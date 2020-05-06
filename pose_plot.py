# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 02:19:53 2019

@author: VR LAB PC3
"""

import numpy as np
import matplotlib.pyplot as plt

pose = np.load('pose_train/all/0002_c1s1_000551_01.npy')
pose_coord = []
plt.figure(figsize=(4,8))
for i in range(int(len(pose)/2)):
    x = pose[i*2]
    y = pose[i*2+1]
    pose_coord.append(np.asarray([x,y]))
#    plt.scatter(x,y)
#plt.show()

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


def plot_pose(pose_vector):   
#    pose_vector = pose_vector*128
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
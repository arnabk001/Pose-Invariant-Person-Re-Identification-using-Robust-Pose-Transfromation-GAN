# -*- coding: utf-8 -*-
"""
Created on Fri May  3 00:56:28 2019

@author: VR LAB PC3
"""

import numpy as np
from numpy import pi, r_
import matplotlib.pyplot as plt
from scipy import optimize
import os
from sklearn.mixture import GaussianMixture

def plot_pose(pose_vector, name):    
    save_dir = './results'
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
    fig.savefig(os.path.join(save_dir, name +'.png'), cmap='gray')
    plt.show()

mix_models = dict((joints, GaussianMixture(n_components=3, covariance_type='full'))
                   for joints in list(map(str, list(range(25)))) )

n_models = len(mix_models)


dataset_folder = 'C:\\Users\\VR LAB PC3\\Desktop\\Y\\my_model\\pose_train\\all'
pose_list = []
joint_dict = {'0':[], '1':[], '2':[], '3':[], '4':[], '5':[], '6':[], '7':[], '8':[], '9':[], '10':[], '11':[], '12':[], '13':[],
              '14':[], '15':[], '16':[], '17':[], '18':[], '19':[], '20':[], '21':[], '22':[], '23':[], '24':[]}
for pose in os.listdir(dataset_folder):
    pose_vec = np.load(os.path.join(dataset_folder, pose))
#    pose_list.append(pose_vec)    
    for i in range(int(len(pose_vec)/2)):
        x = pose_vec[i*2]
        y = pose_vec[i*2+1]
        if x != 0 and y != 0 :
            joint_dict[str(i)].append([x,y])
            
            
#%%
mean_poses = {'0':[], '1':[], '2':[]}    
for index, (name, model) in enumerate(mix_models.items()):
    # Train the other parameters using the EM algorithm.
    model.fit(joint_dict[str(index)])
    print('joint', index)
#    print(model.means_)
    for index, means in enumerate(model.means_):
        mean_poses[str(index)].append(model.means_[index])        
#    print(model.covariances_)
    
# plot the most average poses # defined = 3
for pose in mean_poses:
    plot_pose(np.asarray(mean_poses[pose]).flatten(), name='mean_pose'+pose)
    print('pose: '+pose+'printed')

#%% sampling work
input_pose = '0896_c1s4_049756_02.npy'
pose_vec = np.load(os.path.join(dataset_folder, input_pose))
fig = plt.figure()
plot_pose(pose_vec, 'original_pose_'+input_pose[:-4])

joint_loc = pose_vec.reshape((25,2))


for i in range(10):
    generated_pose = []
    for index, (name, model) in enumerate(mix_models.items()):       
        if np.sum(joint_loc[index]) > 0 and not (index in [17, 18, 20, 21, 23, 24]):
            # predict the cluster
            cluster = model.predict([joint_loc[index]])
#            print(cluster)
            # choose the mean and covariance of that cluster and sample from it
            joint = np.random.multivariate_normal(model.means_[cluster][0], model.covariances_[cluster][0])
            while joint[0]<=2 or joint[1]<=2 or joint[0]>=64 or joint[1]>=128:
                joint = np.random.multivariate_normal(model.means_[cluster][0], model.covariances_[cluster][0])
                
            generated_pose.append(0.6*joint + 0.4*joint_loc[index])            
#        elif np.sum(joint_loc[index]) == 0 :
#            mean_loc = [0.0, 0.0]
#            for pose in mean_poses:
#                mean_loc = mean_loc + mean_poses[pose][index]
#            generated_pose.append(mean_loc/3)
        else:
            generated_pose.append([0.0, 0.0])
    plot_pose(np.asarray(generated_pose).flatten(), name='generated_num_'+str(i))

# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 16:24:57 2019

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

mix_models = dict((joints, GaussianMixture(n_components=1, covariance_type='full'))
                   for joints in list(map(str, list(range(25)))) )

n_models = len(mix_models)


#%%
#sample_1 = np.load('C:\\Users\\VR LAB PC3\\Desktop\\Y\\my_model\\pose_train\\all\\0002_c1s1_000551_01.npy')
#print(sample_1.shape)

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
            
#%% def whitening(X):
X = np.asarray(joint_dict['0'])
plt.scatter(X[:,0],X[:,1])
X_reshape = X.reshape((-1, np.prod(X.shape[1:])))
X_centered = X_reshape - np.mean(X_reshape, axis=0)
Sigma = np.dot(X_centered.T, X_centered) / X_centered.shape[0]

U, Lambda, _ = np.linalg.svd(Sigma)
W = np.dot(np.diag(1.0 / np.sqrt(Lambda)), U.T)

new_X = np.dot(X_centered, W.T)
plt.scatter(new_X[:,0], X[:,1])

W_inv = np.linalg.inv(W)
X_centered_re = np.dot(new_X, W_inv)


#%% giving an input pose and generating multiple poses from it
input_pose = '0002_c1s2_000841_01.npy'
pose_vec = np.load(os.path.join(dataset_folder, input_pose))
fig = plt.figure()
plot_pose(pose_vec)
#plt.title('original input')
for i in range(10):
    generated_pose = []
    for index, (name, model) in enumerate(mix_models.items()):
#        print(index)
        if pose_vec[index*2]+pose_vec[index*2+1] == 0:
            mean = [pose_vec[index*2], pose_vec[index*2+1]]
            generated_pose.append(mean)
#            print('continue')
            continue
        else:
#            mean = model.means_[0]
            mean = model.means_[0] #+ 0.5*np.asarray([pose_vec[index*2], pose_vec[index*2+1]])
        joint = np.random.multivariate_normal(mean, model.covariances_[0])
        while joint[0]<=2 or joint[1]<=2 or joint[0]>=64 or joint[1]>=128:
            joint = np.random.multivariate_normal(mean, model.covariances_[0])
        generated_pose.append((mean+joint)/2)
    fig = plt.figure()
    plot_pose(np.asarray(generated_pose).flatten())


#%%
mean_pose = []    
for index, (name, model) in enumerate(mix_models.items()):
    # Train the other parameters using the EM algorithm.
    model.fit(joint_dict[str(index)])
    print('joint', index)
    print(model.means_)
    mean_pose.append(model.means_[0][0])
    mean_pose.append(model.means_[0][1])
    print(model.covariances_)
    
# plot the most average image
plot_pose(mean_pose)

# generate a sample from all the pose distributions
sample_pose = []
for index, (name, model) in enumerate(mix_models.items()):
    sample_pose.append(np.random.multivariate_normal(model.means_[0], model.covariances_[0]))
    
plot_pose(np.asarray(sample_pose).flatten())

# visualizing the distribution of joints
from scipy.stats import multivariate_normal
x, y = np.mgrid[0:64:.1, 0:128:.1]
pos = np.dstack((x, y))
for index, (name, model) in enumerate(mix_models.items()):
    rv = multivariate_normal(model.means_[0], model.covariances_[0])
    fig2 = plt.figure(frameon=True)
    w = 128
    h = 256
    DPI = fig2.get_dpi()
    fig2.set_size_inches(w/float(DPI),h/float(DPI))
    ax = plt.Axes(fig2, [0., 0., 1., 1.])
#    ax.set_axis_off()
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    fig2.add_axes(ax)
    ax.set_xlim((0,64))
    ax.set_ylim((0,128))
    ax.contour(x, y, rv.pdf(pos), 3)
    ax.scatter(np.asarray(joint_dict[str(index)])[:,0], np.asarray(joint_dict[str(index)])[:,1], marker='.', s=4)
    plt.title('joint %d' %index)
    plt.gca().invert_yaxis()
    plt.show()
    
    
    
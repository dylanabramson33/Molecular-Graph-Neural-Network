#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pytraj as pt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class TrajectoryBuilder():
    '''
    A wrapper class for calling common trajectory functions
    '''
    def __init__(self,trajectory_file,topology,mask = '@*',skip=1):
        self.computed_dic = {'rmsd' : False, 'k_means' : False}
        self.mask = mask
        self.trajectory_file = trajectory_file
        self.topology = topology
        self.trajectory = pt.Trajectory(trajectory_file, topology)[mask][0::skip]

    def compute_rmsd(self):
        self.rmsd = pt.rmsd(traj=self.trajectory, mask=self.mask)
        self.computed_dic['rmsd'] = True
        return self.rmsd
    
    def plot_rmsd(self):
        if not self.computed_dic['rmsd']:
            self.compute_rmsd()
        df = pd.DataFrame(self.rmsd)
        df.plot()
    
    def compute_kmeans(self,num_clusters):
        self.num_clusters = num_clusters
        self.k_means = pt.cluster.kmeans(self.trajectory,n_clusters=num_clusters)
        self.computed_dic['k_means'] = True
        return self.k_means
    
    def build_MSM(self,num_clusters = None):
        if not self.computed_dic['k_means']:
            self.compute_kmeans(3)
        transition_matrix = [[0] * self.num_clusters for num in range(self.num_clusters)]
        cluster_indices = self.k_means.cluster_index
        for index in range(len(cluster_indices) - 1):
            transition_matrix[cluster_indices[index]][cluster_indices[index+1]] += 1 
        return transition_matrix
    


# In[31]:


def regex_lite(s):
    firstMatch = s.index('[')
    secondMatch = s.index(']')
    f = s[firstMatch+1:secondMatch].split('-')
    arr = []
    bound1 = int(f[0])
    bound2 = int(f[1])
    for i in range(bound1,bound2+1):
        arr.append(f'H{i}')
    return ",".join(arr)


# In[32]:


from scipy.spatial.distance import cdist
hydrogens = regex_lite('H[1-10]')
mask = '@C*,H,' + hydrogens
traj = TrajectoryBuilder('../Hexane/Hexane_wat_strip.trj','../Hexane/Hexane_nowat.prmtop',mask)
hexane_trajectory = traj.trajectory
hexane_trajectory.top.set_nobox()
hexane_trajectory.superpose(ref=0)
hexane_trajectory = hexane_trajectory[::5]
hexane_data = np.empty((0,17,3))
for i in range(len(hexane_trajectory) - 1): 
    newarr = (hexane_trajectory[i].xyz - hexane_trajectory[i+1].xyz).reshape(1,17,3)
    hexane_data = np.append(hexane_data,newarr,axis=0)


# In[33]:


training_data = [(data,0) for data in hexane_data]


# In[34]:


hydrogens = regex_lite('H[1-10]')
mask = '@C*,H,' + hydrogens
traj2 = TrajectoryBuilder('../Hexanol/Hexanol_wat_strip.trj','../Hexanol/Hexanol_nowat.prmtop',mask)
hexanol_trajectory = traj.trajectory
hexanol_trajectory.top.set_nobox()
hexanol_trajectory.superpose(ref=0)
hexanol_trajectory = hexanol_trajectory[::5]
hexanol_data = np.empty((0,17,3))
for i in range(len(hexane_trajectory) - 1): 
    newarr = (hexanol_trajectory[i].xyz - hexanol_trajectory[i+1].xyz).reshape(1,17,3)
    hexanol_data = np.append(hexanol_data,newarr,axis=0)


# In[35]:


for data in hexanol_data:
    training_data.append((data,1))


# In[36]:


hydrogens = regex_lite('H[1-10]')
mask = '@C*,H,' + hydrogens
traj3 = TrajectoryBuilder('../Hexanoic/Hexanoic_wat_strip.trj','../Hexanoic/Hexanoic_nowat.prmtop',mask)
hexanoic_trajectory = traj.trajectory
hexanoic_trajectory.top.set_nobox()
hexanoic_trajectory.superpose(ref=0)
hexanoic_trajectory = hexanoic_trajectory[::5]
hexanoic_data = np.empty((0,17,3))
for i in range(len(hexane_trajectory) - 1): 
    newarr = (hexanoic_trajectory[i].xyz - hexanoic_trajectory[i+1].xyz).reshape(1,17,3)
    hexanoic_data = np.append(hexanoic_data,newarr,axis=0)


# In[37]:


for data in hexanoic_data:
    training_data.append((data,2))


# In[38]:


import random
random.shuffle(training_data)


# In[39]:


validation = training_data[:3005]


# In[40]:


test = training_data[3005:7010]


# In[41]:


training = training_data[7010:]


# In[42]:


training_data = [x[0] for x in training]


# In[43]:


validation_data = [x[0] for x in validation]


# In[44]:


X_train = np.empty((0,17,3))
for data in training_data:
    X_train = np.append(X_train,data.reshape(1,17,3),axis=0)


# In[76]:


Y_train = [x[1] for x in training]
Y_train = np.array(Y_train)


# In[74]:


X_val = np.empty((0,17,3)) 
for data in validation_data:
    X_val = np.append(X_val,data.reshape(1,17,3),axis=0)


# In[77]:


Y_val = [x[1] for x in validation]
Y_val = np.array(Y_val)


# In[48]:


import sys
import math
import networkx as nx

np.set_printoptions(threshold=sys.maxsize)

def build_adjacency_from_topology(topology):
    '''
    builds adjacency matrix with self loops
    num_atoms (int) : number of atoms in the topology
    bond_connections (array) : connections
    '''
    num_atoms = topology.n_atoms
    bond_connections = topology.bond_indices
    adjacency_matrix = np.zeros((num_atoms,num_atoms))
    for row in bond_connections:
        adjacency_matrix[row[0]][row[0]] = 1
        adjacency_matrix[row[0]][row[1]] = 1
        adjacency_matrix[row[1]][row[0]] = 1
        adjacency_matrix[row[1]][row[1]] = 1
    return adjacency_matrix


def compute_distance_tensor(trajectory):
    cutoff_vec = np.vectorize(lambda x: 0 if x > 2.5 else 1)
    matrix = pt.analysis.matrix.dist(trajectory)
    return np.vectorize(cutoff_vec)(matrix)
    
    

def compute_degree_matrix(adjacency_matrix):
    shape = adjacency_matrix.shape[0]
    degree_matrix = np.zeros((shape,shape))
    for i,row in enumerate(adjacency_matrix):
        incident_edges = np.sum(row)
        degree_matrix[i][i] = math.sqrt(1/incident_edges)
    
    return degree_matrix

A_hat = build_adjacency_from_topology(hexane_trajectory.top)
D = compute_degree_matrix(A_hat)

G = nx.Graph(A_hat)
pos = nx.kamada_kawai_layout(G)
nx.draw(G,pos,with_labels = True)


# In[49]:


import tensorflow as tf


# In[78]:


class GraphConvolution(tf.keras.layers.Layer):
    def __init__(self,units,diagonal,adjacency):
        super(GraphConvolution, self).__init__()
        self.units = units
        self.diagonal = diagonal
        self.adjacency = adjacency

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        
    def call(self, inputs):
        hw = tf.matmul(inputs,self.w)
        dh = tf.matmul(self.diagonal,hw)
        ad = tf.matmul(self.adjacency,dh)
        return tf.matmul(self.diagonal,ad) 


# In[79]:


from spektral.layers import GCNConv, GlobalSumPool
from tensorflow.keras.layers import Dense
class GCNN(tf.keras.Model):
    def __init__(self,diagonal,adjacency):
        super(GCNN, self).__init__()
        self.graphconvolve1 = GraphConvolution(17,diagonal,adjacency)
        self.graphconvolve2 = GraphConvolution(10,diagonal,adjacency)
        self.graphconvolve3 = GraphConvolution(5,diagonal,adjacency)
        self.pool = GlobalSumPool()
        
        
        
    def call(self, inputs):
        first_layer = self.graphconvolve1(inputs)
        a1 = tf.keras.layers.Activation('relu')(first_layer)
        second_layer = self.graphconvolve2(a1)
        a2 = tf.keras.layers.Activation('relu')(second_layer)
        third_layer = self.graphconvolve3(a2)
        fourth_layer = self.pool(third_layer)        
        return Dense(3,'softmax')(fourth_layer)
    

model = GCNN(D.astype('float32'),A_hat.astype('float32'))




        


# In[80]:


Y_train = tf.keras.utils.to_categorical(Y_train)
Y_val = tf.keras.utils.to_categorical(Y_val)

model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])

history = model.fit(
    X_train,
    Y_train,
    batch_size=64,
    epochs=2,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(X_val, Y_val),
)


# In[19]:


from sklearn.manifold import TSNE
get_ipython().run_line_magic('matplotlib', 'notebook')
F = np.identity(20)
output = model(F.astype('float32'))
output = output.numpy()


# In[34]:


hexane_differences = np.load('hexane_distance_traj.npy')


# In[35]:


hexanol_differences = np.load('hexanol_trajectory_difference.npy')


# In[36]:


hexanoic_differences = np.load('hexanoic_trajectory_difference.npy')


# In[38]:


hexane_differences.shape


# In[40]:


hexanol_differences.shape


# In[41]:


hexanoic_differences.shape


# In[67]:


hexane_differences[0,0].shape


# In[69]:


import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# In[72]:


y_train.shape


# In[ ]:





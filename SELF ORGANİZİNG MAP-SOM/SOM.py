# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 14:00:24 2020

@author: D-RI
"""
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd 
# array_t=50 + 3.5* np.random.randn(200,3)
# array_1_t= 1+ 4.5 * np.random.randn(200,3)
# array_2_t=15 + 2.5 * np.random.randn(200,3)

# np.save('array', array_t)
# np.save('array1', array_1_t)
# np.save('array2', array_2_t)

array=abs(np.load('array.npy'))
array_1=abs(np.load('array1.npy'))
array_2=abs(np.load('array2.npy'))

x=array[:,0]
y=array[:,1]
z=array[:,2]
x_1=array_1[:,0]
y_1=array_1[:,1]
z_1=array_1[:,2]
x_2=array_2[:,0]
y_2=array_2[:,1]
z_2=array_2[:,2]


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z)
ax.scatter(x_1,y_1,z_1)
ax.scatter(x_2,y_2,z_2)
plt.show()


arrays=np.concatenate((array[0:180,:].reshape(3,180),array_1[0:180,:].reshape(3,180),array_2[0:180,:].reshape(3,180)),axis=1)
test_array=np.concatenate((array[150:200,:].reshape(3,50),array_1[150:200,:].reshape(3,50),array_2[150:200,:].reshape(3,50)),axis=1)


class Som_kohonen:
    def hjfunction(self,weight,j,iteration,iteration_number):
        d=[]
        for i in range(len(weight)):
            d.append(np.linalg.norm(weight[i,:]-weight[j,:]))
        sigma_0=0.305
        sigma=sigma_0*math.exp(-1*(iteration+1)/iteration_number)
        h_ji=[]
       
        for x in range(len(d)):
            h_ji.append(math.exp(-1*(math.pow(d[x], 2))/2*math.pow(sigma,2)))
      
        return h_ji                 
       
    def c_iteration(self,c,iteration,iteration_number):
        sigma_0=0.305
        sigma=sigma_0*math.exp(-1*(iteration+1)/iteration_number)
        
        c_iter=c*math.exp(-1*(iteration+1)/sigma)
        return c_iter
        
    def neuron_win(self,weight,arrays):
        d=[]
         
        for i in range(len(weight)):
            d.append(np.matmul(weight[i,:],arrays.reshape(3,1)))
        d_np=np.array(d)  
        max_index=np.argmax(d_np)
        # max_index = d.index(max(d))
        return max_index    
    
    
    
    
    def update(self,weight,win,array,c,iteration,iteration_number):
        hj=self.hjfunction(weight,win,iteration,iteration_number)
        for i in range(len(weight)):
            for j in range(3):  
                weight[i,j]=weight[i,j]+self.c_iteration(c,iteration,iteration_number)*hj[i]*(array[j]-weight[i,j])
        
        return weight
        
        
    def start_som(self,weight,arrays,c,iteration):
        
        
        for i in range(iteration):
            for j in range(np.size(arrays,1)):
                win_neuron=self.neuron_win(weight,arrays[:,j])
                #w_list.append(self.update(w_list[a],win_neuron,arrays[:,j],c,i,iteration))
                
                self.update(weight,win_neuron,arrays[:,j],c,i,iteration)
                
        
# a=np.random.rand(6,3)*40
# np.save('W_som', a)

W= np.load('W_som.npy')*4
  
c=0.75
iteration=1000
       
som=Som_kohonen()

som.start_som(W,arrays,c,iteration)      
d=[]
for i in range(150):
    d.append(som.neuron_win(W,test_array[:,i]))
    
    
def scatter3d(x,y,z,f):
   
    
    
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(150):
        
        ax.scatter(x[i], y[i], z[i],marker=f[i])
    
    
    plt.show()    


x=test_array[0,:]
y=test_array[1,:]
z=test_array[2,:]

  
f=[]

for i in range(150):
    if d[i]==0:
        f.append("o")
    elif d[i]==1 or d[i]==5:
        f.append("x")
    elif d[i]==2:
        f.append("v")
scatter3d(x,y,z,f)    
        


# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 18:15:05 2021

@author: D-RI
"""
from mpl_toolkits.mplot3d import Axes3D
import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd 
import matplotlib.cm as cmx
data = pd.read_csv("iris.csv ") 

data.head() 
specific_data=data[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
setosa=np.transpose(specific_data[0:50].to_numpy())
versicolor=np.transpose(specific_data[50:100].to_numpy())
virginica=np.transpose(specific_data[100:150].to_numpy())
input_array=np.concatenate((versicolor,setosa,virginica),axis=1)




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
            d.append(np.matmul(weight[i,:],arrays.reshape(4,1)))
        d_np=np.array(d)  
        max_index=np.argmax(d_np)
        # max_index = d.index(max(d))
        return max_index    
    
    
    
    
    def update(self,weight,win,array,c,iteration,iteration_number):
        hj=self.hjfunction(weight,win,iteration,iteration_number)
        for i in range(len(weight)):
            for j in range(4):  
                weight[i,j]=weight[i,j]+self.c_iteration(c,iteration,iteration_number)*hj[i]*(array[j]-weight[i,j])
        
        return weight
        
        
    def start_som(self,weight,arrays,c,iteration):
        
        
        for i in range(iteration):
            for j in range(np.size(arrays,1)):
                win_neuron=self.neuron_win(weight,arrays[:,j])
                #w_list.append(self.update(w_list[a],win_neuron,arrays[:,j],c,i,iteration))
                
                self.update(weight,win_neuron,arrays[:,j],c,i,iteration)
                
        
# a=np.random.rand(6,4)*40

# np.save('W_6noron', a)
W= np.load('W_6noron.npy')

def scatter3d(x,y,z, cs,f, colorsMap='jet'):
    cm = plt.get_cmap(colorsMap)
    cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(150):
        
        ax.scatter(x[i], y[i], z[i], c=scalarMap.to_rgba(cs[i]),marker=f[i])
    scalarMap.set_array(cs)
    fig.colorbar(scalarMap,label='4.boyut')
    plt.show()


c=0.5         #0.60
iteration=1000
som=Som_kohonen()
d=[]
som.start_som(W,input_array,c,iteration)

for i in range(150):
    d.append(som.neuron_win(W,input_array[:,i]))
    

x=input_array[0,:]
y=input_array[1,:]
z=input_array[2,:]
t=input_array[3,:]
  
f=[]
# for i in range(150):
#     if d[i]==3 or d[i]==0:
#         f.append("o")
#     elif d[i]==4:
#         f.append("x")
#     elif d[i]==1 or d[i]==2:
#         f.append("v")
for i in range(150):
    if d[i]==5:
        f.append("o")
    elif d[i]==2:
        f.append("x")
    elif d[i]==0 or d[i]==4:
        f.append("v")
scatter3d(x,y,z,t,f)           
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 15:22:46 2020

@author: D-RI
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import math

column=np.array([[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1],[-1,-1],[-1,0]
                 ,[-1,1],[-3,3],[-3,1],[-3,0],[-3,-1],[-3,-3],[-1,3],[-1,-3]
                 ,[0,3],[0,-3],[1,3],[1,-3],[3,3],[3,1],[3,0],[3,-1],[3,-3]
                 ,[-2,3],[-3,2],[-3,-2],[-2,-3],[2,3],[3,2],[3,-2],[2,-3]])
array=column.transpose()
one_arr1 = np.ones(33).reshape(1,33)
one_arr = np.ones(9)
one_arr_mines = (-1)*np.ones(24)
y=np.concatenate((one_arr,one_arr_mines)).reshape(1,33)
array_bias=np.concatenate((array,one_arr1))
W=[[1,1,1]]
b=0
c=1

for i in range(500):
    truth = 0
    for j in range(33):
        a = round(sum(W[b]*array_bias[:,j]),2)
        #print(a)
       
        if a>0:
            
            a=1
        else:
            a=-1
        if (y[0,j]-a)==0:
            truth=truth+1
        W.append(W[b]+(0.5*c*(y[0,j]-a)*array_bias[:,j]))
        b=b+1    


fig = plt.figure()


plt.scatter(column[0:9,0],column[0:9,1], marker='+')
plt.scatter(column[9:33,0],column[9:33,1], c= 'green', marker='o')
fi=[]
for i in range(33):
    fi.append([(column[i,0]*column[i,0]),math.sqrt(2)*column[i,0]*column[i,1],column[i,1]*column[i,1]])    
fig2 = plt.figure()
ax = fig2.add_subplot(111, projection='3d') 
for i in range(0,9):
     ax.scatter(fi[i][0],fi[i][1],fi[i][2],c='red', marker='+')

    #plt.scatter(fi[i][0],fi[i][1],fi[i][2], marker='+')
for j in range(9,33):
     ax.scatter(fi[j][0],fi[j][1],fi[j][2], c= 'green', marker='o')
   # plt.scatter(fi[i][0],fi[i][1],fi[i][2], c= 'green', marker='o')    
b=0   
W1=[[1,1,1,1]]
three_d= np.asarray(fi)
three_d_ones=np.concatenate((three_d, one_arr1.transpose()),axis=1)
array_bias1=three_d_ones.transpose()
for i in range(500):
    truth1 = 0
    for j in range(33):
        a = round(sum(W1[b]*array_bias1[:,j]),2)
        #print(a)
       
        if a>0:
            
            a=1
        else:
            a=-1
        if (y[0,j]-a)==0:
            truth1=truth1+1
        W1.append(W1[b]+(0.5*c*(y[0,j]-a)*array_bias1[:,j]))
        b=b+1       

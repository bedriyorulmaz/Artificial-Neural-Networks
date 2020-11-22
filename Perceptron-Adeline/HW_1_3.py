# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 23:36:z6 2020

@author: D-RI
"""
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


z=30
x_1=np.random.rand(1,z)
x_2=np.random.randint(0,90,z).reshape(1,30)
x=np.concatenate((x_1,x_2),axis=0)
one_arr = np.ones(z).reshape(1,z)

yd=[]
for i in range(z):
    yd.append(3*x[0,i]+2*math.cos(x[1,i]))

max_yd=max(yd)
min_yd=min(yd)
x_bias=np.concatenate((x,one_arr))
yd=(yd-min_yd)/(max_yd-min_yd)
y=np.zeros([1,30])
a=1
b=0
W=[[1,1,1]]
for j in range(150):
    truth=0
    for i in range(z):
        v=round(sum(W[b]*x_bias[:,i]),2)
        y[0,i]=1/(1+math.exp(-a*v))
        f=(math.exp(-a*v)*a)/pow((1+math.exp(-a*v)),2)

    
        if (round(yd[i],1)-round(y[0,i],1))==0:
            truth=truth+1
        W.append(W[b]+((yd[i]-y[0,i])*f*np.transpose(x_bias[:,i])))
        b=b+1


x_3=np.random.rand(1,z)
x_4=np.random.rand(1,z)
x_test=np.concatenate((x_3,x_4),axis=0)
one_arr = np.ones(z).reshape(1,z) 
yd_test=[]
for i in range(z):
    yd_test.append(3*x_test[0,i]+2*math.cos(x_test[1,i]))   
max_yd_test=max(yd_test)
min_yd_test=min(yd_test)
x_bias_test=np.concatenate((x_test,one_arr))
yd_test=(yd_test-min_yd_test)/(max_yd_test-min_yd_test)
for i in range(z):
    if yd_test[i] <= 0.5:
        yd_test[i]=0
    else:
         yd_test[i]=1
    


truth_test=0    
for k in range(30):
        a = round(sum(W[b]*x_bias_test[:,k]),2)
            #print(a)
           
        if a>0.5:
                
            a=1
        else:
            a=0
        if (yd_test[k]-a)==0:
            truth_test=truth_test+1
            
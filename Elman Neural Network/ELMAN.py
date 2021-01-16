# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 14:25:42 2021

@author: D-RI
"""
import openpyxl
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

def tanh_deriv(x):
    return 1.0 - np.tanh(x)**2

W_u_List=[]
W_x_List=[]
W_y_List=[]
y_train=[]
e_it=[]
x=[]
def elman(iteration,c,data_dimension,hidden_layer,y_layer,u,y,data_count):
    
    W_1X=np.ones((hidden_layer,data_dimension))
    W_2X=np.ones((hidden_layer,hidden_layer))
    W_3X=np.ones((y_layer,hidden_layer))
    
    
    
    W_u_List.append(W_1X)
    W_x_List.append(W_2X)
    W_y_List.append(W_3X)
    
    q=1
    a=np.ones((hidden_layer,1))
    
    x.append(a)
    for i in range(iteration):
    
        
        y_k=[]
        for j in range(data_count):
            
            v_k=np.matmul(W_x_List[q-1],x[q-1])+W_u_List[q-1]*u[j]
            x.append(np.tanh(v_k))
            y_k.append(np.matmul(W_y_List[q-1],x[q]))
            y_train.append(y_k[j])
            e=y[j+2]-y_k[j]
            e=np.array(e).reshape(1,1)
            e_it.append(e)
            W_y_List.append(W_y_List[q-1]+c*np.matmul(e,np.transpose(x[q-1])))
            W_x_List.append(W_x_List[q-1]+c*np.matmul(np.multiply(np.matmul(np.transpose(W_y_List[q-1]),e),tanh_deriv(v_k)),np.transpose(x[q-1])))
            W_u_List.append(W_u_List[q-1]+c*np.matmul(np.multiply(np.matmul(np.transpose(W_y_List[q-1]),e),tanh_deriv(v_k)),np.array(u[j]).reshape(1,1)))
            q=q+1
            
k=2
y=[5,5] 
u=[]            
for i in range(100)  :  
             
    u.append(k)         
    y.append( ((0.8-0.5*math.exp(-1*math.pow(y[k-1],2)))*y[k-1])-((0.3 + 0.9*math.exp(-1*math.pow(y[k-1],2)))*y[k-2]) +( 0.1*math.sin( math.pi *y[k-1]))+ 0.0002)    
    k=k+1
    
    
    
hidden_layer=6
elman(200,0.2,1,hidden_layer,1,u,y,100)    
y_train_test=[]
d=2
a=np.ones((hidden_layer,1))
q=1 
x_test=[]   
x_test.append(a)
y_k=[]
for i in range(100):
     v_k=np.matmul(W_x_List[19900+i],x_test[q-1])+W_u_List[19900+i]*d
     x_test.append(np.tanh(v_k))
     y_k.append(np.matmul(W_y_List[19900+i],x_test[q]))
     y_train_test.append(y_k[i][0,0])
     d=d+1
     q=q+1


fig = plt.figure()
plt.plot(u, y[2:102]) 
plt.title("original billings out")
plt.xlabel('u')  
plt.ylabel('y')
plt.show()  


fig2 = plt.figure()
plt.plot(u,y_train_test) 
plt.title("elman network billings out") 
plt.xlabel('u')  
plt.ylabel('y_train')
plt.show()

fig3 = plt.figure()
plt.xlim([-2 ,2])
plt.plot(y[1:101], y[2:102]) 
plt.xlabel('y[n-1]')  
plt.ylabel('y[n]')
plt.title("original billings out")
plt.show()  

fig4 = plt.figure()
plt.xlim([-2 ,2])
plt.title("elman network billings out") 
plt.xlabel('y_train[n-1]')  
plt.ylabel('y_train[n]')
plt.plot(y_train_test[0:99],y_train_test[1:100])  
plt.show()
  
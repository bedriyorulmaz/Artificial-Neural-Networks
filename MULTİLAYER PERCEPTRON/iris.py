# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 20:42:23 2020

@author: D-RI
"""
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
data = pd.read_csv("iris.csv ") 

data.head() 


specific_data=data[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]

 
setosa=np.transpose(specific_data[0:50].to_numpy())
versicolor=np.transpose(specific_data[50:100].to_numpy())
virginica=np.transpose(specific_data[100:150].to_numpy())
input_array=np.concatenate((setosa,versicolor,virginica),axis=1)
bias=np.ones((1,150))
input_array=np.concatenate((input_array,bias),axis=0)

y_setosa=np.array([0,0,1])
y_setosa=np.transpose(np.tile(y_setosa, (50, 1)))
y_versicolor=np.array([0,1,0])
y_versicolor=np.transpose(np.tile(y_versicolor, (50, 1)))

y_virginica=np.array([1,0,0])
y_virginica=np.transpose(np.tile(y_virginica, (50, 1)))
y_d=np.concatenate((y_setosa,y_versicolor,y_virginica),axis=1)


def aktivasyonfunc(v,a,c):
    z=[]
    for i in range(c):
        z.append(1/(1+math.exp(-a*v[i])))
    return z
def aktivasyonfunc_turev(v,a,c):
    z=[]
    for i in range(c):
         z.append((math.exp(-v[i]))/pow((1+math.exp(-a*v[i])),2))
        #z.append(1/(1+math.exp(-a*v[i]))*(1-1/(1+math.exp(-a*v[i]))))

    return z
def subtract(v,f,c):
    z=[]
    for i in range(c):
        z.append(v[i]-f[i])
    return z


first_layer=10
second_layer=5

c=0.7 #öğrenme hızı

W_1X=np.ones((first_layer,5))
W_2X=np.ones((second_layer,first_layer+1))
W_3X=np.ones((3,second_layer+1))

W_1_List=[]
W_2_List=[]
W_3_List=[]

W_1_List.append(W_1X)
W_2_List.append(W_2X)
W_3_List.append(W_3X)
y=[]

q=0

for i in range(100):
    
    
    for j in range(150):
       
        v_1=np.matmul(np.array(W_1_List[q]),input_array[:,j])
        
        y_1=np.array(aktivasyonfunc(v_1,1,np.size(v_1,0))).reshape(np.size(v_1,0),1)
        y_1=np.concatenate((y_1,[[1]]))
       
     
        v_2=np.matmul(np.array(W_2_List[q]),y_1)
      
        y_2=np.array(aktivasyonfunc(v_2,1,np.size(v_2,0))).reshape(np.size(v_2,0),1)
        y_2=np.concatenate((y_2,[[1]]))
        
        v_3=np.matmul(np.array(W_3_List[q]),y_2)
       
        y_3=np.array(aktivasyonfunc(v_3,1,np.size(v_3,0))).reshape(np.size(v_3,0),1)
        y.append(y_3)
        e=np.array(subtract(np.transpose(y_d[:,j]),y_3,np.size(y_3,0)))
        E=1/2*np.matmul(np.transpose(e),e)
        
        #gradyan
        
        gradyan_3=e*(np.array(aktivasyonfunc_turev(v_3, 1, np.size(v_3,0))).reshape(np.size(v_3,0),1) )
        B=np.array(W_3_List[q])
        B=np.transpose(B[:,:-1])
        gradyan_2=np.matmul(B,gradyan_3)*(np.array(aktivasyonfunc_turev(v_2,1,np.size(v_2,0))).reshape(np.size(v_2,0),1))
        C=np.array(W_2_List[q])
        C=np.transpose(C[:,:-1])
        gradyan_1=np.matmul(C,gradyan_2)*(np.array(aktivasyonfunc_turev(v_1,1,np.size(v_1,0))).reshape(np.size(v_1,0),1))
        
        x=input_array[:,j].reshape(1,5)
        
        #Ağırlık güncelleme
        
        W_3_List.append(np.array(W_3_List[q])+c*np.matmul(gradyan_3,y_2.reshape(1,second_layer+1)))
        W_2_List.append(np.array(W_2_List[q])+c*np.matmul(gradyan_2,y_1.reshape(1,first_layer+1)))
        W_1_List.append(np.array(W_1_List[q])+c*np.matmul(gradyan_1,x))
        q=q+1
        
              

e_test=[]  
y_test=[]  
for j in range(150):
       
        v_1_test=np.matmul(np.array(W_1_List[q]),input_array[:,j])
        
        y_1_test=np.array(aktivasyonfunc(v_1_test,1,np.size(v_1_test,0))).reshape(np.size(v_1_test,0),1)
        y_1_test=np.concatenate((y_1_test,[[1]]))
       
     
        v_2_test=np.matmul(np.array(W_2_List[q]),y_1_test)
      
        y_2_test=np.array(aktivasyonfunc(v_2_test,1,np.size(v_2_test,0))).reshape(np.size(v_2_test,0),1)
        y_2_test=np.concatenate((y_2_test,[[1]]))
        
        v_3_test=np.matmul(np.array(W_3_List[q]),y_2_test)
       
        y_3_test=np.array(aktivasyonfunc(v_3_test,1,np.size(v_3_test,0))).reshape(np.size(v_3_test,0),1)
        y_test.append(np.around(y_3_test,1))
        e_test.append(np.array(subtract(np.transpose(y_d[:,j]),np.around(y_3_test,1),np.size(y_3_test,0))) )
    
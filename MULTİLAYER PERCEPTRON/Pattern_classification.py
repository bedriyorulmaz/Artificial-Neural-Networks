# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 20:21:58 2020

@author: D-RI
"""
import math
import matplotlib.pyplot as plt
import numpy as np
sign_equal= np.ones((5, 10))
sign_plus= np.ones((5, 10))
sign_minus= np.ones((5, 10))
sign_times= np.ones((5, 10))
fig = plt.figure()

sign_equal[1,:]=0
sign_equal[3,:]=0
equal=fig.add_subplot(2,2,1)                    #0001
plt.imshow(sign_equal, cmap='gray')
plt.show()

sign_plus[2,:]=0
sign_plus[:,4:6]=0
equal=fig.add_subplot(2,2,2)                    #0010
plt.imshow(sign_plus, cmap='gray')
plt.show()

sign_minus[2,:]=0
equal=fig.add_subplot(2,2,3)                    #0100
plt.imshow(sign_minus, cmap='gray')
plt.show()

for i in range(3):
    sign_times[i,i+2]=0
    sign_times[i,7-i]=0
    sign_times[4-i,7-i]=0                       #1000
    sign_times[4-i,i+2]=0
    equal=fig.add_subplot(2,2,4)
plt.imshow(sign_times, cmap='gray')
plt.show()

fig2 = plt.figure()
#noisy

def noisy(image):
    noisy = image + 0.2 * np.random.rand(5, 10)
    noisy = noisy/noisy.max() 
    return noisy
  
noisy_sign_equal=noisy(sign_equal)  
noisy_sign_plus=noisy(sign_plus)  
noisy_sign_minus=noisy(sign_minus)
noisy_sign_times=noisy(sign_times)
noisy=fig2.add_subplot(2,2,1)  
plt.imshow(noisy_sign_equal, cmap='gray')
plt.show()
noisy=fig2.add_subplot(2,2,2)
plt.imshow(noisy_sign_plus, cmap='gray')
plt.show()
noisy=fig2.add_subplot(2,2,3)
plt.imshow(noisy_sign_minus, cmap='gray')
plt.show()
noisy=fig2.add_subplot(2,2,4)
plt.imshow(noisy_sign_times, cmap='gray')
plt.show()

fig3= plt.figure()
#görüntü bozmaa
def equal(image):
    x= np.ones((5, 10))
    for i in range(5):
        for j in range(10):
            x[i,j]=image[i,j]
    return x 
    
    
    
broke_sign_equal= np.ones((5, 10))
broke_sign_plus= np.ones((5, 10))
broke_sign_minus= np.ones((5, 10))
broke_sign_times= np.ones((5, 10))

broke_sign_equal=equal(sign_equal)
broke_sign_equal[0,0:1]=0
broke_sign_equal[4,4:7]=0
broke=fig3.add_subplot(2,2,1)
plt.imshow(broke_sign_equal, cmap='gray')
plt.show()

broke_sign_plus=equal(sign_plus) 
broke_sign_plus[1,2:3]=0
broke_sign_plus[4,7:9]=0
broke_sign_plus[0,9]=0
broke=fig3.add_subplot(2,2,2)
plt.imshow(broke_sign_plus, cmap='gray')
plt.show()

broke_sign_minus=equal(sign_minus)
broke_sign_minus[3:4,2]=0
broke_sign_minus[0:2,8]=0
broke_sign_minus[4,9]=0
broke=fig3.add_subplot(2,2,3)
plt.imshow(broke_sign_minus, cmap='gray')
plt.show()


broke_sign_times=equal(sign_times) 
broke_sign_times[:,0]=0
broke=fig3.add_subplot(2,2,4)
plt.imshow(broke_sign_times, cmap='gray')
plt.show()

def change_0_1(image):
    a=[]
    for i in range(5):
        for j in range(10):
            if image[i,j]==1:
                image[i,j]=0.9
            if image[i,j]==0:
                image[i,j]=0.1
            a.append(image[i,j])
    return a            

input_array_equal= np.ones((50,3))
input_array_equal[:,0]=change_0_1(sign_equal)
input_array_equal[:,1]=change_0_1(noisy_sign_equal)
input_array_equal[:,2]=change_0_1(broke_sign_equal)

input_array_plus= np.ones((50,3))
input_array_plus[:,0]=change_0_1(sign_plus)
input_array_plus[:,1]=change_0_1(noisy_sign_plus)
input_array_plus[:,2]=change_0_1(broke_sign_plus)


input_array_minus= np.ones((50,3))
input_array_minus[:,0]=change_0_1(sign_minus)
input_array_minus[:,1]=change_0_1(noisy_sign_minus)
input_array_minus[:,2]=change_0_1(broke_sign_minus)

input_array_times= np.ones((50,3))
input_array_times[:,0]=change_0_1(sign_times)
input_array_times[:,1]=change_0_1(noisy_sign_times)
input_array_times[:,2]=change_0_1(broke_sign_times)

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


input_array=np.concatenate((input_array_equal,input_array_plus,input_array_minus,input_array_times),axis=1)
bias=np.ones((1,12))
input_array=np.concatenate((input_array,bias))

y_d=[[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]]
y_d=np.array(y_d)
y_d=np.transpose(y_d)



second_layer=8
first_layer=19


W_1X=np.ones((first_layer,51))
W_2X=np.ones((second_layer,first_layer+1))
W_3X=np.ones((4,second_layer+1))

W_1_List=[]
W_2_List=[]
W_3_List=[]

W_1_List.append(W_1X)
W_2_List.append(W_2X)
W_3_List.append(W_3X)
y=[]

q=0
for i in range(80):
    for j in range(12):
      
        
      
        
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
        gradyan_2=(np.matmul(B,gradyan_3))*(np.array(aktivasyonfunc_turev(v_2,1,np.size(v_2,0))).reshape(np.size(v_2,0),1))
        C=np.array(W_2_List[q])
        C=np.transpose(C[:,:-1])
        gradyan_1=(np.matmul(C,gradyan_2))*(np.array(aktivasyonfunc_turev(v_1,1,np.size(v_1,0))).reshape(np.size(v_1,0),1))
        
        x=input_array[:,j].reshape(1,51)
        
        #Ağırlık güncelleme
        W_3_List.append(np.array(W_3_List[q])+1*np.matmul(gradyan_3,y_2.reshape(1,second_layer+1)))
        W_2_List.append(np.array(W_2_List[q])+1*np.matmul(gradyan_2,y_1.reshape(1,first_layer+1)))
        W_1_List.append(np.array(W_1_List[q])+1*np.matmul(gradyan_1,x))
        q=q+1
        
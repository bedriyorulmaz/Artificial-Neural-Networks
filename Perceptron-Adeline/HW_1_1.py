# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 16:49:02 2020

@author: D-RI   
"""
import statistics
from functools import reduce
import numpy as np
#a=np.random.randint(1,10,size=(4,40))
#b=np.random.randint(1,10,size=(19,6))
#np.save('arr', a)

#random array oluşturduk ve yorum satırı haline getirdik.

random_matrix= np.load('arr.npy')

        

one_arr= np.ones(20).reshape(1,20)
minusone_arr= one_arr* (-1)


      
küme1=random_matrix[:,0:20] 
küme2=random_matrix[:,20:40]

küme1=np.concatenate((küme1,minusone_arr))   
küme2=np.concatenate((küme2,one_arr))
küme1=np.concatenate((küme1,one_arr))
küme2=np.concatenate((küme2,one_arr)) 

egitim_mix=[]

for i in range(12):
    egitim_mix.append(küme1[:,i].tolist())
    egitim_mix.append(küme2[:,i].tolist())
egitim_mix.append(küme2[:,13])

egitim_mix_array=np.array(egitim_mix)
egitim_mix_array=np.transpose(egitim_mix_array)


egitim=np.concatenate((küme1[:,0:12],küme2[:,0:13]),axis=1)
test=np.concatenate((küme1[:,12:20],küme2[:,13:20]),axis=1)




W_list=[[1,1,1,1,1,1],[1.5,1.5,1.5,1.5,1.5,1.5],[2,2,2,2,2,2],[2.5,2.5,2.5,2.5,2.5,2.5],[3,3,3,3,3,3]
       ,[3.5,3.5,3.5,3.5,3.5,3.5],[4,4,4,4,4,4],[4.5,4.5,4.5,4.5,4.5,4.5],[5,5,5,5,5,5]
       ,[5.5,5.5,5.5,5.5,5.5,5.5],[6,6,6,6,6,6],[6.5,6.5,6.5,6.5,6.5,6.5],[7,7,7,7,7,7]
       ,[7.5,7.5,7.5,7.5,7.5,7.5],[8,8,8,8,8,8],[8.5,8.5,8.5,8.5,8.5,8.5],[9,9,9,9,9,9]
       ,[9.5,9.5,9.5,9.5,9.5,9.5],[10,10,10,10,10,10]]
W=[[]]
W_mix=[[]]

iter_value=[]
iter_value_mix=[]
iter_value_test=[]
b_mix=1
b=1
c=1

for w in W_list:
    W.append(w)
    W_mix.append(w)
    for i in range(70):
         truth = 0
         for j in range(25):
            a = round(sum(W[b]*egitim[:,j]),2)
           
           
            if a>0:
                
                a=1
            else:
                a=-1
            if (egitim[4,j]-a)==0:
                truth=truth+1
            W.append(W[b]+(0.5*c*(egitim[4,j]-a)*egitim[:,j]))
            b=b+1    
         if truth == 25:
            iter_value.append(i+1)
            break
        
    for i in range(70):  
         truth_mix = 0
         for j in range(25):
             a_mix = round(sum(W_mix[b_mix]*egitim_mix_array[:,j]),2)
             #print(a)
           
             if a_mix>0:
                
                 a_mix=1
             else:
                 a_mix=-1
             if (egitim_mix_array[4,j]-a_mix)==0:
                 truth_mix=truth_mix+1
             W_mix.append(W_mix[b_mix]+(0.5*c*(egitim_mix_array[4,j]-a_mix)*egitim_mix_array[:,j]))
             b_mix=b_mix+1    
         if truth_mix == 25:
             iter_value_mix.append(i+1)
             break
         
        
         
         
    
    
    
    truth_test=0
    for k in range(15):
        a = round(sum(W[b]*test[:,k]),2)
            #print(a)
           
        if a>0:
                
            a=1
        else:
            a=-1
        if (test[4,k]-a)==0:
            truth_test=truth_test+1
        if k==14:    
            iter_value_test.append(truth_test)   
    

def Average(lst): 
    return reduce(lambda a, b: a + b, lst) / len(lst)
Avarage_iter=Average(iter_value)
standart_devision=statistics.stdev(iter_value)

Avarage_iter_mix=Average(iter_value_mix)
standart_devision_mix=statistics.stdev(iter_value_mix)
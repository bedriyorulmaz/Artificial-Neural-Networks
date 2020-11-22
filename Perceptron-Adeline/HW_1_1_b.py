# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 12:29:18 2020

@author: D-RI
"""

import numpy as np
random_matrix_new=np.random.randint(1,10,size=(5,40))


one_arr = np.ones(40).reshape(1,40)
random_matrix_new=np.concatenate((random_matrix_new,one_arr))


W=[[1,1,1,1,1,1]]

b=0
c=1

for i in range(500):
    truth = 0
    for j in range(40):
        a = round(sum(W[b]*random_matrix_new[:,j]),2)
        #print(a)
       
        if a>0:
            
            a=1
        else:
            a=-1
        if (random_matrix_new[4,j]-a)==0:
            truth=truth+1
        W.append(W[b]+(0.5*c*(random_matrix_new[4,j]-a)*random_matrix_new[:,j]))
        b=b+1    
    if truth is 40:
        break
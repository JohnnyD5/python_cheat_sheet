# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 15:40:46 2019

@author: zding5
"""
import numpy as np

t = np.array([0,1,2,3,4])
x = np.array([-2,-1,0,1,2])
y = np.array([-3,-1.5,0,1.5,3])
v = np.array([10,20,30,40,50])
new_matrix = np.hstack((t.reshape(len(t),1),
                        x.reshape(len(x),1),
                        y.reshape(len(y),1),
                        v.reshape(len(v),1)))
print(new_matrix)
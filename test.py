# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 15:40:46 2019

@author: zding5
"""


import numpy as np
def f(x,y):
    return 10*x+y
a = np.fromfunction(f,(5,4))
print(a[-1])
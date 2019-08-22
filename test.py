# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 15:40:46 2019

@author: zding5
"""


import numpy as np
A = np.arange(2,6)
B = np.arange(1,5)*2
print(A,'\n',B)
print(np.column_stack((A,B)))
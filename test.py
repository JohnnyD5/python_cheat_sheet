# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 15:40:46 2019

@author: zding5
"""


import numpy as np
import pandas as pd
path = "test_import2.csv"
df=pd.read_csv(path, delimiter = ',', header = 0)
print(df)
a = df.iloc[:,:2].values
print(a)
print(type(a))
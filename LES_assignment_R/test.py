# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 11:03:04 2019

@author: zding5
"""

file = open("data/u_v_time_4nodes_re1000.dat","r")
count = 0
for line in file:
    print(line)
    count += 1
    if count == 10:
        data = line
        break


print(data)
for i in data:
    print(i, type(i))

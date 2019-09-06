# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 10:02:50 2019

@author: zding5
"""
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('precision', 6)
pd.set_option('expand_frame_repr', True)

class LES():
    def __init__(self, path = None):
        if len(path) < 1:
            path = 'C:/Users/zding5/OneDrive - Louisiana State University/Cheat_sheet/python_cheat_sheet/LES_assignment_R/data'
        self.path = path + '/u_v_time_4nodes_re1000.dat'
        pdf=pd.read_csv(self.path,skiprows = 9, dtype= 'float64',  names = ["u1", "v1", "u2", "v2",
                       "u3", "v3", "u4", "v4"])
        pdf.to_hdf(path + '/cloud.h5',key='cloud')
        self.data = pd.read_hdf(path + '/cloud.h5', key='cloud')
        #self.data['v_mag'] = (self.data['vx']**2 + self.data['vy']**2)**0.5

    def plot_angularV_over_t(self, t = None, y = None):
        # Plot the angular velocity of the ellipse over time
        if t is None:
            t = self.data['t']
        if y is None:
            y = abs(self.data['wz'])
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(t, y)
        ax.set_xlabel('t')
        ax.set_ylabel(r'$\hat{\omega}$')
        #removing top and right borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # set x limit and y limit
        ax.set_xlim(left = 0, right = 300)
        plt.show()
        return

    def test(self):
        print(self.data)
        return 0

if __name__ == '__main__':
    path = input("Please identify the path of desired folder; the folder should contain cloud.out file:\n")
    ### convert windows path to linux path
    path = list(path)
    for i, c in enumerate(path):
        if c == '\\':
            path[i] = '/'
    path = ''.join(path)
    ###
    case = LES(path)
    #case.plot_angularV_over_t()
    case.test()

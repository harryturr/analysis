#!/usr/bin/env python2

# harrisonn griffin 2019
# @harryturr

import numpy as np
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


file_number = np.array([%s]) % #number of ifle
label_list = np.array([%s]) % #label


filename_prefix = 'prefix'
filename_suffix = 'suffix'


vcolumn = 1
dfcolumn = 2
disscolumn = 3
ampcolumn = 4


# moving average box by convolution
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def minimum(y, pts):
    val, idx = min((val, idx) for (idx, val) in enumerate(smooth(y,pts))) 
    print val, idx	
    return idx


index = 0
for f_number in file_number:
    fnum = str(f_number).zfill(3)
    filename = filename_prefix + fnum + filename_suffix
    print filename

    # extracting data
    data = np.genfromtxt(filename, delimiter=',')
    Vbias = data[:, vcolumn]
    df = data[:, dfcolumn]
    amp = data[:, ampcolumn]
    diss = data[:, disscolumn]

    h_Vbias = Vbias[:len(Vbias)/2]
    h_df = df[:len(Vbias)/2] * 50  # scaling to Hz (50 hz/V)
    h_amp = amp[:len(Vbias)/2]
    h_diss = diss[:len(Vbias)/2]

    # determining where to split the data
    idx = minimum(h_df,50)
    print len(h_df)

    # splitting freq shift
    h_df_l = h_df[:idx]
    h_df_r = h_df[idx:2*idx]
    h_df_r = list(reversed(h_df_r))
    # calculating difference	
    h_df_diff = [h_df_l[i] - h_df_r[i] for i in range(0,len(h_df_l))]

    # splitting dissipation
    h_diss_l = h_diss[:idx]
    h_diss_r = h_diss[idx:2*idx]
    h_diss_r = list(reversed(h_diss_r)) 
    h_diss_diff = [h_diss_l[i] - h_diss_r[i] for i in range(0,len(h_diss_l))]

    # define figure environment
    fig1 = plt.figure(1)
    fig1.set_figheight(6.8)
    fig1.set_figwidth(8.5)


    # plotting dissipation vs freq shift
    ax0=fig1.add_subplot(1,1,1)

    ax0.plot(h_df[:idx], h_diss[:idx],color = 'lime', label = 'left')
    ax0.plot(h_df[idx:], h_diss[idx:], color = 'orange',label = 'right')
    ax0.set_title("")
    ax0.set_xlabel('frequency shift (V)',fontsize=16)
    ax0.set_ylabel('dissipation (V)',fontsize=16)
    ax0.tick_params(direction='in', length=6, width=2)
    ax0.legend(loc='upper right', shadow=True, fontsize='large')
    ax0.set_title('')

    fig2 = plt.figure(2)
    fig2.set_figheight(11)
    fig2.set_figwidth(8.5)

    ax1=fig2.add_subplot(3,1,1)
    ax2=fig2.add_subplot(3,1,2,sharex=ax1)
    ax3=fig2.add_subplot(3,1,3,sharex=ax1)

    # fitting
    # a = np.polyfit(h_Vbias, h_df, 2)
    # b = np.poly1d(a)


    ax1.plot(h_Vbias,h_df, label = label_list[index])
    ax1.plot(h_Vbias,smooth(h_df,100), label = 'smooth')
    ax2.plot(h_Vbias,h_amp,label = label_list[index])
    ax2.plot(h_Vbias,smooth(h_amp,5), label = 'smooth')
    ax3.plot(h_Vbias,h_diss,label = label_list[index])
    ax3.plot(h_Vbias,smooth(h_diss,5),label = 'smooth')
    ax1.legend(loc='upper right', shadow=True, fontsize='large')

    ax1.set_title("")
    ax1.set_xlabel('')
    ax1.set_ylabel('frequency shift (hz)', fontsize = 16)
    ax1.tick_params(direction='in', length=6, width=2)
    ax2.set_title("")
    ax2.set_xlabel('')
    ax2.set_ylabel('amplitude (V)', fontsize = 16)
    ax2.tick_params(direction='in', length=6, width=2)
    ax3.set_title("")
    ax3.set_xlabel('')
    ax3.set_ylabel('dissipation (V)', fontsize = 16)
    ax3.tick_params(direction='in', length=6, width=2)

    fig2.subplots_adjust(hspace=0, right = 0.8)
    fig1.subplots_adjust(hspace=0, right = 0.8)
    ax1.set_title('')
    
    fig3 = plt.figure(3)
    fig3.set_figheight(6.8)
    fig3.set_figwidth(8.5)


    # plotting left and right overlap ~~~ ~~~ ~~~
    ax6=fig3.add_subplot(2,1,1)
    ax7 = fig3.add_subplot(2,1,2,sharex=ax6)

    ax6.plot(h_Vbias,h_df, label = 'full')
    ax6.plot(h_Vbias[:len(h_df_l)], h_df_l,color='lime',label = 'right')
    ax6.plot(h_Vbias[:len(h_df_r)], h_df_r, color='orange', label = 'left')
    ax6.set_ylabel('df (hz)', fontsize=16)
    ax6.tick_params(direction='in', length=6, width=2)
    ax6.legend(loc='upper right', shadow=True, fontsize='large')

    ax60 = ax6.twinx()
    ax60.plot(h_Vbias[:len(h_df_l)], h_df_diff,'r',alpha=0.1)
    ax60.set_ylabel('residuals', color='r')
    ax60.tick_params('y', colors='r', direction='in')
    ax60.set_ylim(-30, 10)
    ax60.set_ylabel('residuals', color='r', fontsize=16)
    ax60.tick_params(direction='in', length=6, width=2)


    ax7.plot(h_Vbias,h_diss, label = 'full')
    ax7.plot(h_Vbias[:len(h_diss_l)], h_diss_l, color = 'lime',label = 'left')
    ax7.plot(h_Vbias[:len(h_diss_r)], h_diss_r, color='orange',label = 'right')
    ax7.set_ylabel('dissipation (V)', fontsize=16)
    ax7.set_xlabel('bias (V))', fontsize=16)
    ax7.tick_params(direction='in', length=6, width=2)
    ax7.legend(loc='upper right', shadow=True, fontsize='large')


    ax70 = ax7.twinx()
    ax70.plot(h_Vbias[:len(h_diss_l)], h_diss_diff, 'r', alpha=0.1)
    ax70.set_ylabel('residuals', color='r', fontsize=16)
    ax70.tick_params(direction='in', length=6, width=2)
    ax70.tick_params('y', colors='r', direction='in')

    index = index +1

    np.savetxt("df_qd.csv", np.column_stack((h_Vbias, h_df)), delimiter=",", fmt='%s')

plt.show()

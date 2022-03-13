import argparse
import pandas as pd
import numpy as np
import matplotlib
import xgboost as xgb
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
from sklearn.model_selection import train_test_split
import pickle
from itertools import product
from keras.utils import np_utils 
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, auc

signal = ['qqH_Rest',
        'QQ2HQQ_GE2J_MJJ_60_120',
        'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25',
        'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25',
        'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25',
        'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_GT25',
        'QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200']

labelNames = [r'qqH rest', 
            r'60<m$_{jj}$<120',
            r'350<m$_{jj}$<700, 0<p$_{T}^{H}$<200, 0<p$_{T}^{H_{jj}}$<25',
            r'350<m$_{jj}$<700, 0<p$_{T}^{H}$<200, p$_{T}^{H_{jj}}$>25',
            r'm$_{jj}$>700, 0<p$_{T}^{H}$<200, 0<p$_{T}^{H_{jj}}$<25',
            r'm$_{jj}$>700, 0<p$_{T}^{H}$<200, p$_{T}^{H_{jj}}$>25',
            r'm$_{jj}$>350, p$_{T}^{H}$>200'
            ]

color  = ['silver','indianred','salmon','lightgreen','seagreen','mediumturquoise','darkslategrey','skyblue','steelblue','lightsteelblue','mediumslateblue']


for i in range(len(signal)):

    fig, ax = plt.subplots()
    #BDT
    name_fpr = 'csv_files/BDT_binary_fpr_' + signal[i]
    name_tpr = 'csv_files/BDT_binary_tpr_' + signal[i]
    fpr_keras = np.loadtxt(name_fpr, delimiter = ',')
    tpr_keras = np.loadtxt(name_tpr, delimiter = ',')
    auc_test = auc(fpr_keras, tpr_keras)
    ax.plot(fpr_keras, tpr_keras, label = 'AUC = {0} BDT'.format(round(auc_test, 3)), color = 'blue')
    #NNs
    name_fpr = 'csv_files/NN_binary_fpr_' + signal[i]
    name_tpr = 'csv_files/NN_binary_tpr_' + signal[i]
    fpr_keras = np.loadtxt(name_fpr, delimiter = ',')
    tpr_keras = np.loadtxt(name_tpr, delimiter = ',')
    auc_test = auc(fpr_keras, tpr_keras)
    ax.plot(fpr_keras, tpr_keras, label = 'AUC = {0} NN'.format(round(auc_test, 3)), color = 'red')
    #Cuts
    name_fpr = 'csv_files/Cuts_binary_fpr_' + signal[i]
    name_tpr = 'csv_files/Cuts_binary_tpr_' + signal[i]
    fpr_keras = np.loadtxt(name_fpr, delimiter = ',')
    tpr_keras = np.loadtxt(name_tpr, delimiter = ',')
    auc_test = auc(fpr_keras, tpr_keras)
    ax.plot(fpr_keras, tpr_keras, label = 'AUC = {0} Cuts'.format(round(auc_test, 3)), color = 'green')

    ax.legend(loc = 'lower right', fontsize = 'small')
    ax.set_xlabel('Background Efficiency', ha='right', x=1, size=9)
    ax.set_ylabel('Signal Efficiency',ha='right', y=1, size=9)
    ax.grid(True, 'major', linestyle='dotted', color='grey', alpha=0.5)
    plt.tight_layout()
    
    name = 'plotting/BDT_qqH_binary_Multi_ROC_curve' + signal[i]
    plt.savefig(name, dpi = 1200)
    print("Plotting ROC Curve")
    plt.close()


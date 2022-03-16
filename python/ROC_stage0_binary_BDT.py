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

signal = ['ggH','qqH','VH','ttH','tH'] 
labelNames = ['ggH','qqH','VH leptonic','ttH','tH']

color = ['#54aaf8', '#f08633', '#1fcf57', '#cf1f9f', 'gold']

BDT_cm = np.loadtxt('csv_files/BDT_stage_0_BDT_binary_cm', delimiter = ',')
NN_cm = np.loadtxt('csv_files/NN_stage_0_BDT_binary_cm', delimiter = ',')

for i in range(len(signal)):

    len_signal = []
    len_signal.append(np.sum(BDT_cm[:,i]))   
    len_signal.append(np.sum(NN_cm[:,i]))      
    len_signal = np.array(len_signal)
    scale = np.max(len_signal)
    scale_array = len_signal/scale

    fig, ax = plt.subplots()
    #BDT
    BDT_name_fpr = 'csv_files/BDT_stage_0_BDT_binary_fpr_' + signal[i]
    BDT_fpr_keras = np.loadtxt(BDT_name_fpr, delimiter = ',')
    BDT_array_min = abs(np.min(BDT_fpr_keras))
    BDT_array_max = abs(np.max(BDT_fpr_keras))
    BDT_array_range = BDT_array_max + BDT_array_min
    BDT_fpr_keras = (BDT_fpr_keras + BDT_array_min)/BDT_array_range

    BDT_name_tpr = 'csv_files/BDT_stage_0_BDT_binary_tpr_' + signal[i]
    BDT_tpr_keras = np.loadtxt(BDT_name_tpr, delimiter = ',')
    BDT_array_min_tpr = abs(np.min(BDT_tpr_keras))
    BDT_array_max_tpr = abs(np.max(BDT_tpr_keras))
    BDT_array_range_tpr = BDT_array_max_tpr + BDT_array_min_tpr
    BDT_tpr_keras = (BDT_tpr_keras + BDT_array_min_tpr)/BDT_array_range_tpr
    BDT_tpr_keras_scale = BDT_tpr_keras*scale_array[0]

    BDT_auc_test = auc(BDT_fpr_keras, BDT_tpr_keras)*scale_array[0]
    ax.plot(BDT_fpr_keras, BDT_tpr_keras_scale, label = 'BDT AUC = {0}'.format(round(BDT_auc_test, 3)), color = 'blue')

    #NNs
    NN_name_fpr = 'csv_files/NN_stage_0_BDT_binary_fpr_' + signal[i]
    NN_fpr_keras = np.loadtxt(NN_name_fpr, delimiter = ',')
    NN_array_min = abs(np.min(NN_fpr_keras))
    NN_array_max = abs(np.max(NN_fpr_keras))
    NN_array_range = NN_array_max + NN_array_min
    NN_fpr_keras = (NN_fpr_keras + NN_array_min)/NN_array_range

    NN_name_tpr = 'csv_files/NN_stage_0_BDT_binary_tpr_' + signal[i]
    NN_tpr_keras = np.loadtxt(NN_name_tpr, delimiter = ',')
    NN_array_min = abs(np.min(NN_tpr_keras))
    NN_array_max = abs(np.max(NN_tpr_keras))
    NN_array_range = NN_array_max + NN_array_min
    NN_tpr_keras = (NN_tpr_keras + NN_array_min)/NN_array_range
    NN_tpr_keras_scale = NN_tpr_keras*scale_array[1]

    NN_auc_test = auc(NN_fpr_keras, NN_tpr_keras)*scale_array[1]
    ax.plot(NN_fpr_keras, NN_tpr_keras_scale, label = 'NN AUC = {0}'.format(round(NN_auc_test, 3)), color = 'red')

    ax.legend(loc = 'lower right', fontsize = 'small')
    ax.set_xlabel('Background Efficiency', ha='right', x=1, size=9)
    ax.set_ylabel('Signal Efficiency',ha='right', y=1, size=9)
    ax.grid(True, 'major', linestyle='dotted', color='grey', alpha=0.5)
    plt.tight_layout()
    
    name = 'plotting/BDT_stage_0_binary_Multi_ROC_curve' + signal[i]
    plt.savefig(name, dpi = 1200)
    print("Plotting ROC Curve")
    plt.close()
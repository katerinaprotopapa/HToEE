import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import pickle
import ROOT as r
r.gROOT.SetBatch(True)
import sys
from os import path, system
from array import array
from root_numpy import tree2array, fill_hist
from math import pi
import h5py
from itertools import product
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential 
from keras.initializers import RandomNormal 
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Nadam, adam, Adam
from keras.regularizers import l2 
from keras.callbacks import EarlyStopping 
from keras.utils import np_utils 
from keras.metrics import categorical_crossentropy, binary_crossentropy

map_def_2 = [
['QQ2HQQ_FWDH',200],
['rest', 201, 202, 203, 205],
['QQ2HQQ_GE2J_MJJ_60_120',204],
['QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200',206],
['QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25',207],
['QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25',208],
['QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25',209],
['QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_GT25',210],
['WH',300,301,302,303,304,305],
['ZH',400,401,402,403,404,405],
]

binNames = ['QQ2HQQ_FWDH','rest','QQ2HQQ_GE2J_MJJ_60_120','QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200','QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25','QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25','QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25','QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_GT25'] 
bins = 50


train_vars = ['diphotonPt', 'diphotonMass', 'diphotonCosPhi', 'diphotonEta','diphotonPhi', 'diphotonSigmaMoM',
     'dijetMass', 'dijetAbsDEta', 'dijetDPhi', 'dijetCentrality',
     'dijetPt','dijetEta','dijetPhi','dijetMinDRJetPho','dijetDiphoAbsDEta',
     'leadPhotonEta', 'leadPhotonIDMVA', 'leadPhotonEn', 'leadPhotonPt', 'leadPhotonPhi', 'leadPhotonPtOvM',
     'leadJetPt', 'leadJetPUJID', 'leadJetBTagScore', 'leadJetMass',
     'leadJetDiphoDEta','leadJetDiphoDPhi','leadJetEn','leadJetEta','leadJetPhi',
     'subleadPhotonEta', 'subleadPhotonIDMVA', 'subleadPhotonPhi',
     'subleadPhotonEn','subleadPhotonPt', 'subleadPhotonPtOvM',
     'subleadJetDiphoDPhi','subleadJetDiphoDEta',
     'subleadJetPt', 'subleadJetPUJID', 'subleadJetBTagScore', 'subleadJetMass',
     'subleadJetEn','subleadJetEta','subleadJetPhi',
     'subsubleadJetEn','subsubleadJetPt','subsubleadJetEta','subsubleadJetPhi', 'subsubleadJetBTagScore', 
     'subsubleadJetMass',
     'metPt','metPhi','metSumET',
     'nSoftJets',
     'leadElectronEn', 'leadElectronMass', 'leadElectronPt', 'leadElectronEta', 'leadElectronPhi', 'leadElectronCharge',
     'leadMuonEn', 'leadMuonMass', 'leadMuonPt', 'leadMuonEta', 'leadMuonPhi', 'leadMuonCharge',
     'subleadElectronEn', 'subleadElectronMass', 'subleadElectronPt', 'subleadElectronEta', 'subleadElectronPhi', 'subleadElectronCharge', 
     'subleadMuonEn', 'subleadMuonMass', 'subleadMuonPt', 'subleadMuonEta', 'subleadMuonPhi', 'subleadMuonCharge'
     ]

train_vars.append('proc')
train_vars.append('weight')
train_vars.append('HTXS_stage_0')
train_vars.append('HTXS_stage1_2_cat_pTjet30GeV')

#Load the dataframe
dataframes = []
#dataframes.append(pd.read_csv('2017/MC/DataFrames/ggH_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/VBF_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/VH_VBF_BDT_df_2017.csv'))
#dataframes.append(pd.read_csv('2017/MC/DataFrames/ttH_VBF_BDT_df_2017.csv'))
#dataframes.append(pd.read_csv('2017/MC/DataFrames/tHq_VBF_BDT_df_2017.csv'))
#dataframes.append(pd.read_csv('2017/MC/DataFrames/tHW_VBF_BDT_df_2017.csv'))
data = pd.concat(dataframes, sort=False, axis=0 )

# Pre-selection cuts
data = data[data.diphotonMass>100.]
data = data[data.diphotonMass<180.]
data = data[data.leadPhotonPtOvM>0.333]
data = data[data.subleadPhotonPtOvM>0.25]

def mapping(map_list,stage):
    proc_list = []
    num_list = []
    proc = []
    for i in range(len(map_list)):
        proc_list.append(map_list[i][0])
        temp = []
        for j in range(len(map_list[i])-1):
            temp.append(map_list[i][j+1])
        num_list.append(temp)
    for i in stage:
        for j in range(len(num_list)):
            if i in num_list[j]:
                proc.append(proc_list[j])
    return proc

data['proc_original'] = mapping(map_list=map_def_2,stage=data['HTXS_stage1_2_cat_pTjet30GeV'])

# now I only want to keep the qqH - 7class
#data = data.drop(data[(data.proc_original == 'QQ2HQQ_FWDH') & (data.proc_original == 'WH') & (data.proc_original == 'ZH')].index)
data = data[data.proc_original != 'QQ2HQQ_FWDH']
data = data[data.proc_original != 'WH']
data = data[data.proc_original != 'ZH']

#Define the procs as the labels
#ggh: 0, VBF:1, VH: 2, ttH: 3
#num_categories = data['proc'].nunique()
#y_train_labels_num, y_train_labels_def = pd.factorize(data['proc'])

num_categories = data['proc_original'].nunique()
y_train_labels_num, y_train_labels_def = pd.factorize(data['proc_original'])

#Label definition:
print('Label Definition:')
label_def = []
for i in range(num_categories):
    label_def.append([i,y_train_labels_def[i]])
    print(i,y_train_labels_def[i])

data['proc_num'] = y_train_labels_num

y_train_labels = np.array(data['proc_original'])
#y_train_labels = np.array(data['proc'])
y_train_labels_num = np.array(data['proc_num'])
y_train_labels_hot = np_utils.to_categorical(y_train_labels_num, num_classes=num_categories)
weights = np.array(data['weight'])

data = data.drop(columns=['proc'])
data = data.drop(columns=['proc_num'])
data = data.drop(columns=['HTXS_stage_0'])
data = data.drop(columns=['HTXS_stage1_2_cat_pTjet30GeV'])

# well also need to only keep the qqH 7-class btw

# pTHjj and njets variable construction
# my soul has exited my body since I have tried every possible pandas way to do this ... I will turn to numpy arrays now for my own sanity
# most inefficient code ever written lessgoooo

leadJetPt = np.array(data['leadJetPt'])
leadJetPhi = np.array(data['leadJetPhi'])
subleadJetPt = np.array(data['subleadJetPt'])
subleadJetPhi = np.array(data['subleadJetPhi'])
leadPhotonPt = np.array(data['leadPhotonPt'])
leadPhotonPhi = np.array(data['leadPhotonPhi'])
subleadPhotonPt = np.array(data['subleadPhotonPt'])
subleadPhotonPhi = np.array(data['subleadPhotonPhi'])

# creating pTHjj variable
pTHjj = []
check = 0
for i in range(data.shape[0]):
    if leadJetPt[i] != -999.0 or leadJetPhi[i] != -999.0:
        px_jet1 = leadJetPt[i]*np.cos(leadJetPhi[i])
        py_jet1 = leadJetPt[i]*np.sin(leadJetPhi[i])
    else:
        px_jet1 = 0
        py_jet1 = 0
        check += 1
    if subleadJetPt[i] != -999.0 or subleadJetPhi[i] != -999.0:
        px_jet2 = subleadJetPt[i]*np.cos(subleadJetPhi[i])
        py_jet2 = subleadJetPt[i]*np.sin(subleadJetPhi[i])
    else:
        px_jet2 = 0
        py_jet2 = 0
        check += 1
    if leadPhotonPt[i] != -999.0 or leadPhotonPhi[i] != -999.0:
        px_ph1 = leadPhotonPt[i]*np.cos(leadPhotonPhi[i])
        py_ph1 = leadPhotonPt[i]*np.sin(leadPhotonPhi[i])
    else:
        px_ph1 = 0
        py_ph1 = 0
        check += 1
    if subleadPhotonPt[i] != -999.0 or subleadPhotonPhi[i] != -999.0:
        px_ph2 = subleadPhotonPt[i]*np.cos(subleadPhotonPhi[i])
        py_ph2 = subleadPhotonPt[i]*np.sin(subleadPhotonPhi[i])
    else:
        px_ph2 = 0
        py_ph2 = 0
        check += 1 

    px_sum = px_jet1 + px_jet2 + px_ph1 + px_ph2
    py_sum = py_jet1 + py_jet2 + py_ph1 + py_ph2

    if check == 4:
        pTHjj.append(-999.0)
    else:
        pTHjj.append(np.sqrt(px_sum**2 + py_sum**2))    
    check = 0

data['pTHjj'] = pTHjj
print('Done 1')

# creating n-jets variable
njets = []
num_jet = 0
for i in range(data.shape[0]):
    if leadJetPt[i] != -999.0:
        if subleadJetPt[i] != -999.0:
            num_jet = 2
        else:
            num_jet = 1
    else:
        num_jet = 0
    njets.append(num_jet)
data['njets'] = njets
print('Done 2')

# manually setting cuts
dijetmass = np.array(data['dijetMass'])
njets = np.array(data['njets'])
diphotonpt = np.array(data['diphotonPt'])
diphotonjetspt = np.array(data['pTHjj'])

proc = []
for i in range(data.shape[0]):
    #print('eeee')
    if njets[i] == 0 or njets[i] == 1:
        proc_value = 'rest'
    else:
        if dijetmass[i] < 350:
            if dijetmass[i] > 60 and dijetmass[i] < 120:
                proc_value = 'VH'
            else:
                proc_value = 'rest'
        else:
            if diphotonpt[i] < 200:
                if  dijetmass[i] < 700 and diphotonjetspt[i] < 25:
                    proc_value = 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25'
                elif dijetmass[i] < 700 and diphotonjetspt[i] >= 25:
                    proc_value = 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25'
                elif dijetmass[i] >= 700 and diphotonjetspt[i] < 25:
                    proc_value = 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25'
                else:
                    proc_value = 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_GT25'
            else: 
                proc_value = 'QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200'
    proc.append(proc_value)
data['proc_new'] = proc

# Confusion Matrix

y_true = data['proc_original']
y_pred = data['proc_new']

cm = confusion_matrix(y_true=y_true,y_pred=y_pred)

def plot_confusion_matrix(cm,classes,normalize=True,title='Confusion matrix',cmap=plt.cm.Blues):
    fig, ax = plt.subplots(figsize = (10,10))
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks,classes)
    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        for i in range(len(cm[0])):
            for j in range(len(cm[1])):
                cm[i][j] = float("{:.2f}".format(cm[i][j]))
    thresh = cm.max()/2.
    print(cm)
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    for i, j in product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,cm[i,j],horizontalalignment='center',color='white' if cm[i,j]>thresh else 'black')
    plt.tight_layout()
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicted label')
    name = 'plotting/Cuts/Cuts_qqH_Sevenclass_Confusion_Matrix'
    fig.savefig(name, dpi = 1200)

plot_confusion_matrix(cm,binNames,normalize=True)



# what I need to do
# qqH - 7 class cuts: to have a dataframe with only qqH [qqH proc from 1.2 and proc_new - our predictions from cuts] + subsub - yes
# qqH - 7class NN and BDT - yes
# confusion matrices of qqH 7-class NN, BDT, cuts - yes
# check labeling - fixed this

# inform Philipp about everything




















"""
# well also need to only keep the qqH 7-class btw
# using pandas
exit(0)
data['proc_new'] = ''
print 'okay1'
data.loc[(data.njets == 2) & (data.diphotonPt < 120) & (data.dijetMass >60), 'proc_new'] = 'VH'
print 'okay4'
data.loc[(data.njets == 2) & (data.dijetMass > 120) & (data.dijetMass <60) & (data.dijetMass<350), 'proc_new'] = 'rest'
print 'okay5'
data.loc[(data.njets == 2) & (data.diphotonPt < 200) & (data.dijetMass <700) & (data.pTHjj<25), 'proc_new'] = 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25'
print 'okay6'
data.loc[(data.njets == 2) & (data.diphotonPt < 200) & (data.dijetMass <700) & (data.pTHjj>=25), 'proc_new'] = 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25'
print 'okay7'
data.loc[(data.njets == 2) & (data.diphotonPt < 200) & (data.dijetMass >=700) & (data.pTHjj>=25), 'proc_new'] = 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_GT25'
print 'okay8'
data.loc[(data.njets == 2) & (data.diphotonPt < 200) & (data.dijetMass >= 700) & (data.pTHjj<25), 'proc_new'] = 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25'
print 'okay9'
data.loc[(data.njets == 2) & (data.diphotonPt > 200), 'proc_new'] = 'QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200'
print 'okay3'
data.loc[(data.njets == 0), 'proc_new'] = 'rest'
print 'okay2'
data.loc[(data.njets == 1), 'proc_new'] = 'rest'
print 'Done 3'
exit(0)

#Preselection cuts
data = data[data.diphotonMass>100.]
data = data[data.diphotonMass<180.]
data = data[data.leadPhotonPtOvM>0.333]
data = data[data.subleadPhotonPtOvM>0.25]


# our numpy arrays again - variables for cuts
dijetmass = np.array(data['dijetMass'])
njets = np.array(data['njets'])
diphotonpt = np.array(data['diphotonPt'])
diphotonjetspt = np.array(data['pTHjj'])

proc = []
for i in range(data.shape[0]):
    if njets[i] == 0 or njets == 1:
        proc_value = 'rest'
    else:
        if dijetmass[i] < 350:
            if dijetmass[i] > 60 and dijetmass[i] < 120:
                proc_value = 'VH'
            else:
                proc_value = 'rest'
        else:
            if diphotonpt[i] < 200:
                if  dijetmass[i] < 700 and diphotonjetspt[i] < 25:
                    proc_value = 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25'
                elif dijetmass[i] < 700 and diphotonjetspt[i] >= 25:
                    proc_value = 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25'
                elif dijetmass[i] >= 700 and diphotonjetspt[i] < 25:
                    proc_value = 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25'
                else:
                    proc_value = 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_GT25'
            else: 
                proc_value = 'QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200'
    proc.append(proc_value)
data['proc'] = proc
exit(0)
# using pandas
data.loc[data.njets == 0, 'proc'] = 'rest'
data.loc[data.njets == 1, 'proc'] = 'rest'
data.loc[data.njets == 2 & data.diphotonPt < 120 & data.dijetMass >60, 'proc'] = 'VH'
data.loc[data.njets == 2 & data.dijetMass > 120 & data.dijetMass <60 & data.dijetMass<350, 'proc'] = 'rest'
data.loc[data.njets == 2 & data.diphotonPt < 200 & data.dijetMass <700 & data.pTHjj<25, 'proc'] = 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25'
data.loc[data.njets == 2 & data.diphotonPt < 200 & data.dijetMass <700 & data.pTHjj>=25, 'proc'] = 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25'
data.loc[data.njets == 2 & data.diphotonPt < 200 & data.dijetMass >=700 & data.pTHjj>=25, 'proc'] = 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_GT25'
data.loc[data.njets == 2 & data.diphotonPt < 200 & data.dijetMass >= 700 & data.pTHjj<25, 'proc'] = 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25'
data.loc[data.njets == 2 & data.diphotonPt > 200, 'proc'] = 'QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200'



print('New Variables')
print('pTHjj: ', data['pTHjj'])
print('njets: ', data['njets'])
print('proc: ', data['procs'])


# plotting some output scores now
train_vars = ['diphotonPt', 'diphotonMass', 'diphotonCosPhi', 'diphotonEta','diphotonPhi', 'diphotonSigmaMoM',
     'dijetMass', 'dijetAbsDEta', 'dijetDPhi', 'dijetCentrality',
     'dijetPt','dijetEta','dijetPhi','dijetMinDRJetPho','dijetDiphoAbsDEta',
     'leadPhotonEta', 'leadPhotonIDMVA', 'leadPhotonEn', 'leadPhotonPt', 'leadPhotonPhi', 'leadPhotonPtOvM',
     'leadJetPt', 'leadJetPUJID', 'leadJetBTagScore', 'leadJetMass',
     'leadJetDiphoDEta','leadJetDiphoDPhi','leadJetEn','leadJetEta','leadJetPhi',
     'subleadPhotonEta', 'subleadPhotonIDMVA', 'subleadPhotonPhi',
     'subleadPhotonEn','subleadPhotonPt', 'subleadPhotonPtOvM',
     'subleadJetDiphoDPhi','subleadJetDiphoDEta',
     'subleadJetPt', 'subleadJetPUJID', 'subleadJetBTagScore', 'subleadJetMass',
     'subleadJetEn','subleadJetEta','subleadJetPhi',
     'subsubleadJetEn','subsubleadJetPt','subsubleadJetEta','subsubleadJetPhi', 'subsubleadJetBTagScore', 
     'subsubleadJetMass',
     'metPt','metPhi','metSumET',
     'nSoftJets',
     'leadElectronEn', 'leadElectronMass', 'leadElectronPt', 'leadElectronEta', 'leadElectronPhi', 'leadElectronCharge',
     'leadMuonEn', 'leadMuonMass', 'leadMuonPt', 'leadMuonEta', 'leadMuonPhi', 'leadMuonCharge',
     'subleadElectronEn', 'subleadElectronMass', 'subleadElectronPt', 'subleadElectronEta', 'subleadElectronPhi', 'subleadElectronCharge', 
     'subleadMuonEn', 'subleadMuonMass', 'subleadMuonPt', 'subleadMuonEta', 'subleadMuonPhi', 'subleadMuonCharge'
     ]

list_vars = ['dijetmass']


# what i want to do before I hand over to Philipp
# do new qqH (7 class)
# check labeling that is correct
# do this cut-based approach nicely







exit(0)
"""

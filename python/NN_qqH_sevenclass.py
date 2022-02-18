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

#Define key quantities

#HPs
#Original
#num_epochs = 2
#batch_size = 400
#val_split = 0.3
#learning_rate = 0.001

#Optimized according to 4class
num_epochs = 50
batch_size = 64
val_split = 0.3
learning_rate = 0.0001


#STXS mapping
map_def_0 = [['ggH',10,11],['qqH',20,21,22,23],['WH',30,31],['ZH',40,41],['ttH',60,61],['tH',80,81]]
map_def_1 = [
['ggH',100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116],
['QQ2HQQ_FWDH',200],
['QQ2HQQ_0J',201],
['QQ2HQQ_1J',202],
['QQ2HQQ_GE2J_MJJ_0_60',203],
['QQ2HQQ_GE2J_MJJ_60_120',204],
['QQ2HQQ_GE2J_MJJ_120_350',205],
['QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200',206],
['QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25',207],
['QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25',208],
['QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25',209],
['QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_GT25',210],
['WH',300,301,302,303,304,305],
['ZH',400,401,402,403,404,405],
['ttH',600,601,602,603,604,605],
['tH',800,801]
]

epochs = np.linspace(1,num_epochs,num_epochs,endpoint=True).astype(int) #For plotting
#binNames = ['ggH','qqH','ZH','WH','ttH','tH'] 
#binNames = ['ggH','QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_GT25','QQ2HQQ_GE2J_MJJ_120_350',
#'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25','QQ2HQQ_1J','QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25',
#'QQ2HQQ_0J','QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200','QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25',
#'QQ2HQQ_GE2J_MJJ_0_60','QQ2HQQ_GE2J_MJJ_60_120','QQ2HQQ_FWDH','ZH','WH','ttH','tH'] 
binNames = ['ggH','qqH1','qqH2','qqH3','qqH4','qqH5','qqH6','qqH7','qqH8','qqH9','qqH10','qqH0','ZH','WH','ttH','tH'] 
bins = 50

#Directories
modelDir = 'neural_networks/models/'
plotDir  = 'neural_networks/plots/'

#Define the input features
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


#Add proc and weight to shuffle with data
train_vars.append('proc')
train_vars.append('weight')
train_vars.append('HTXS_stage_0')
train_vars.append('HTXS_stage1_2_cat_pTjet30GeV')

#Load the dataframe
dataframes = []
dataframes.append(pd.read_csv('2017/MC/DataFrames/ggH_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/VBF_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/VH_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/ttH_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/tHq_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/tHW_VBF_BDT_df_2017.csv'))
df = pd.concat(dataframes, sort=False, axis=0 )

#dataframe of train_vars
data = df[train_vars]

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

print('New Variables')
print('pTHjj: ', data['pTHjj'])
print('njets: ', data['njets'])


exit(0)





# today
# new vars - done
# cut-based - have a base
# new qqh + subsub
# labels and confusion



















"""
# waiiiit i got a new smart way
# ...

# getting rid of the -999
data.loc[:, 'leadJetPt_new'] = data['leadJetPt']
data['leadJetPt_new'].replace(-999.0, 0)
data.loc[:, 'leadJetPhi_new'] = data['leadJetPhi'] 
data['leadJetPhi_new'].replace(-999.0, 0)

data.loc[:, 'subleadJetPt_new'] = data['subleadJetPt']
data['subleadJetPt_new'].replace(-999.0, 0)
data.loc[:, 'subleadJetPhi_new'] = data['subleadJetPhi'] 
data['subleadJetPhi_new'].replace(-999.0, 0)

data.loc[:, 'leadPhotonPt_new'] = data['leadPhotonPt']
data['leadPhotonPt_new'].replace(-999.0, 0)
data.loc[:, 'leadPhotonPhi_new'] = data['leadPhotonPhi'] 
data['leadPhotonPhi_new'].replace(-999.0, 0)

data.loc[:, 'subleadPhotonPt_new'] = data['subleadPhotonPt']
data['subleadPhotonPt_new'].replace(-999.0, 0)
data.loc[:, 'subleadPhotonPhi_new'] = data['subleadPhotonPhi'] 
data['subleadPhotonPhi_new'].replace(-999.0, 0)


# performing calculation

data.loc[:, 'px_sum'] = data['leadJetPt_new'] * np.cos(data['leadJetPhi_new']) + data['subleadJetPt_new'] * np.cos(data['subleadJetPhi_new']) + data['leadPhotonPt_new'] * np.cos(data['leadPhotonPhi_new']) + data['subleadPhotonPt_new'] * np.cos(data['subleadPhotonPhi_new'])
data.loc[:, 'py_sum'] = data['leadJetPt_new'] * np.sin(data['leadJetPhi_new']) + data['subleadJetPt_new'] * np.sin(data['subleadJetPhi_new']) + data['leadPhotonPt_new'] * np.sin(data['leadPhotonPhi_new']) + data['subleadPhotonPt_new'] * np.sin(data['subleadPhotonPhi_new'])
data.loc[:, 'pTHjj'] = np.sqrt(data['px_sum']**2 + data['py_sum']**2)
print data['pTHjj']
exit(0)

# check for wrong calculation due to replacement of -999 to 0 and remove
# check if all 
data.loc[data['']]


train_w_df.loc[train_w_df.proc == 'WH','weight'] = (train_w_df[train_w_df['proc'] == 'WH']['weight'] * ggh_sum_w / wh_sum_w)


# check for wrong calculation due to the replacement of -999 to 0 and remove

data.loc[df['leadJetPt_new'] == -999.0, 'leadJetPt_new'] = 0


exit 
data.loc[data['leadJetPt'] != -999.0, 'new'] = 0

df['col1'] = df.apply(lambda x: x['col3'] if x['col1'] < x['col2'] else x['col1'], axis=1)

df.loc[row_indexes,'elderly']="yes"
data.loc[data['leadJetPt'] != -999.0, 'new'] = data['leadJetPt']

data['leadJetPt_new'] = np.where(data['leadJetPt'] != -999.0, data['leadJetPt'], 0)

data['check_jet1'] = data.where(data['leadJetPt'] == -999.0 | data['leadJetPhi'] == -999.0)
data['check_jet2'] = data.where((data['subleadJetPt'] == -999) | (data['subleadJetPhi'] == -999), True, False)
data['check_ph1'] = data.where((data['leadPhotonPt'] == -999) | (data['leadPhotonPhi'] == -999), True, False)
data['check_ph2'] = data.where(data['subleadPhotonPt'] == -999 or data['subleadPhotonPhi'] == -999, True, False)




print 'Trying a'

for i in range(data.shape[0]):
    # creating pTHjj variable
    if data['leadJetPt'].iloc[i] != -999 | data['leadJetPhi'].iloc[i] != -999:
        px_jet1 = data['leadJetPt'].iloc[i]*np.cos(data['leadJetPhi'].iloc[i])
        py_jet1 = data['leadJetPt'].iloc[i]*np.sin(data['leadJetPhi'].iloc[i])
    else:
        px_jet1 = 0
        py_jet1 = 0
    if data['subleadJetPt'].iloc[i] != -999 | data['subleadJetPhi'].iloc[i] != -999:
        px_jet2 = data['subleadJetPt'].iloc[i]*np.cos(data['subleadJetPhi'].iloc[i])
        py_jet2 = data['subleadJetPt'].iloc[i]*np.sin(data['subleadJetPhi'].iloc[i])
    else:
        px_jet2 = 0
        py_jet2 = 0
    if data['leadPhotonPt'].iloc[i] != -999 | data['leadPhotonPhi'].iloc[i] != -999:
        px_ph1 = data['leadPhotonPt'].iloc[i]*np.cos(data['leadPhotonPhi'].iloc[i])
        py_ph1 = data['leadPhotonPt'].iloc[i]*np.sin(data['leadPhotonPhi'].iloc[i])
    else:
        px_ph1 = 0
        py_ph1 = 0
    if data['subleadPhotonPt'].iloc[i] != -999 | data['subleadPhotonPhi'].iloc[i] != -999:
        px_ph2 = data['subleadPhotonPt'].iloc[i]*np.cos(data['subleadPhotonPhi'].iloc[i])
        py_ph2 = data['subleadPhotonPt'].iloc[i]*np.sin(data['subleadPhotonPhi'].iloc[i])
    else:
        px_ph1 = 0
        py_ph2 = 0

    px_sum = px_jet1 + px_jet2 + px_ph1 + px_ph2
    py_sum = py_jet1 + py_jet2 + py_ph1 + py_ph2

    data['pTHjj'] = np.sqrt(px_sum**2 + py_sum**2)

print(data['pTHjj'])
# all only the ones actually
# remember to make if pthjj equal to -999 if 0 so i can remove it later no? just make sure that i dont actually count it here epidi
# apla evala 0 to make it easier alla in reality einai epidi einai undefined

# relabeling
data['check_jet1'] = data.where(data['leadJetPt'] == -999 | data['leadJetPhi'] == -999)
data['check_jet2'] = data.where((data['subleadJetPt'] == -999) | (data['subleadJetPhi'] == -999), True, False)
data['check_ph1'] = data.where((data['leadPhotonPt'] == -999) | (data['leadPhotonPhi'] == -999), True, False)
data['check_ph2'] = data.where(data['subleadPhotonPt'] == -999 or data['subleadPhotonPhi'] == -999, 1, 0)

"""

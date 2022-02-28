import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
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
from keras.callbacks import LearningRateScheduler
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
num_epochs = 200
batch_size = 64
test_split = 0.2
val_split = 0.1
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
map_def_2 = [
['QQ2HQQ_FWDH',200],
['qqH_Rest', 201, 202, 203, 205],
['QQ2HQQ_GE2J_MJJ_60_120',204],
['QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200',206],
['QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25',207],
['QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25',208],
['QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25',209],
['QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_GT25',210],
['WH',300,301,302,303,304,305],
['ZH',400,401,402,403,404,405],
]

epochs = np.linspace(1,num_epochs,num_epochs,endpoint=True).astype(int) #For plotting
#binNames = ['ggH','qqH','ZH','WH','ttH','tH'] 
#binNames = ['ggH','QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_GT25','QQ2HQQ_GE2J_MJJ_120_350',
#'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25','QQ2HQQ_1J','QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25',
#'QQ2HQQ_0J','QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200','QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25',
#'QQ2HQQ_GE2J_MJJ_0_60','QQ2HQQ_GE2J_MJJ_60_120','QQ2HQQ_FWDH','ZH','WH','ttH','tH'] 
binNames = ['qqH_Rest',
            'QQ2HQQ_GE2J_MJJ_60_120',
            'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25',
            'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25',
            'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25',
            'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_GT25',
            'QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200']

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
     #'leadElectronEn', 'leadElectronMass', 'leadElectronPt', 'leadElectronEta', 'leadElectronPhi', 'leadElectronCharge',
     #'leadMuonEn', 'leadMuonMass', 'leadMuonPt', 'leadMuonEta', 'leadMuonPhi', 'leadMuonCharge',
     #'subleadElectronEn', 'subleadElectronMass', 'subleadElectronPt', 'subleadElectronEta', 'subleadElectronPhi', 'subleadElectronCharge', 
     #'subleadMuonEn', 'subleadMuonMass', 'subleadMuonPt', 'subleadMuonEta', 'subleadMuonPhi', 'subleadMuonCharge'
     ]

"""
train_vars = ['dijetMass', 'diphotonPt', 'leadJetPt', 'leadJetPhi', 'subleadJetPt', 'subleadJetPhi', 
            'leadPhotonPt', 'leadPhotonPhi', 'subleadPhotonPt', 'subleadPhotonPhi', 'diphotonMass', 'leadPhotonPtOvM', 'subleadPhotonPtOvM']
"""

#Add proc and weight to shuffle with data
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
df = pd.concat(dataframes, sort=False, axis=0 )

#dataframe of train_vars
data = df[train_vars]
print 'we are here'


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


#Preselection cuts

data = data[data.diphotonMass>100.]
data = data[data.diphotonMass<180.]
data = data[data.leadPhotonPtOvM>0.333]
data = data[data.subleadPhotonPtOvM>0.25]
print 'we are also here'

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

#data['proc_new'] = mapping(map_list=map_def,stage=data['HTXS_stage_0'])
data['proc_new'] = mapping(map_list=map_def_2,stage=data['HTXS_stage1_2_cat_pTjet30GeV'])

# now I only want to keep the qqH - 7class
data = data[data.proc_new != 'QQ2HQQ_FWDH']
data = data[data.proc_new != 'WH']
data = data[data.proc_new != 'ZH']

print 'also here'


#Define the procs as the labels
#ggh: 0, VBF:1, VH: 2, ttH: 3
num_categories = data['proc_new'].nunique()

proc_new = np.array(data['proc_new'])
#Assign the numbers in the same order as the binNames above
y_train_labels_num = []
for i in proc_new:
    if i == 'qqH_Rest':
        y_train_labels_num.append(0)
    if i == 'QQ2HQQ_GE2J_MJJ_60_120':
        y_train_labels_num.append(1)
    if i == 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25':
        y_train_labels_num.append(2)
    if i == 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25':
        y_train_labels_num.append(3)
    if i == 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25':
        y_train_labels_num.append(4)
    if i == 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_GT25':
        y_train_labels_num.append(5)
    if i == 'QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200':
        y_train_labels_num.append(6)

data['proc_num'] = y_train_labels_num

#Shuffle dataframe
data = data.sample(frac=1)

y_train_labels = np.array(data['proc_new'])
y_train_labels_num = np.array(data['proc_num'])
y_train_labels_hot = np_utils.to_categorical(y_train_labels_num, num_classes=num_categories)
weights = np.array(data['weight'])

#Remove proc after shuffle
data = data.drop(columns=['proc','weight','proc_num','HTXS_stage_0','HTXS_stage1_2_cat_pTjet30GeV','proc_new'])

#data = data.drop(columns = ['leadJetPt', 'leadJetPhi', 'subleadJetPt', 'subleadJetPhi', 'leadPhotonPt', 'leadPhotonPhi', 'subleadPhotonPt', 'subleadPhotonPhi', 'diphotonMass', 'leadPhotonPtOvM', 'subleadPhotonPtOvM'])

#Set -999.0 values to -10.0 to decrease effect on scaling 
data = data.replace(-999.0,-10.0) 

#Scaling the variables to a range from 0-1
scaler = MinMaxScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

#Input shape for the first hidden layer
num_inputs  = data_scaled.shape[1]

#Splitting the dataframe into training and test
x_train, x_test, y_train, y_test, train_w, test_w, proc_arr_train, proc_arr_test = train_test_split(data_scaled, y_train_labels_hot, weights, y_train_labels, test_size = test_split, shuffle = True)

#Initialize the model
model=Sequential([Dense(units=400,input_shape=(num_inputs,),activation='relu'),
                Dense(units=400,activation='relu'),
                #Dense(units=100,activation='relu'),
                Dense(units=num_categories,activation='softmax')]) #For multiclass NN use softmax

#Compile the model
model.compile(optimizer=Adam(lr=learning_rate),loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()

# callbacks
def scheduler(epoch, lr):
    print("epoch: ", epoch)
    if epoch < 10:
        print("lr: ", lr)
        return lr
    else:
        lr *= math.exp(-0.1)
        print("lr: ", lr)
        return lr
callback_lr = LearningRateScheduler(scheduler)
#callback_earlystop = EarlyStopping(monitor='val_loss', min_delta = 0.001, patience=10)

#Equalizing weights
train_w_df = pd.DataFrame()
train_w = 300 * train_w # to make loss function O(1)
train_w_df['weight'] = train_w
train_w_df['proc'] = proc_arr_train

qqh1_sum_w = train_w_df[train_w_df['proc'] == 'qqH_Rest']['weight'].sum()
qqh2_sum_w = train_w_df[train_w_df['proc'] == 'QQ2HQQ_GE2J_MJJ_60_120']['weight'].sum()
qqh3_sum_w = train_w_df[train_w_df['proc'] == 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25']['weight'].sum()
qqh4_sum_w = train_w_df[train_w_df['proc'] == 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25']['weight'].sum()
qqh5_sum_w = train_w_df[train_w_df['proc'] == 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25']['weight'].sum()
qqh6_sum_w = train_w_df[train_w_df['proc'] == 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_GT25']['weight'].sum()
qqh7_sum_w = train_w_df[train_w_df['proc'] == 'QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200']['weight'].sum()

train_w_df.loc[train_w_df.proc == 'QQ2HQQ_GE2J_MJJ_60_120','weight'] = (train_w_df[train_w_df['proc'] == 'QQ2HQQ_GE2J_MJJ_60_120']['weight'] * qqh1_sum_w / qqh2_sum_w)
train_w_df.loc[train_w_df.proc == 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25','weight'] = (train_w_df[train_w_df['proc'] == 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25']['weight'] * qqh1_sum_w / qqh3_sum_w)
train_w_df.loc[train_w_df.proc == 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25','weight'] = (train_w_df[train_w_df['proc'] == 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25']['weight'] * qqh1_sum_w / qqh4_sum_w)
train_w_df.loc[train_w_df.proc == 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25','weight'] = (train_w_df[train_w_df['proc'] == 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25']['weight'] * qqh1_sum_w / qqh5_sum_w)
train_w_df.loc[train_w_df.proc == 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_GT25','weight'] = (train_w_df[train_w_df['proc'] == 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_GT25']['weight'] * qqh1_sum_w / qqh6_sum_w)
train_w_df.loc[train_w_df.proc == 'QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200','weight'] = (train_w_df[train_w_df['proc'] == 'QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200']['weight'] * qqh1_sum_w / qqh7_sum_w)
train_w = np.array(train_w_df['weight'])

#Training the model
history = model.fit(x=x_train,y=y_train,batch_size=batch_size,epochs=num_epochs,sample_weight=train_w,shuffle=True,verbose=2, validation_split = val_split, callbacks=[callback_lr])

# Output Score
y_pred_test = model.predict_proba(x=x_test)
x_test['proc'] = proc_arr_test
x_test['weight'] = test_w

x_test['output_score_qqh1'] = y_pred_test[:,0]
x_test['output_score_qqh2'] = y_pred_test[:,1]
x_test['output_score_qqh3'] = y_pred_test[:,2]
x_test['output_score_qqh4'] = y_pred_test[:,3]
x_test['output_score_qqh5'] = y_pred_test[:,4]
x_test['output_score_qqh6'] = y_pred_test[:,5]
x_test['output_score_qqh7'] = y_pred_test[:,6]

output_score_qqh1 = np.array(y_pred_test[:,0])
output_score_qqh2 = np.array(y_pred_test[:,1])
output_score_qqh3 = np.array(y_pred_test[:,2])
output_score_qqh4 = np.array(y_pred_test[:,3])
output_score_qqh5 = np.array(y_pred_test[:,4])
output_score_qqh6 = np.array(y_pred_test[:,5])
output_score_qqh7 = np.array(y_pred_test[:,6])

x_test_qqh1 = x_test[x_test['proc'] == 'qqH_Rest']
x_test_qqh2 = x_test[x_test['proc'] == 'QQ2HQQ_GE2J_MJJ_60_120']
x_test_qqh3 = x_test[x_test['proc'] == 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25']
x_test_qqh4 = x_test[x_test['proc'] == 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25']
x_test_qqh5 = x_test[x_test['proc'] == 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25']
x_test_qqh6 = x_test[x_test['proc'] == 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_GT25']
x_test_qqh7 = x_test[x_test['proc'] == 'QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200']

qqh1_w = x_test_qqh1['weight'] / x_test_qqh1['weight'].sum()
qqh2_w = x_test_qqh2['weight'] / x_test_qqh2['weight'].sum()
qqh3_w = x_test_qqh3['weight'] / x_test_qqh3['weight'].sum()
qqh4_w = x_test_qqh4['weight'] / x_test_qqh4['weight'].sum()
qqh5_w = x_test_qqh5['weight'] / x_test_qqh5['weight'].sum()
qqh6_w = x_test_qqh6['weight'] / x_test_qqh6['weight'].sum()
qqh7_w = x_test_qqh7['weight'] / x_test_qqh7['weight'].sum()
total_w = x_test['weight'] / x_test['weight'].sum()

# for later weights
qqh0_w2 = x_test_qqh1['weight']
qqh1_w2 = x_test_qqh2['weight']
qqh2_w2 = x_test_qqh3['weight']
qqh3_w2 = x_test_qqh4['weight']
qqh4_w2 = x_test_qqh5['weight']
qqh5_w2 = x_test_qqh6['weight']
qqh6_w2 = x_test_qqh7['weight']

# stack the weights 
#weights_array = np.stack((qqh1_w, qqh2_w, qqh3_w, qqh4_w, qqh5_w, qqh6_w, qqh7_w), axis = -1)

#Accuracy Score
y_pred = y_pred_test.argmax(axis=1)
y_true = y_test.argmax(axis=1)
print 'Accuracy score: '
NNaccuracy = accuracy_score(y_true, y_pred)
print(NNaccuracy)

#Confusion Matrix
cm = confusion_matrix(y_true=y_true,y_pred=y_pred)


def plot_output_score(data='output_score_qqh', density=False,):
    #Can then change it to plotting proc
    print('Plotting',data)
    output_score_qqh1 = np.array(x_test_qqh1[data])
    output_score_qqh2 = np.array(x_test_qqh2[data])
    output_score_qqh3 = np.array(x_test_qqh3[data])
    output_score_qqh4 = np.array(x_test_qqh4[data])
    output_score_qqh5 = np.array(x_test_qqh5[data])
    output_score_qqh6 = np.array(x_test_qqh6[data])
    output_score_qqh7 = np.array(x_test_qqh7[data])

    fig, ax = plt.subplots()
    #ax.hist(output_score_qqh0, bins=50, label='FWDH', histtype='step',weights=qqh0_w)
    ax.hist(output_score_qqh1, bins=50, label='rest', histtype='step',weights=qqh1_w)
    ax.hist(output_score_qqh2, bins=50, label='QQ2HQQ_GE2J_MJJ_60_120', histtype='step',weights=qqh2_w)
    ax.hist(output_score_qqh3, bins=50, label='QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200', histtype='step',weights=qqh3_w)
    ax.hist(output_score_qqh4, bins=50, label='QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25', histtype='step',weights=qqh4_w)
    ax.hist(output_score_qqh5, bins=50, label='QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25', histtype='step',weights=qqh5_w)
    ax.hist(output_score_qqh6, bins=50, label='QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25', histtype='step',weights=qqh6_w)
    ax.hist(output_score_qqh7, bins=50, label='QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_GT25', histtype='step',weights=qqh7_w)
    plt.legend()
    plt.title('Output Score')
    plt.ylabel('Fraction of Events')
    plt.xlabel('NN Score')
    name = 'plotting/NN_plots/NN_qqH_sevenclass_'+data
    plt.savefig(name, dpi = 1200)


#Confusion Matrix
def plot_confusion_matrix(cm,classes,normalize=True,title='Confusion matrix',cmap=plt.cm.Blues):
    fig, ax = plt.subplots(figsize = (10,10))
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=90)
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
    name = 'plotting/NN_plots/NN_qqH_Sevenclass_Confusion_Matrix'
    fig.savefig(name, dpi = 1200)

def plot_performance_plot(cm=cm,labels=binNames, normalize = True):
    cm = cm.astype('float')/cm.sum(axis=0)[np.newaxis,:]
    for i in range(len(cm[0])):
        for j in range(len(cm[1])):
            cm[i][j] = float("{:.3f}".format(cm[i][j]))
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize = (12,12))
    plt.rcParams.update({
    'font.size': 9})
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks,labels,rotation=90)
    bottom = np.zeros(len(labels))
    for i in range(len(cm)):
        ax.bar(labels, cm[i,:],label=labels[i],bottom=bottom)
        bottom += np.array(cm[i,:])
    plt.legend()
    current_bottom, current_top = ax.get_ylim()
    ax.set_ylim(bottom=0, top=current_top*1.3)
    #plt.title('Performance Plot')
    plt.ylabel('Fraction of events')
    ax.set_xlabel('Events', ha='right',x=1,size=9) #, x=1, size=13)
    name = 'plotting/NN_plots/NN_qqH_Sevenclass_Performance_Plot'
    plt.savefig(name, dpi = 500)
    plt.show()

def plot_roc_curve(signal = 'QQ2HQQ_GE2J_MJJ_60_120', y_test = y_test, y_pred_test = y_pred_test, x_test = x_test):
    # sample weights
    # find weighted average 
    fig, ax = plt.subplots()
    #y_pred_test  = clf.predict_proba(x_test)
    for i in range(num_categories):
        if binNames[i] == signal:
            sig_y_test  = np.where(y_test==i, 1, 0)
            print('sig_y_test', type(sig_y_test))
            y_pred_test_array = y_pred_test[:,i]
            print('y_pred_test_array', type(y_pred_test_array))
            print('Here')
            #fpr_keras, tpr_keras, thresholds_keras = roc_curve(sig_y_test, y_pred_test_array, sample_weight = test_w)
            test_w = test_w.reshape(1, -1)
            print('test_w', type(test_w))
            auc = roc_auc_score(sig_y_test, y_pred_test_array, sample_weight = test_w)
            print('auc: ', auc)
            print('Here')
            fpr_keras.sort()
            tpr_keras.sort()
            auc_test = auc(fpr_keras, tpr_keras)
            ax.plot(fpr_keras, tpr_keras, label = 'AUC = {0} with Signal =  {1}'.format(round(auc_test, 3), binNames[i]))
    ax.legend(loc = 'lower right', fontsize = 'x-small')
    ax.set_xlabel('Background Efficiency', ha='right', x=1, size=9)
    ax.set_ylabel('Signal Efficiency',ha='right', y=1, size=9)
    ax.grid(True, 'major', linestyle='solid', color='grey', alpha=0.5)
    name = 'plotting/NN_plots/NN_qqH_Sevenclass_ROC_curve_' + signal
    plt.savefig(name, dpi = 600)
    print("Plotting ROC Curve")
    plt.close()

plot_performance_plot()

#plot_roc_curve()
"""
plot_output_score(data='output_score_qqh1')
plot_output_score(data='output_score_qqh2')
plot_output_score(data='output_score_qqh3')
plot_output_score(data='output_score_qqh4')
plot_output_score(data='output_score_qqh5')
plot_output_score(data='output_score_qqh6')
plot_output_score(data='output_score_qqh7')
"""
#plot_accuracy()
#plot_loss()
plot_confusion_matrix(cm,binNames,normalize=True)


#save as a pickle file
#trainTotal.to_pickle('%s/nClassNNTotal.pkl'%frameDir)
#print 'frame saved as %s/nClassNNTotal.pkl'%frameDir
#Read in pickle file
#trainTotal = pd.read_pickle(opts.dataFrame)
#print 'Successfully loaded the dataframe'



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

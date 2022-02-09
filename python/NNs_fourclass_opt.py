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
import math
from math import pi
import h5py
from itertools import product
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, roc_curve, auc, roc_auc_score, auc
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential 
from keras.initializers import RandomNormal 
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Nadam, adam, Adam
from keras.regularizers import l2 
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.utils import np_utils 
from keras.metrics import categorical_crossentropy, binary_crossentropy


#Define key quantities, use to tune NN
num_epochs = 50
batch_size = 64
test_split = 0.15
val_split = 0.15
#learning_rate = 0.001

epochs = np.linspace(1,num_epochs,num_epochs,endpoint=True).astype(int) #For plotting
binNames = ['ggH','qqH','VH','ttH'] 
bins = 50

#Directories
modelDir = 'neural_networks/models/'
plotDir  = 'neural_networks/plots/'

# Kat
train_vars = ['diphotonPt', 'diphotonMass', 'diphotonCosPhi', 'diphotonEta','diphotonPhi', 'diphotonSigmaMoM',
     'dijetMass', 'dijetAbsDEta', 'dijetDPhi', 'dijetCentrality',
     'dijetPt','dijetEta','dijetPhi','dijetMinDRJetPho','dijetDiphoAbsDEta',
     'leadPhotonEta', 'leadPhotonPtOvM', 'leadPhotonEn', 'leadPhotonPt', 'leadPhotonPhi',
     'leadJetPt', 'leadJetPUJID', 'leadJetBTagScore', 'leadJetMass',
     'leadJetDiphoDEta','leadJetDiphoDPhi','leadJetEn','leadJetEta','leadJetPhi',
     'subleadPhotonEta', 'subleadPhotonPtOvM', 'subleadPhotonIDMVA', 'subleadPhotonPhi',
     'subleadPhotonEn','subleadPhotonPt', 
     'subleadJetDiphoDPhi','subleadJetDiphoDEta',
     'subleadJetPt', 'subleadJetPUJID', 'subleadJetBTagScore', 'subleadJetMass',
     'subleadJetEn','subleadJetEta','subleadJetPhi',
     'subsubleadJetEn','subsubleadJetPt','subsubleadJetEta','subsubleadJetPhi', 'subsubleadJetBTagScore', 
     'subsubleadJetMass',
     'metPt','metPhi','metSumET',
     'nSoftJets'
     ]

#Add proc and weight to shuffle with data
train_vars.append('proc')
train_vars.append('weight')
train_vars.append('HTXS_stage_0')
#train_vars.append('HTXS_stage1_2_cat_pTjet30GeV')

#Load the dataframe
dataframes = []
dataframes.append(pd.read_csv('2017/MC/DataFrames/ggH_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/VBF_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/VH_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/ttH_VBF_BDT_df_2017.csv'))
df = pd.concat(dataframes, sort=False, axis=0 )

#Data dataframe
#dataframes = []
#dataframes.append(pd.read_csv('2017/Data/DataFrames/Data_VBF_ggH_BDT_df_2017.csv'))
#x_data = pd.concat(dataframes, sort=False, axis=0 )

#dataframe of train_vars
data = df[train_vars]

#Preselection cuts
data = data[data.diphotonMass>100.]
data = data[data.diphotonMass<180.]
data = data[data.leadPhotonPtOvM>0.333]
data = data[data.subleadPhotonPtOvM>0.25]

proc_temp = np.array(data['HTXS_stage_0'])
proc_new = []
for i in proc_temp:
    if i == 10 or i == 11:
        proc_new.append('ggH')
    elif i == 20 or i == 21 or i == 22 or i == 23:
        proc_new.append('qqH')
    elif i == 30 or i == 31 or i == 40 or i == 41:
        proc_new.append('VH')
    elif i == 60 or i == 61:
        proc_new.append('ttH')
    else:
        proc_new.append(i)
        print(i)
data['proc_new'] = proc_new

#Define the procs as the labels
#ggh: 0, VBF:1, VH: 2, ttH: 3
num_categories = data['proc_new'].nunique()
y_train_labels_num, y_train_labels_def = pd.factorize(data['proc_new'])

#Label definition:
print('Label Definition:')
label_def = []
for i in range(num_categories):
    label_def.append([y_train_labels_def[i]])
    print(i,y_train_labels_def[i])

data['proc_num'] = y_train_labels_num

#Shuffle dataframe
data = data.sample(frac=1)

y_train_labels = np.array(data['proc_new'])
y_train_labels_num = np.array(data['proc_num'])
y_train_labels_hot = np_utils.to_categorical(y_train_labels_num, num_classes=num_categories)
weights = np.array(data['weight'])

#Remove proc after shuffle
data = data.drop(columns=['proc'])
data = data.drop(columns=['weight'])
data = data.drop(columns=['proc_num'])
data = data.drop(columns=['HTXS_stage_0'])
data = data.drop(columns=['proc_new'])

#Set -999.0 values to -10.0 to decrease effect on scaling 
data = data.replace(-999.0,-10.0) 

#Scaling the variables to a range from 0-1
scaler = MinMaxScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

#Input shape for the first hidden layer
num_inputs  = data_scaled.shape[1]

def model_fn_activ_analysis(nodes, activation, num_layers, learning_rate, num_inputs = num_inputs):
    model=Sequential()
    # beggining layer
    model.add(Dense(units=nodes,input_shape=(num_inputs,),activation=activation))
    # intermediate layers
    for i in range(num_layers):
        model.add(Dense(units=nodes,activation=activation))
    # final layer
    model.add(Dense(units=num_categories,activation='softmax')) 
        
    #Compile the model
    model.compile(optimizer=Adam(lr=learning_rate),loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()

    return model

def scheduler(epoch, lr):
    print("epoch: ", epoch)
    if epoch < 10:
        print("lr: ", lr)
        return lr
    else:
        lr *= math.exp(-0.1)
        print("lr: ", lr)
        return lr

 # --------------------------------

# Optimization - grid search
nodes = [100, 200, 400]
layers = [1,5,10]
learn_rate = [0.01, 0.001, 0.0001]
activation_value = 'relu'
#activation = ['relu', 'sigmoid', 'softmax']

param_comb = []
acc_values = []

for nodes_value in nodes:
    for layers_value in layers:
        for lr_value in learn_rate:
            #Splitting the dataframe into training and test
            x_train, x_test, y_train, y_test, train_w, test_w, proc_arr_train, proc_arr_test = train_test_split(data_scaled, y_train_labels_hot, weights, y_train_labels, test_size = test_split, shuffle = True)
            #Initialize the model
            model = model_fn_activ_analysis(nodes = nodes_value, activation = activation_value, num_layers = layers_value, learning_rate = lr_value)
            #Equalizing weights
            train_w_df = pd.DataFrame()
            train_w = 100 * train_w # to make loss function O(1)
            train_w_df['weight'] = train_w
            train_w_df['proc'] = proc_arr_train
            qqh_sum_w = train_w_df[train_w_df['proc'] == 'qqH']['weight'].sum()
            ggh_sum_w = train_w_df[train_w_df['proc'] == 'ggH']['weight'].sum()
            vh_sum_w = train_w_df[train_w_df['proc'] == 'VH']['weight'].sum()
            tth_sum_w = train_w_df[train_w_df['proc'] == 'ttH']['weight'].sum()
            train_w_df.loc[train_w_df.proc == 'qqH','weight'] = (train_w_df[train_w_df['proc'] == 'qqH']['weight'] * ggh_sum_w / qqh_sum_w)
            train_w_df.loc[train_w_df.proc == 'VH','weight'] = (train_w_df[train_w_df['proc'] == 'VH']['weight'] * ggh_sum_w / vh_sum_w)
            train_w_df.loc[train_w_df.proc == 'ttH','weight'] = (train_w_df[train_w_df['proc'] == 'ttH']['weight'] * ggh_sum_w / tth_sum_w)
            train_w = np.array(train_w_df['weight'])

            # Callbacks
            callback_lr = LearningRateScheduler(scheduler)
            callback_earlystop = EarlyStopping(monitor='val_loss', min_delta = 0.001, patience=10)

            #Training the model
            history = model.fit(x=x_train,y=y_train,batch_size=batch_size,epochs=num_epochs,sample_weight=train_w,validation_split = val_split,shuffle=True,verbose=2,callbacks=[callback_lr,callback_earlystop])


            y_pred_test = model.predict_proba(x=x_test)
            #Accuracy Score
            y_pred = y_pred_test.argmax(axis=1)
            y_true = y_test.argmax(axis=1)
            #print 'Accuracy score: '
            NNaccuracy = accuracy_score(y_true, y_pred)
            #print(NNaccuracy)
            param_comb.append([nodes_value,layers_value,lr_value])
            acc_values.append(NNaccuracy)
            print('Parameter combination: ', nodes_value,layers_value,lr_value)
            print('Accuracy score: ', NNaccuracy)

max_value = [np.max(acc_values)]
index_max_value = acc_values.index(max_value)
best_param_comb = param_comb[index_max_value]
print('Best parameter combination:',best_param_comb)
print('Best Acc Score:',max_value)
np.savetxt('neural_networks/models/best_param_combo_nn_fourclass.txt', best_param_comb, delimiter=',')
np.savetxt('neural_networks/models/best_acc_score_nn_fourclass.txt', max_value, delimiter=',')







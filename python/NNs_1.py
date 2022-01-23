# imports
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
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
#from sklearn import cross_validation
from os import path, system
from array import array
#from addRowFunctions import addPt, truthDipho, reco, diphoWeight, altDiphoWeight, truthJets, jetWeight, truthClass, jetPtToClass, procWeight
#from otherHelpers import prettyHist, getAMS, computeBkg, getRealSigma
from root_numpy import tree2array, fill_hist
#import usefulStyle as useSty
from math import pi

from keras.models import Sequential
from keras.initializers import RandomNormal
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import *
from keras.optimizers import Nadam
from keras.optimizers import adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import h5py
from keras.layers import Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.metrics import binary_crossentropy

#Define key quantities, use to tune NN
num_epochs = 4
batch_size = 400
val_split = 0.3
learning_rate = 0.001

epochs = np.linspace(1,num_epochs,num_epochs,endpoint=True).astype(int) #For plotting
binNames = ['ggH','VBF'] 
bins = 50

#Directories
modelDir = 'neural_networks/models/'
plotDir  = 'neural_networks/plots/'

#Define the variables, this is an arbitrary selection
# Philipp
"""
 = ['leadPhotonIDMVA','subleadPhotonIDMVA',
    'diphotonMass','diphotonPt',
    'leadPhotonPtOvM','subleadPhotonPtOvM',
    'leadPhotonEta','subleadPhotonEta',
    'dijetMass','dijetAbsDEta','dijetDPhi','diphotonCosPhi',
    'leadJetPUJID','subleadJetPUJID','subsubleadJetPUJID',
    'leadJetPt','leadJetEn','leadJetEta','leadJetPhi',
    'subleadJetPt','subleadJetEn','subleadJetEta','subleadJetPhi',
    'subsubleadJetPt','subsubleadJetEn','subsubleadJetEta','subsubleadJetPhi'
    ]
    """
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

#Add proc to shuffle with data
train_vars.append('proc')

#Load the dataframe
dataframes = []
dataframes.append(pd.read_csv('2017/MC/DataFrames/VBF_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/ggH_VBF_BDT_df_2017.csv'))
df = pd.concat(dataframes, sort=False, axis=0 )

#Data dataframe
#dataframes = []
#dataframes.append(pd.read_csv('2017/Data/DataFrames/Data_VBF_ggH_BDT_df_2017.csv'))
#x_data = pd.concat(dataframes, sort=False, axis=0 )

#dataframe of train_vars
data = df[train_vars]

#Shuffle dataframe
data = data.sample(frac=1)

#Define the procs as the labels
y_train_labels = np.array(data['proc'])
y_train_labels_hot = np.where(y_train_labels=='VBF',1,0)
#y_train_labels_hot = np_utils.to_categorical(y_train_labels_num, num_classes=2) # removing one hot encoding for binary classifier | keep for multiclass

#Remove proc after shuffle
data = data.drop(columns=['proc'])

#Preselection cuts
data = data[data.diphotonMass>100.]
data = data[data.diphotonMass<180.]
data = data[data.leadPhotonPtOvM>0.333]
data = data[data.subleadPhotonPtOvM>0.25]

#Set -999.0 values to -10.0 to decrease effect on scaling 
data = data.replace(-999.0,-10.0) 

#Scaling the variables to a range from 0-1
scaler = MinMaxScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

#Input shape for the first hidden layer
num_inputs  = data_scaled.shape[1]

#Splitting the dataframe into training and test
x_train, x_test, y_train, y_test, train_w, test_w, proc_arr_train, proc_arr_test = train_test_split(data_scaled, y_train_labels_hot, df['weight'], df['proc'], test_size = val_split, shuffle=True)

#Initialize the model
model=Sequential([Dense(units=100,input_shape=(num_inputs,),activation='relu'),
                Dense(units=100,activation='relu'),
                #Dense(units=100,activation='relu'),
                Dense(units=1,activation='sigmoid')]) #activation = 'sigmoid': binary classifier | activation = 'softmax': multiclass classifier

#Compile the model
model.compile(optimizer=Adam(lr=learning_rate),loss='binary_crossentropy',metrics=['accuracy'])

model.summary()

#Training the model
history = model.fit(x=x_train,y=y_train,batch_size=batch_size,epochs=num_epochs,shuffle=True,verbose=2)

# --------------------------------------------------------------
# OUTPUT SCORE
y_pred_test = model.predict_proba(x = x_test)
x_test['proc'] = proc_arr_test.tolist()
x_test['weight'] = test_w.to_numpy()
x_test_vbf = x_test[x_test['proc'] == 'VBF']
x_test_ggh = x_test[x_test['proc'] == 'ggH']
# now weights
vbf_w = x_test_vbf['weight'] / x_test_vbf['weight'].sum()
ggh_w = x_test_ggh['weight'] / x_test_ggh['weight'].sum()

x_test_vbf = x_test_vbf.drop(columns=['proc'])
x_test_ggh = x_test_ggh.drop(columns=['proc'])
x_test_vbf = x_test_vbf.drop(columns=['weight'])
x_test_ggh = x_test_ggh.drop(columns=['weight'])

output_vbf = model.predict_proba(x=x_test_vbf)
output_ggh = 1 - model.predict_proba(x=x_test_ggh)

# ----
# ROC CURVE
# testing
#mask_vbf = (y_test[:] == 1)
#mask_ggh = (y_test[:] == 0)
#y_test = np.concatenate((y_test[mask_vbf], y_test[mask_ggh]), axis = None)
#y_pred_test = np.concatenate((output_vbf, output_ggh), axis = None)
#y_pred_test = model.predict_proba(x = x_test_old)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_test)
auc_keras_test = roc_auc_score(y_test, y_pred_test)
#np.savetxt('neural_networks/models/nn_roc_fpr.csv', fpr_keras, delimiter=',')
#np.savetxt('neural_networks/models/nn_roc_tpr.csv', tpr_keras, delimiter=',')
print("Area under ROC curve for testing: ", auc_keras_test)

# training
y_pred_train = model.predict_proba(x = x_train)
fpr_keras_tr, tpr_keras_tr, thresholds_keras = roc_curve(y_train, y_pred_train)
auc_keras_train = roc_auc_score(y_train, y_pred_train)
print("Area under ROC curve for training: ", auc_keras_train)



# -------------------------------------------------------------------
# PLOTTING AYE

# OUTPUT SCORE
def plot_output_score(signal=output_vbf,bkg=output_ggh,name='plotting/NN_plots/NN_Output_Score',signal_label='VBF',bkg_label='ggH',bins=50,density=False,histtype='step', sig_w = vbf_w, bkg_w = ggh_w):
     fig, ax = plt.subplots()
     ax.hist(signal, bins=bins, label=signal_label, weights = sig_w, histtype=histtype)
     ax.hist(bkg, bins=bins, label=bkg_label, weights = bkg_w, histtype=histtype) 
     ax.set_xlabel('Output Score', ha='right', x=1, size=9)
     ax.grid(True, 'major', linestyle='solid', color='grey', alpha=0.5)
     ax.legend()
     plt.savefig(name, dpi = 200)
     print("Plotting Output Score")
     plt.close()

# ROC CURVE
def roc_score(fpr_train = fpr_keras_tr, tpr_train = tpr_keras_tr,fpr_test = fpr_keras, tpr_test = tpr_keras, name = 'plotting/NN_plots/NN_ROC_curve', train = False):
     fig, ax = plt.subplots()
     if train:
          ax.plot(fpr_train, tpr_train, label = 'Train')
     ax.plot(fpr_test, tpr_test, label = 'Test')
     ax.legend()
     ax.set_xlabel('Background Efficiency', ha='right', x=1, size=9)
     ax.set_ylabel('Signal Efficiency',ha='right', y=1, size=9)
     ax.grid(True, 'major', linestyle='solid', color='grey', alpha=0.5)
     plt.savefig(name, dpi = 200)
     print("Plotting ROC Score")
     plt.close()
     
# TRAIN VS TEST



# ----------------------------------------------------------------------
# RUN
plot_output_score()
roc_score(train = True)


'''

#Evaluate performance, no priors
y_prob = model.predict(x=X_test_scaled,batch_size=10,verbose=0)
y_pred = y_prob.argmax(axis=1)
print 'Accuracy score: '
NNaccuracy = accuracy_score(y_test, y_pred)  #, sample_weight=w_mc_test)
print(NNaccuracy)

#history = modelfit(
#    X_train_scaled,
#    y_train_onehot,
#    sample_weight=w_mc_train,
#    batch_size=batchSize,
#    epochs=1000,
#    shuffle=True,
#    callbacks=callbacks # add function to print stuff out there
#    )
'''

'''
#save as a pickle file
#trainTotal.to_pickle('%s/nClassNNTotal.pkl'%frameDir)
#print 'frame saved as %s/nClassNNTotal.pkl'%frameDir
#Read in pickle file
#trainTotal = pd.read_pickle(opts.dataFrame)
#print 'Successfully loaded the dataframe'
'''



# RANDOM STUFF THAT DIDNT WANT TO DELETE
#model.compile(optimizer=Adam(learning_rate=0.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#model.compile(optimizer=Adam,loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#model.fit(x=x_train_scaled,y=y_train_onehot,batch_size=400,epochs=3,shuffle=True,verbose=2)  # verbose = num of classes
#print(model.summary())
# commented them out for now until I finish the roc curve thing

#The 61 columns are:
#'Unnamed: 0', 'diphotonMass', 'diphotonPt', 'diphotonEta',
#'diphotonPhi', 'diphotonCosPhi', 'diphotonSigmaMoM',
#'leadPhotonIDMVA', 'leadPhotonPtOvM', 'leadPhotonEta',
#'leadPhotonEn', 'leadPhotonMass', 'leadPhotonPt', 'leadPhotonPhi',
#'subleadPhotonIDMVA', 'subleadPhotonPtOvM', 'subleadPhotonEta',
#'subleadPhotonEn', 'subleadPhotonMass', 'subleadPhotonPt',
#'subleadPhotonPhi', 'dijetMass', 'dijetPt', 'dijetEta', 'dijetPhi',
#'dijetDPhi', 'dijetAbsDEta', 'dijetCentrality', 'dijetMinDRJetPho',
#'dijetDiphoAbsDEta', 'leadJetPUJID', 'leadJetPt', 'leadJetEn',
#'leadJetEta', 'leadJetPhi', 'leadJetMass', 'leadJetBTagScore',
#'leadJetDiphoDEta', 'leadJetDiphoDPhi', 'subleadJetPUJID',
#'subleadJetPt', 'subleadJetEn', 'subleadJetEta', 'subleadJetPhi',
#'subleadJetMass', 'subleadJetBTagScore', 'subleadJetDiphoDPhi',
#'subleadJetDiphoDEta', 'subsubleadJetPUJID', 'subsubleadJetPt',
#'subsubleadJetEn', 'subsubleadJetEta', 'subsubleadJetPhi',
#'subsubleadJetMass', 'subsubleadJetBTagScore', 'weight',
#'centralObjectWeight', 'nSoftJets', 'genWeight', 'proc', 'year']
#Need to remove: 'Unnamed: 0','weight','centralObjectWeight', 'genWeight', 'year'
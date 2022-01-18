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

modelDir = 'neural_networks/models/'
plotDir  = 'neural_networks/plots/'

#Define the variables, this is an arbitrary selection
# Philipp
"""
train_vars = ['leadPhotonIDMVA','subleadPhotonIDMVA',
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
allVars = ['diphotonPt', 'diphotonMass', 'diphotonCosPhi', 'diphotonEta','diphotonPhi', 'diphotonSigmaMoM',
     'dijetMass', 'dijetAbsDEta', 'dijetDPhi', 'dijetCentrality',
     'dijetPt','dijetEta','dijetPhi','dijetMinDRJetPho','dijetDiphoAbsDEta',
     'leadPhotonEta', 'leadPhotonIDMVA', 'leadPhotonEn', 'leadPhotonPt', 'leadPhotonPhi',
     'leadJetPt', 'leadJetPUJID', 'leadJetBTagScore', 'leadJetMass',
     'leadJetDiphoDEta','leadJetDiphoDPhi','leadJetEn','leadJetEta','leadJetPhi',
     'subleadPhotonEta', 'subleadPhotonIDMVA', 'subleadPhotonPhi',
     'subleadPhotonEn','subleadPhotonPt', 
     'subleadJetDiphoDPhi','subleadJetDiphoDEta',
     'subleadJetPt', 'subleadJetPUJID', 'subleadJetBTagScore', 'subleadJetMass',
     'subleadJetEn','subleadJetEta','subleadJetPhi',
     'subsubleadJetEn','subsubleadJetPt','subsubleadJetEta','subsubleadJetPhi', 'subsubleadJetBTagScore', 
     'subsubleadJetMass',
     'metPt','metPhi','metSumET',
     'nSoftJets'
     ]

#dataframes = []
#dataframes.append(pd.read_csv('2017/Data/DataFrames/Data_VBF_ggH_BDT_df_2017.csv'))
#df_data = pd.concat(dataframes, sort=False, axis=0 )

dataframes = []
dataframes.append(pd.read_csv('2017/MC/DataFrames/VBF_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/ggH_VBF_BDT_df_2017.csv'))
x_train = pd.concat(dataframes, sort=False, axis=0 )
#x_test = pd.concat(dataframes, sort=False, axis=0 )

#Shuffle dataframe
x_train = x_train.sample(frac=1)

#Define the procs as the labels
y_train_labels = np.array(x_train['proc'])#,columns='proc')
#y_train_labels_num = pd.DataFrame()
#y_train_labels_num['proc'] = np.where(y_train_labels=='VBF',1,0)
y_train_labels_num = np.where(y_train_labels=='VBF',1.0,0.0)

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

x_train = x_train.drop(columns=['Unnamed: 0','weight','centralObjectWeight', 'genWeight', 'year','proc'])
#x_test = x_test.drop(columns=['Unnamed: 0','weight','centralObjectWeight', 'genWeight', 'year'])

#Preselection cuts
#    'diphotonMass > 100 and diphotonMass < 180 and leadPhotonPtOvM > 0.333 and subleadPhotonPtOvM > 0.25'
x_train = x_train[x_train.diphotonMass>100.]
x_train = x_train[x_train.diphotonMass<180.]
x_train = x_train[x_train.leadPhotonPtOvM>0.333]
x_train = x_train[x_train.subleadPhotonPtOvM>0.25]

#Set -999.0 values to -10.0 to decrease effect on scaling
x_train = x_train.replace(-999.0,-10.0)

#Need to scale the variables to a range from 0-1
scaler = MinMaxScaler()
x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)

num_inputs  = x_train_scaled.shape[1]

model=Sequential([Dense(units=100,input_shape=(63,),activation='relu'),  # activation = 'relu' - hidden layer
                Dense(units=100,activation='relu'),
                #Dense(units=100,activation='relu'),
                Dense(units=1,activation='softmax')])  # activation = 'softmax' - output layer

model.compile(optimizer=Adam(lr=0.01),loss='binary_crossentropy',metrics=['accuracy'])
#model.compile(optimizer=Adam(learning_rate=0.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#model.compile(optimizer=Adam,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.summary()

#Not sure why this works, x_train_scaled is a df, y_train_labels is an array, I thought they would have to be the same
model.fit(x=x_train_scaled,y=y_train_labels_num,batch_size=400,epochs=5,shuffle=True,verbose=2)  # verbose = num of classes
print(model.summary())
# commented them out for now until I finish the roc curve thing


# further stuff that we are working on
"""
# Output Score
x_test['NN_output'] = model.predict(x=x_test,batch_size=400,verbose=0)
y_pred_vbf = x_test[x_test['proc'] == 'VBF']['NN_output']
y_pred_vbf = x_test[x_test['proc'] == 'ggH']['NN_output']

bins = 50
axes.hist(y_pred_vbf, bins=bins, label='VBF', density = True, histtype='step') #weights=sig_w_true
axes.hist(y_pred_ggh, bins=bins, label='ggH', density = True, histtype='step') #weights=sig_w_true
plt.savefig('neural_networks/models/plots/output_score_trial', dpi = 200)


# ROC curve attempts
# split to train and test data
x_train, x_test, y_train, y_test = train_test_split(x_train_scaled, y_train_labels_num, test_size = 0.3)
model.fit(x=x_train, y=y_train, batch_size=400, epochs=5, shuffle=True, verbose=2)  # verbose = num of classes
print(model.summary())

y_pred_test = model.predict(x_test).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_test)
y_pred_keras = model.predict(x_test)
auc_keras_test = auc(fpr_keras, tpr_keras)
print("Area under ROC curve for testing: ", auc_keras_test)

plt.plot(fpr_keras, tpr_keras)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig('neural_networks/models/plots/roc_trial', dpi = 200)
plt.close()


# things to add - for the train/test break - Philipp
#x_test = pd.concat(dataframes, sort=False, axis=0 )
#x_test = x_test.drop(columns=['Unnamed: 0','weight','centralObjectWeight', 'genWeight', 'year','proc'])
#y_pred = model.predict(x=x_test,batch_size=400,verbose=0)
"""

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
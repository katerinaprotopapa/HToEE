# imports
import argparse
import pandas as pd
import numpy as np
import matplotlib
import math
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import pickle
import ROOT as r
r.gROOT.SetBatch(True)
import sys
from itertools import product
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from os import path, system
from array import array
from root_numpy import tree2array, fill_hist
from math import pi
from keras.models import Sequential
from keras.initializers import RandomNormal
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import LeakyReLU
from keras.layers import *
from keras.optimizers import Nadam
from keras.optimizers import adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.utils import np_utils
import h5py
from keras.layers import Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.metrics import binary_crossentropy
import keras.metrics


class NN_binary():
    """
    Functions to train a binary class NN for VBF/ggH separation

    params dataframe: dataframe of the VBF and ggH signal
    type dataframe: pandas dataframe

    """
    def __init__(self, dataframe, activation = 'relu', nodes = 120, lr = 0.001, test_split = 0.30, val_split = 0.10, batch_size = 400, num_epochs = 10, num_layers = 1):
        self.data = dataframe
        self.activation = activation 
        self.num_inputs = 0
        self.nodes = nodes
        self.lr = lr
        self.val_split =  val_split
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.test_split = test_split
        self.num_layers = num_layers

        self.output_vbf = 0
        self.output_ggh = 0
        self.vbf_w = 0
        self.ggh_w = 0

    
    def initialize(self):
        data = self.data
        #Preselection cuts
        data = data[data.diphotonMass>100.]
        data = data[data.diphotonMass<180.]
        data = data[data.leadPhotonPtOvM>0.333]
        data = data[data.subleadPhotonPtOvM>0.25]

        #Shuffle dataframe
        data = data.sample(frac=1)

        # re-grouping to ggH, qqH, VH leptonic and ttH
        # this syntax: pandas.DataFrame.loc[condition, column_label] = new_value
        """data.loc[data.HTXS_stage_0 == 10, 'proc_new'] = 'ggH'
        data.loc[data.HTXS_stage_0 == 20 or data.HTXS_stage_0 == 22, 'proc_new'] = 'qqH'
        data.loc[data.HTXS_stage_0 == 30 or data.HTXS_stage_0 == 40, 'proc_new'] = 'VH'
        data.loc[data.HTXS_stage_0 == 60, 'proc_new'] = 'ttH'"""

        #Define the procs as the labels
        y_train_labels = np.array(data['proc'])
        y_train_labels_hot = np.where(y_train_labels=='VBF',1,0)
        #y_train_labels_hot = np_utils.to_categorical(y_train_labels_num, num_classes=2) # removing one hot encoding for binary classifier | keep for multiclass
        weights = np.array(data['weight'])

        #Remove proc and weight after shuffle
        data = data.drop(columns=['proc'])
        data = data.drop(columns=['weight']) 
        data = data.drop(columns=['HTXS_stage_0'])
        #data = data.drop(columns=['HTXS_stage1_2_cat_pTjet30GeV'])

        #Set -999.0 values to -10.0 to decrease effect on scaling 
        data = data.replace(-999.0,-10.0) 

        #Scaling the variables to a range from 0-1
        scaler = MinMaxScaler()
        data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

        #Input shape for the first hidden layer
        self.num_inputs  = data_scaled.shape[1]
    
        return data_scaled, y_train_labels_hot, weights, y_train_labels

    
    def model_fn(self):
        #Initialize the model

        print("nodes: ", self.nodes)
        self.model=Sequential([Dense(units=self.nodes,input_shape=(self.num_inputs,),activation=self.activation),
                Dense(units=self.nodes,activation=self.activation),
                Dense(units=self.nodes,activation=self.activation),
                Dense(units=1,activation='sigmoid')]) #activation = 'sigmoid': binary classifier | activation = 'softmax': multiclass classifier

        #Compile the model
        self.model.compile(optimizer=Adam(lr=self.lr),loss='binary_crossentropy',metrics=['accuracy'])
        self.model.summary()
    
    def model_fn_activ_analysis(self):

        self.model=Sequential()
        # beggining layer
        self.model.add(Dense(units=self.nodes,input_shape=(self.num_inputs,),activation=self.activation))
        # intermediate layers
        for i in range(self.num_layers):
            self.model.add(Dense(units=self.nodes,activation=self.activation))
        # final layer
        self.model.add(Dense(units=1,activation='sigmoid')) 
        
        #Compile the model
        self.model.compile(optimizer=Adam(lr=self.lr),loss='binary_crossentropy',metrics=['accuracy'])
        self.model.summary()

    def scheduler(self, epoch, lr):
        print("epoch: ", epoch)
        if epoch < 10:
            print("lr: ", self.lr)
            return self.lr
        else:
            self.lr *= math.exp(-0.1)
            print("lr: ", self.lr)
            return self.lr

    def training(self, data_scaled, y_train_labels_hot, weights, y_train_labels, layers_analysis_bool = False):
        x_train, x_test, y_train, y_test, train_w, test_w, proc_arr_train, proc_arr_test = train_test_split(data_scaled, y_train_labels_hot, weights, y_train_labels, test_size = self.test_split, shuffle=True)
        #print("x_test", x_test)
        #if layers_analysis_bool == False:
        #    self.model_fn()  # compile the model
        #else:
        #    self.model_fn_activ_analysis()
        self.model_fn_activ_analysis()

        # Equalizing training weights
        train_w_df = pd.DataFrame()
        train_w = 100 * train_w # to make loss function O(1)
        train_w_df['weight'] = train_w
        train_w_df['proc'] = proc_arr_train
        vbf_sum_w = train_w_df[train_w_df['proc'] == 'VBF']['weight'].sum()
        ggh_sum_w = train_w_df[train_w_df['proc'] == 'ggH']['weight'].sum()
        train_w_df.loc[train_w_df.proc == 'VBF','weight'] = (train_w_df[train_w_df['proc'] == 'VBF']['weight'] * ggh_sum_w / vbf_sum_w) # THIS LINE !
        train_w = np.array(train_w_df['weight'])

        #Training the model
        #history = self.model.fit(x=x_train,y=y_train,batch_size=self.batch_size,epochs=self.num_epochs,sample_weight=train_w,shuffle=True,verbose=2)
        #print("shape x_test ", x_test.shape)

        callback_lr = LearningRateScheduler(self.scheduler)
        callback_earlystop = EarlyStopping(monitor='val_loss', min_delta = 0.001, patience=10)
        history = self.model.fit(x=x_train,y=y_train,batch_size=self.batch_size,epochs=self.num_epochs,sample_weight=train_w,shuffle=True,verbose=2, validation_split = self.val_split, callbacks=[callback_lr,callback_earlystop])
        #round(self.model.optimizer.lr.np(), 5)

        return x_train, x_test, y_train, y_test, train_w, test_w, proc_arr_train, proc_arr_test

    
    def output_score(self, x_test, proc_arr_test, test_w):
        #print("size x_test: ", x_test.shape)
        y_pred_test = self.model.predict_proba(x=x_test)  
        x_test['proc'] = proc_arr_test 
        x_test['weight'] = test_w 
        x_test['output_score'] = y_pred_test
        x_test_vbf = x_test[x_test['proc'] == 'VBF'] 
        x_test_ggh = x_test[x_test['proc'] == 'ggH'] 
        # Weights 
        vbf_w = x_test_vbf['weight'] / x_test_vbf['weight'].sum() 
        ggh_w = x_test_ggh['weight'] / x_test_ggh['weight'].sum() 

        output_vbf = np.array(x_test_vbf['output_score']) 
        output_ggh = np.array(x_test_ggh['output_score']) 

        x_test_vbf = x_test_vbf.drop(columns=['proc'])
        x_test_ggh = x_test_ggh.drop(columns=['proc'])
        x_test_vbf = x_test_vbf.drop(columns=['weight'])
        x_test_ggh = x_test_ggh.drop(columns=['weight'])
        x_test_vbf = x_test_vbf.drop(columns=['output_score'])
        x_test_ggh = x_test_ggh.drop(columns=['output_score'])

        x_test = x_test.drop(columns=['proc'])
        x_test = x_test.drop(columns=['weight'])
        x_test = x_test.drop(columns=['output_score'])
        #print("size x_test: ", x_test.shape)

        #print("output_vbf_size ",len(output_vbf))
        #print("vbf_w size ", len(vbf_w))
        self.output_vbf = output_vbf
        self.output_ggh = output_ggh
        self.vbf_w = vbf_w
        self.ggh_w = ggh_w

        self.plot_output_score()
    
    def roc_curve(self, x_train, x_test, y_train, y_test, train_w, test_w, plot_roc = True):    
        y_pred_test = self.model.predict_proba(x=x_test)    
        # testing
        fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_test, sample_weight = test_w)
        fpr_keras.sort()
        tpr_keras.sort()
        auc_test = auc(fpr_keras, tpr_keras)
        #auc_keras_test = roc_auc_score(y_test, y_pred_test)
        print("Area under ROC curve for testing: ", auc_test)

        # training
        y_pred_train = self.model.predict_proba(x = x_train)
        fpr_keras_tr, tpr_keras_tr, thresholds_keras = roc_curve(y_train, y_pred_train, sample_weight = train_w)
        fpr_keras_tr.sort()
        tpr_keras_tr.sort()
        auc_train = auc(fpr_keras_tr,tpr_keras_tr)
        #auc_keras_train = roc_auc_score(y_train, y_pred_train, sample_weight = train_w)
        print("Area under ROC curve for training: ", auc_train)

        if plot_roc:
            self.plot_roc_score(fpr_keras_tr, tpr_keras_tr, fpr_keras, tpr_keras, train = True)
        
        compare_with_bdt = False
        if compare_with_bdt:
            np.savetxt('neural_networks/models/y_test.csv', y_test, delimiter=',')
            np.savetxt('neural_networks/models/y_pred_test.csv', y_pred_test, delimiter=',')
            np.savetxt('neural_networks/models/test_w.csv', test_w, delimiter=',')

        return auc_test

    def plot_output_score(self, name='plotting/NN_plots/NN_Output_Score',signal_label='VBF',bkg_label='ggH',bins=50,density=False,histtype='step'):
        signal = self.output_vbf
        bkg = self.output_ggh
        sig_w = self.vbf_w
        bkg_w = self.ggh_w
        fig, ax = plt.subplots()
        #print("signal ",len(signal))
        #print("sig_w size ", len(sig_w))
        ax.hist(signal, bins=bins, label=signal_label, weights = sig_w, histtype=histtype)
        ax.hist(bkg, bins=bins, label=bkg_label, weights = bkg_w, histtype=histtype) 
        ax.set_xlabel('Output Score', ha='right', x=1, size=9)
        ax.grid(True, 'major', linestyle='solid', color='grey', alpha=0.5)
        ax.legend()
        plt.savefig(name, dpi = 200)
        print("Plotting Output Score")
        plt.close()

    def plot_roc_score(self, fpr_keras_tr, tpr_keras_tr, fpr_keras, tpr_keras, name = 'plotting/NN_plots/NN_ROC_curve', train = False):
        fpr_train = fpr_keras_tr
        tpr_train = tpr_keras_tr
        fpr_test = fpr_keras
        tpr_test = tpr_keras
        fig, ax = plt.subplots()
        if train:
            ax.plot(fpr_train, tpr_train, label = 'Train')
        ax.plot(fpr_test, tpr_test, label = 'Test')
        ax.legend()
        ax.set_xlabel('Background Efficiency', ha='right', x=1, size=9)
        ax.set_ylabel('Signal Efficiency',ha='right', y=1, size=9)
        ax.grid(True, 'major', linestyle='solid', color='grey', alpha=0.5)
        plt.savefig(name, dpi = 200)
        print("Plotting ROC Curve")
        plt.close()


    def run(self, output_score = False, roc_curve_bool = False):

        data_scaled, y_train_labels_hot, weights, y_train_labels = self.initialize()  # make the final dataframe
        x_train, x_test, y_train, y_test, train_w, test_w, proc_arr_train, proc_arr_test = self.training(data_scaled, y_train_labels_hot, weights, y_train_labels) # train the model
        print('Done')

        if output_score:
            self.output_score(x_test, proc_arr_test, test_w)
            x_test = x_test.drop(columns=['proc'])
            x_test = x_test.drop(columns=['weight'])
            x_test = x_test.drop(columns=['output_score'])

        if roc_curve_bool:
            auc_test = self.roc_curve(x_train, x_test, y_train, y_test, train_w, test_w)
            return auc_test

    def nodes_analysis(self, num_nodes = 10, increase_nodes = 10, nodes_analysis_plot = True):

        auc_scores = []
        nodes = []

        data_scaled, y_train_labels_hot, weights, y_train_labels = self.initialize()  # make the final dataframe
        for i in range(num_nodes):
            x_train, x_test, y_train, y_test, train_w, test_w, proc_arr_train, proc_arr_test = self.training(data_scaled, y_train_labels_hot, weights, y_train_labels) # train the model
            auc_test = self.roc_curve(x_train, x_test, y_train, y_test, plot_roc = False)
            auc_scores.append(auc_test)
            nodes.append(self.nodes)
            self.nodes += increase_nodes
            print("i ", i)

        print("AUC scores: ", auc_scores)
        print("Nodes: ", nodes)

        if nodes_analysis_plot:
            self.nodes_analysis_plot(auc_scores, nodes)
    
    def activation_analysis(self, activation_list, activation_analysis_plot = True):

        auc_scores = []
        activ_list = []

        data_scaled, y_train_labels_hot, weights, y_train_labels = self.initialize()  # make the final dataframe
        for i in range(len(activation_list)):
            self.activation = activation_list[i]
            x_train, x_test, y_train, y_test, train_w, test_w, proc_arr_train, proc_arr_test = self.training(data_scaled, y_train_labels_hot, weights, y_train_labels) # train the model
            auc_test = self.roc_curve(x_train, x_test, y_train, y_test, plot_roc = False)
            auc_scores.append(auc_test)
            activ_list.append(self.activation)
            print("i ", i)

        print("AUC scores: ", auc_scores)
        print("Activation list: ", activ_list)

        if activation_analysis_plot:
            self.activation_analysis_plot(auc_scores, activ_list)
    
    def layers_analysis(self, num_intermediate_layers = 1, layers_analysis_plot = True):

        auc_scores = []
        num_int_layers = []

        data_scaled, y_train_labels_hot, weights, y_train_labels = self.initialize()  # make the final dataframe
        for i in range(num_intermediate_layers):   
            self.num_layers = i
            x_train, x_test, y_train, y_test, train_w, test_w, proc_arr_train, proc_arr_test = self.training(data_scaled, y_train_labels_hot, weights, y_train_labels, layers_analysis_bool = True) # train the model
            auc_test = self.roc_curve(x_train, x_test, y_train, y_test, plot_roc = False)
            auc_scores.append(auc_test)
            num_int_layers.append(self.num_layers)
            print("i ", i)

        print("AUC scores: ", auc_scores)
        print("Number of intermediate layers: ", num_int_layers)

        if layers_analysis_plot:
            self.layers_analysis_plot(auc_scores, num_int_layers)

    def nodes_analysis_plot(self, auc_scores, nodes, name = 'plotting/NN_plots/NN_nodes_analysis'):
        fig, ax = plt.subplots()
        ax.plot(nodes, auc_scores, 'o')
        ax.set_xlabel('No of Nodes', ha='right', x=1, size=10)
        ax.set_ylabel('AUC score',ha='right', y=1, size=10)
        ax.grid(True, 'major', linestyle='solid', color='grey', alpha=0.5)
        plt.savefig(name, dpi = 200)
        print("Plotting NN_nodes_analysis")
        plt.close()

    def activation_analysis_plot(self, auc_scores, activ_list, name = 'plotting/NN_plots/NN_activation_analysis'):
        fig, ax = plt.subplots()
        ax.plot(activ_list, auc_scores, 'o')
        ax.set_xlabel('Intermediate Activation Types', ha='right', x=1, size=10)
        ax.set_ylabel('AUC score',ha='right', y=1, size=10)
        ax.grid(True, 'major', linestyle='solid', color='grey', alpha=0.5)
        plt.savefig(name, dpi = 200)
        print("Plotting NN_activation_analysis")
        plt.close()
    
    def layers_analysis_plot(self, auc_scores, num_int_layers, name = 'plotting/NN_plots/NN_layers_analysis'):
        fig, ax = plt.subplots()
        ax.plot(num_int_layers, auc_scores, 'o')
        ax.set_xlabel('Number of intermediate layers', ha='right', x=1, size=10)
        ax.set_ylabel('AUC score',ha='right', y=1, size=10)
        ax.grid(True, 'major', linestyle='solid', color='grey', alpha=0.5)
        plt.savefig(name, dpi = 200)
        print("Plotting NN_layers_analysis")
        plt.close()
        

        




















































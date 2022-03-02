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
num_epochs = 1
batch_size = 64
val_split = 0.15
test_split = 0.15
learning_rate = 0.0001

#STXS mapping
map_def = [['ggH',10,11],['qqH',20,21,22,23],['WH',30,31],['ZH',40,41],['ttH',60,61],['tH',80,81]]
color = ['#54aaf8', '#f08633', '#8bfa71', '#166e02','#ea3cf7', '#fef050']
epochs = np.linspace(1,num_epochs,num_epochs,endpoint=True).astype(int) #For plotting
binNames = ['ggH','qqH','WH','ZH','ttH','tH'] 
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
dataframes.append(pd.read_csv('2017/MC/DataFrames/tHq_VBF_BDT_df_2017.csv', nrows = 254039)) # putting a quarter each - to avoid "killing"
dataframes.append(pd.read_csv('2017/MC/DataFrames/tHW_VBF_BDT_df_2017.csv', nrows = 130900))
df = pd.concat(dataframes, sort=False, axis=0 )

#dataframe of train_vars
data = df[train_vars]

#Preselection cuts
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

data['proc_new'] = mapping(map_list=map_def,stage=data['HTXS_stage_0'])

#Define the procs as the labels
#ggh: 0, VBF:1, VH: 2, ttH: 3
num_categories = data['proc_new'].nunique()

proc_new = np.array(data['proc_new'])
#Assign the numbers in the same order as the binNames above
y_train_labels_num = []
for i in proc_new:
    if i == 'ggH':
        y_train_labels_num.append(0)
    if i == 'qqH':
        y_train_labels_num.append(1)
    if i == 'WH':
        y_train_labels_num.append(2)
    if i == 'ZH':
        y_train_labels_num.append(3)
    if i == 'ttH':
        y_train_labels_num.append(4)
    if i == 'tH':
        y_train_labels_num.append(5)

data['proc_num'] = y_train_labels_num

#Shuffle dataframe
data = data.sample(frac=1)

y_train_labels = np.array(data['proc_new'])
y_train_labels_num = np.array(data['proc_num'])
y_train_labels_hot = np_utils.to_categorical(y_train_labels_num, num_classes=num_categories)
weights = np.array(data['weight'])

#Remove proc after shuffle
data = data.drop(columns=['proc','weight','proc_num','HTXS_stage_0','HTXS_stage1_2_cat_pTjet30GeV','proc_new'])

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
model=Sequential([Dense(units=100,input_shape=(num_inputs,),activation='relu'),
                Dense(units=100,activation='relu'),
                #Dense(units=100,activation='relu'),
                Dense(units=num_categories,activation='softmax')]) #For multiclass NN use softmax

#Compile the model
model.compile(optimizer=Adam(lr=learning_rate),loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()

#Equalizing weights
train_w_df = pd.DataFrame()
train_w = 300 * train_w # to make loss function O(1)
train_w_df['weight'] = train_w
train_w_df['proc'] = proc_arr_train
qqh_sum_w = train_w_df[train_w_df['proc'] == 'qqH']['weight'].sum()
ggh_sum_w = train_w_df[train_w_df['proc'] == 'ggH']['weight'].sum()
wh_sum_w = train_w_df[train_w_df['proc'] == 'WH']['weight'].sum()
zh_sum_w = train_w_df[train_w_df['proc'] == 'ZH']['weight'].sum()
tth_sum_w = train_w_df[train_w_df['proc'] == 'ttH']['weight'].sum()
th_sum_w = train_w_df[train_w_df['proc'] == 'tH']['weight'].sum()
train_w_df.loc[train_w_df.proc == 'qqH','weight'] = (train_w_df[train_w_df['proc'] == 'qqH']['weight'] * ggh_sum_w / qqh_sum_w)
train_w_df.loc[train_w_df.proc == 'WH','weight'] = (train_w_df[train_w_df['proc'] == 'WH']['weight'] * ggh_sum_w / wh_sum_w)
train_w_df.loc[train_w_df.proc == 'ZH','weight'] = (train_w_df[train_w_df['proc'] == 'ZH']['weight'] * ggh_sum_w / zh_sum_w)
train_w_df.loc[train_w_df.proc == 'ttH','weight'] = (train_w_df[train_w_df['proc'] == 'ttH']['weight'] * ggh_sum_w / tth_sum_w)
train_w_df.loc[train_w_df.proc == 'tH','weight'] = (train_w_df[train_w_df['proc'] == 'tH']['weight'] * ggh_sum_w / th_sum_w)
train_w = np.array(train_w_df['weight'])

#Training the model
history = model.fit(x=x_train,y=y_train,batch_size=batch_size,epochs=num_epochs,sample_weight=train_w,shuffle=True,verbose=2)

# Output Score
y_pred_test = model.predict_proba(x=x_test)
x_test['proc'] = proc_arr_test
x_test['weight'] = test_w
x_test['output_score_ggh'] = y_pred_test[:,0]
x_test['output_score_qqh'] = y_pred_test[:,1]
x_test['output_score_wh'] = y_pred_test[:,2]
x_test['output_score_zh'] = y_pred_test[:,3]
x_test['output_score_tth'] = y_pred_test[:,4]
x_test['output_score_th'] = y_pred_test[:,5]

output_score_ggh = np.array(y_pred_test[:,0])
output_score_qqh = np.array(y_pred_test[:,1])
output_score_wh = np.array(y_pred_test[:,2])
output_score_zh = np.array(y_pred_test[:,3])
output_score_tth = np.array(y_pred_test[:,4])
output_score_th = np.array(y_pred_test[:,5])

x_test_ggh = x_test[x_test['proc'] == 'ggH']
x_test_qqh = x_test[x_test['proc'] == 'qqH']
x_test_wh = x_test[x_test['proc'] == 'WH']
x_test_zh = x_test[x_test['proc'] == 'ZH']
x_test_tth = x_test[x_test['proc'] == 'ttH']
x_test_th = x_test[x_test['proc'] == 'tH']

ggh_w = x_test_ggh['weight'] / x_test_ggh['weight'].sum()
qqh_w = x_test_qqh['weight'] / x_test_qqh['weight'].sum()
wh_w = x_test_wh['weight'] / x_test_wh['weight'].sum()
zh_w = x_test_zh['weight'] / x_test_zh['weight'].sum()
tth_w = x_test_tth['weight'] / x_test_tth['weight'].sum()
th_w = x_test_th['weight'] / x_test_th['weight'].sum()
total_w = x_test['weight'] / x_test['weight'].sum()

#Accuracy Score
y_pred = y_pred_test.argmax(axis=1)
y_true = y_test.argmax(axis=1)
print 'Accuracy score: '
NNaccuracy = accuracy_score(y_true, y_pred)
print(NNaccuracy)

#Confusion Matrix
cm = confusion_matrix(y_true=y_true,y_pred=y_pred, sample_weight = test_w)

def roc_comp_step1(ypred,output):
    y_pred_prob = []
    for i in range(len(ypred)):
        if ypred[i] == 0:
            y_pred_prob.append(0)
        elif ypred[i] == 1:
            y_pred_prob.append(output[i])
    return y_pred_prob

def y_pred_prob(ytrue=y_true,ypred=y_pred,labeldef=label_def,proc='ggH',output=output_score_ggh):
    for i in range(len(labeldef)):
        if proc in labeldef[i]:
            index=i
    y_true_proc = np.where(y_true == index, 1, 0)
    y_pred_proc = np.where(y_pred == index, 1, 0)
    y_pred_proc_prob = roc_comp_step1(ypred=y_pred_proc,output=output)
    return y_true_proc, y_pred_proc_prob

y_true_ggh, y_pred_ggh_prob = y_pred_prob(proc='ggH',output=output_score_ggh)



def plot_output_score(data='output_score_qqh', density=False,):
    #Can then change it to plotting proc
    print('Plotting',data)
    output_score_ggh = np.array(x_test_ggh[data])
    output_score_qqh = np.array(x_test_qqh[data])
    output_score_wh = np.array(x_test_wh[data])
    output_score_zh = np.array(x_test_zh[data])
    output_score_tth = np.array(x_test_tth[data])
    output_score_th = np.array(x_test_th[data])

    fig, ax = plt.subplots()
    ax.hist(output_score_ggh, bins=50, label='ggH', histtype='step',weights=ggh_w, color = color[0])#,density=True) 
    ax.hist(output_score_qqh, bins=50, label='qqH', histtype='step',weights=qqh_w, color = color[1]) #density=True)
    ax.hist(output_score_wh, bins=50, label='WH', histtype='step',weights=wh_w, color = color[2]) #density=True) 
    ax.hist(output_score_zh, bins=50, label='ZH', histtype='step',weights=zh_w, color = color[3]) #density=True) 
    ax.hist(output_score_tth, bins=50, label='ttH', histtype='step',weights=tth_w, color = color[4]) #density=True)
    ax.hist(output_score_th, bins=50, label='tH', histtype='step',weights=th_w, color = color[5]) #density=True)
    plt.legend()
    plt.title('Output Score')
    plt.ylabel('Fraction of Events')
    plt.xlabel('NN Score')
    name = 'plotting/NN_plots/NN_Sixclass_'+data
    plt.savefig(name, dpi = 1200)

#Plotting:
#Plot accuracy
def plot_accuracy():
    #val_accuracy = history.history['val_acc']
    accuracy = history.history['acc']
    fig, ax = plt.subplots(1)
    #plt.plot(epochs,val_accuracy,label='Validation')
    plt.plot(epochs,accuracy,label='Train')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.xticks(epochs_int)
    plt.legend()
    name = 'plotting/NN_plots/NN_Accuracy'
    fig.savefig(name)

#Plot loss
def plot_loss():
    #val_loss = history.history['val_loss']
    loss = history.history['loss']
    fig, ax = plt.subplots(1)
    #plt.plot(epochs,val_loss,label='Validation')
    plt.plot(epochs,loss,label='Train')
    plt.title('Loss function')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xticks(epochs)
    plt.legend()
    name = 'plotting/NN_plots/NN_Loss'
    fig.savefig(name)



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
    name = 'plotting/NN_plots/NN_Sixclass_Confusion_Matrix'
    fig.savefig(name, dpi = 1200)

def plot_performance_plot(cm=cm,labels=binNames, normalize = True, color = color):
    #cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
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
    #color = ['#24b1c9','#e36b1e','#1eb037','#c21bcf','#dbb104']
    for i in range(len(cm)):
        #ax.bar(labels, cm[i,:],label=labels[i],bottom=bottom)
        #bottom += np.array(cm[i,:])
        ax.bar(labels, cm[i,:],label=labels[i],bottom=bottom,color=color[i])
        bottom += np.array(cm[i,:])
    plt.legend()
    current_bottom, current_top = ax.get_ylim()  
    ax.set_ylim(bottom=0, top=current_top*1.3)
    #plt.title('Performance Plot')
    plt.ylabel('Fraction of events')
    ax.set_xlabel('Events', ha='right',x=1,size=9) #, x=1, size=13)
    name = 'plotting/NN_plots/NN_Sixclass_Performance_Plot'
    plt.savefig(name, dpi = 1200)
    plt.show()

def plot_roc_curve(binNames = binNames, y_test = y_test, y_pred_test = y_pred_test, x_test = x_test, color = color):
    # sample weights
    # find weighted average 
    fig, ax = plt.subplots()
    #y_pred_test  = clf.predict_proba(x_test)
    for k in range(len(binNames)):
        signal = binNames[k]
        for i in range(num_categories):
            if binNames[i] == signal:
                #sig_y_test  = np.where(y_test==i, 1, 0)
                sig_y_test = y_test[:,i]
                print('sig_y_test', sig_y_test)
                y_pred_test_array = y_pred_test[:,i]
                print('y_pred_test_array', y_pred_test_array)
                print('Here')
                #test_w = test_w.reshape(1, -1)
                print('test_w', test_w)
                #auc = roc_auc_score(sig_y_test, y_pred_test_array, sample_weight = test_w)
                fpr_keras, tpr_keras, thresholds_keras = roc_curve(sig_y_test, y_pred_test_array, sample_weight = test_w)
                #print('auc: ', auc)
                print('Here')
                fpr_keras.sort()
                tpr_keras.sort()
                auc_test = auc(fpr_keras, tpr_keras)
                ax.plot(fpr_keras, tpr_keras, label = 'AUC = {0}, {1}'.format(round(auc_test, 3), binNames[i]), color = color[i])
    ax.legend(loc = 'lower right', fontsize = 'x-small')
    ax.set_xlabel('Background Efficiency', ha='right', x=1, size=9)
    ax.set_ylabel('Signal Efficiency',ha='right', y=1, size=9)
    ax.grid(True, 'major', linestyle='dotted', color='grey', alpha=0.5)
    name = 'plotting/NN_plots/NN_Sixclass_ROC_curve'
    plt.savefig(name, dpi = 1200)
    print("Plotting ROC Curve")
    plt.close()

plot_confusion_matrix(cm,binNames,normalize=True)

plot_performance_plot()

plot_roc_curve()

"""
plot_output_score(data='output_score_qqh')
plot_output_score(data='output_score_ggh')
plot_output_score(data='output_score_wh')
plot_output_score(data='output_score_zh')
plot_output_score(data='output_score_tth')
plot_output_score(data='output_score_th')
"""

#plot_accuracy()
#plot_loss()


#save as a pickle file
#trainTotal.to_pickle('%s/nClassNNTotal.pkl'%frameDir)
#print 'frame saved as %s/nClassNNTotal.pkl'%frameDir
#Read in pickle file
#trainTotal = pd.read_pickle(opts.dataFrame)
#print 'Successfully loaded the dataframe'
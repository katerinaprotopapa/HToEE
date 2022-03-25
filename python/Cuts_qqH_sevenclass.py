from __future__ import division
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

color  = ['silver','indianred','salmon','lightgreen','seagreen','mediumturquoise','darkslategrey','skyblue','steelblue','lightsteelblue','mediumslateblue']

binNames = ['qqH_Rest',
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
     #'leadElectronEn', 'leadElectronMass', 'leadElectronPt', 'leadElectronEta', 'leadElectronPhi', 'leadElectronCharge',
     #'leadMuonEn', 'leadMuonMass', 'leadMuonPt', 'leadMuonEta', 'leadMuonPhi', 'leadMuonCharge',
     #'subleadElectronEn', 'subleadElectronMass', 'subleadElectronPt', 'subleadElectronEta', 'subleadElectronPhi', 'subleadElectronCharge', 
     #'subleadMuonEn', 'subleadMuonMass', 'subleadMuonPt', 'subleadMuonEta', 'subleadMuonPhi', 'subleadMuonCharge'
     ]

train_vars.append('proc')
train_vars.append('weight')
train_vars.append('HTXS_stage_0')
train_vars.append('HTXS_stage1_2_cat_pTjet30GeV')

def error_fn(num_correct, num_all, sigma_correct, sigma_all):
    error = (((1/num_all)**2) * (sigma_correct**2) + ((-num_correct / (num_all**2))**2) * (sigma_correct**2))**0.5
    return error

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
proc_original = np.array(data['proc_original'])
#Assign the numbers in the same order as the binNames above
y_train_labels_num = []
for i in proc_original:
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
# turns out that it was just getting killed by the huge number of data

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
y_train_labels_num_pred = []
for i in range(data.shape[0]):
    #print('eeee')
    if njets[i] == 0 or njets[i] == 1:
        proc_value = 'qqH_Rest'
        proc_value_num = 0
    else:
        if dijetmass[i] < 350:
            if dijetmass[i] > 60 and dijetmass[i] < 120:
                proc_value = 'QQ2HQQ_GE2J_MJJ_60_120'
                proc_value_num = 1
            else:
                proc_value = 'qqH_Rest'
                proc_value_num = 0
        else:
            if diphotonpt[i] < 200:
                if  dijetmass[i] < 700 and diphotonjetspt[i] < 25:
                    proc_value = 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25'
                    proc_value_num = 2
                elif dijetmass[i] < 700 and diphotonjetspt[i] >= 25:
                    proc_value = 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25'
                    proc_value_num = 3
                elif dijetmass[i] >= 700 and diphotonjetspt[i] < 25:
                    proc_value = 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25'
                    proc_value_num = 4
                else:
                    proc_value = 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_GT25'
                    proc_value_num = 5
            else: 
                proc_value = 'QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200'
                proc_value_num = 6
    proc.append(proc_value)
    y_train_labels_num_pred.append(proc_value_num)
y_train_labels_num_pred = np.array(y_train_labels_num_pred)
data['proc_new'] = proc

# Confusion Matrix

y_true_old = data['proc_original']
y_pred_old = data['proc_new']
y_true = y_train_labels_num
y_pred = y_train_labels_num_pred

#cm_old = confusion_matrix(y_true=y_true,y_pred=y_pred)
cm = confusion_matrix(y_true=y_true,y_pred=y_pred,sample_weight=weights)
cm_old = cm
cm_old_no_weights = confusion_matrix(y_true=y_true,y_pred=y_pred)
#cm_new = np.zeros((len(binNames),len(binNames)),dtype=int)
#for i in range(len(y_true)):
#    cm_new[y_true[i]][y_pred[i]] += 1

test_w = weights
cm_new = np.zeros((len(labelNames),len(labelNames)),dtype=float)
cm_weights_new_squared = np.zeros((len(labelNames),len(labelNames)),dtype=float)
cm_weights_new = np.zeros((len(labelNames),len(labelNames)),dtype=float)
for i in range(len(y_true)):
    cm_new[y_true[i]][y_pred[i]] += 1
    cm_weights_new_squared[y_true[i]][y_pred[i]] += test_w[i]**2
    cm_weights_new[y_true[i]][y_pred[i]] += test_w[i]

#Accuracy Score
"""
print 'Accuracy score - function: '
NNaccuracy = accuracy_score(y_true, y_pred, sample_weight = test_w)
print(NNaccuracy)
"""

num_correct = 0
num_all = 0
sigma_correct = 0
sigma_all = 0
yield_all = 0
yield_correct = 0
for i in range(cm_new.shape[0]):
    for j in range(cm_new.shape[1]):
        num_all += cm_new[i][j]
        sigma_all += cm_weights_new_squared[i][j]
        yield_all += cm_weights_new[i][j]
        if i == j:     # so diagonal
            num_correct += cm_new[i][j]
            sigma_correct += cm_weights_new_squared[i][j] 
            yield_correct += cm_weights_new[i][j]
sigma_all = sigma_all**0.5
sigma_correct = sigma_correct**0.5
accuracy = num_correct / num_all

acc_score_error = error_fn(yield_correct, yield_all, sigma_correct, sigma_all)
print('Final Accuracy Score: ', accuracy)
print('with error: ', acc_score_error)


s_in = []
s_in_w = []
s_in_w_squared = []
s_tot = []
s_tot_w = []
s_tot_w_squared = []
e_s = []
signal_error_list = []
b_in = []
b_in_w = []
b_in_w_squared = []
b_tot = []
b_tot_w = []
b_tot_w_squared = []
e_b = []
bckg_error_list = []

for i in range(len(labelNames)):
    s_in.append(cm_new[i][i])
    s_in_w.append(cm_weights_new[i][i])
    s_in_w_squared.append(cm_weights_new_squared[i][i])
    s_tot.append(np.sum(cm_new[i,:]))
    s_tot_w.append(np.sum(cm_weights_new[i,:]))
    s_tot_w_squared.append(np.sum(cm_weights_new_squared[i,:]))
    e_s.append(s_in[i]/s_tot[i])

    b_in.append(np.sum(cm_new[:,i]) - s_in[i])
    b_in_w.append(np.sum(cm_weights_new[:,i]) - s_in_w[i])
    b_in_w_squared.append(np.sum(cm_weights_new_squared[:,i]) - s_in_w_squared[i])
    b_tot.append(np.sum(cm_new) - s_tot[i])
    b_tot_w.append(np.sum(cm_weights_new) - s_tot_w[i])
    b_tot_w_squared.append(np.sum(cm_weights_new_squared) - s_tot_w_squared[i])
    e_b.append(b_in[i]/b_tot[i])

    signal_error = error_fn(s_in_w[i], s_tot_w[i], np.sqrt(s_in_w_squared[i]), np.sqrt(s_tot_w_squared[i]))
    #signal_error = 1 / np.sqrt(np.sum(cm_new[i,:]))
    print('Final Signal Efficiency: ', e_s[i])
    print('with error: ', signal_error)
    signal_error_list.append(signal_error)

    bckg_error = error_fn(b_in_w[i], b_tot_w[i], np.sqrt(b_in_w_squared[i]), np.sqrt(b_tot_w_squared[i]))
    #bckg_error = 1 / np.sqrt(np.sum(cm_new[i,:]))
    print('Final Signal Efficiency: ', e_b[i])
    print('with error: ', bckg_error)
    bckg_error_list.append(bckg_error)

num_correct_2 = 0
num_all_2 = 0
for i in range(1, cm.shape[0]):
    for j in range(cm.shape[1]):
        num_all_2 += cm[i][j]
        if i == j:     # so diagonal
            num_correct_2 += cm[i][j]
accuracy_2 = num_correct_2 / num_all_2
print('Cuts Final Accuracy Score without qqH rest: ', accuracy_2)


#Confusion Matrix
def plot_confusion_matrix(cm,classes,labels = labelNames, normalize=True,title='Confusion matrix',cmap=plt.cm.Blues):
    fig, ax = plt.subplots(figsize = (10,10))
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,labels,rotation=45, horizontalalignment = 'right')
    plt.yticks(tick_marks,labels)
    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        for i in range(len(cm[0])):
            for j in range(len(cm[1])):
                cm[i][j] = float("{:.2f}".format(cm[i][j]))
    thresh = cm.max()/2.
    print(cm)
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    #plt.title(title)
    for i, j in product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,cm[i,j],horizontalalignment='center',color='white' if cm[i,j]>thresh else 'black')
    plt.tight_layout()
    plt.colorbar()
    plt.ylabel('True Label', size = 12)
    plt.xlabel('Predicted label', size = 12)
    name = 'plotting/Cuts/Cuts_qqH_Sevenclass_Confusion_Matrix'
    plt.tight_layout()
    fig.savefig(name, dpi = 1200)

def plot_performance_plot(cm=cm,labels=labelNames, normalize = True, color = color):
    #cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
    cm = cm.astype('float')/cm.sum(axis=0)[np.newaxis,:]
    for i in range(len(cm[0])):
        for j in range(len(cm[1])):
            cm[i][j] = float("{:.3f}".format(cm[i][j]))
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize = (10,10))
    plt.rcParams.update({
    'font.size': 9})
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks,labels,rotation=45, horizontalalignment = 'right')
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
    plt.ylabel('Fraction of events', size = 12)
    ax.set_xlabel('Events',size=12) #, x=1, size=13)
    name = 'plotting/Cuts/Cuts_qqH_Sevenclass_Performance_Plot'
    plt.tight_layout()
    plt.savefig(name, dpi = 1200)
    plt.show()

#plot_confusion_matrix(cm,binNames,normalize=True)
#plot_performance_plot()

#print('Cuts_qqH_sevenclass: ', NNaccuracy)
#print('Cuts Final Accuracy Score with qqH rest: ', accuracy)
#print('Cuts Final Accuracy Score without qqH rest: ', accuracy_2)

#exit(0)

# ------------------------ 
# Binary BDT for signal purity
# okayy lessgooo

# data_new['proc']  # are the true labels
# data_new['weight'] are the weights
s_in_2 = []
s_in_w_2 = []
s_in_w_squared_2 = []
s_tot_2 = []
s_tot_w_2 = []
s_tot_w_squared_2 = []
e_s_2 = []
signal_error_list_2 = []
b_in_2 = []
b_in_w_2 = []
b_in_w_squared_2 = []
b_tot_2 = []
b_tot_w_2 = []
b_tot_w_squared_2 = []
e_b_2 = []
bckg_error_list_2 = []

error_final_array = []


num_estimators = 200
test_split = 0.30

signal = binNames
#signal = ['qqH_Rest','QQ2HQQ_GE2J_MJJ_60_120'] # for debugging
#conf_matrix = np.zeros((2,1)) # for the final confusion matrix
conf_matrix_w = np.zeros((2,len(signal)))
conf_matrix_no_w = np.zeros((2,len(signal)))

conf_matrix_w2 = np.zeros((1,len(signal)))
conf_matrix_no_w2 = np.zeros((1,len(signal)))

fig, ax = plt.subplots()
for i in range(len(signal)):
    clf_2 = xgb.XGBClassifier(objective='binary:logistic', n_estimators=num_estimators, 
                            eta=0.1, maxDepth=6, min_child_weight=0.01, 
                            subsample=0.6, colsample_bytree=0.6, gamma=4)
    data_new = data.copy()  
    # now i want to get the predicted labels
    proc_pred = []      
    for j in range(len(y_pred)):
        if(y_pred[j] == i): # so that the predicted label is the signal
            proc_pred.append(signal[i])
        else:
            proc_pred.append('background')
    data_new['proc_pred'] = proc_pred    

    #exit(0)
    rest, data_new = train_test_split(data_new, test_size = test_split, shuffle = True)

    # now cut down the dataframe to the predicted ones -  this is the split for the different dataframes
    data_new = data_new[data_new.proc_pred == signal[i]] 

    # now from proc make signal against background (binary classifier)

    proc_true = np.array(data_new['proc_original'])
    y_train_labels_num = []
    y_train_labels = []
    for j in range(len(proc_true)):
        if proc_true[j] == signal[i]:
            y_train_labels.append(signal[i])
            y_train_labels_num.append(1)
        else: 
            y_train_labels.append('background')
            y_train_labels_num.append(0)
    y_train_labels = np.array(y_train_labels)
    y_train_labels_num = np.array(y_train_labels_num)
    
    weights_new = np.array(data_new['weight'])

    
    data_new = data_new.drop(columns=['weight'])
    data_new = data_new.drop(columns=['proc_original'])
    data_new = data_new.drop(columns=['proc_pred'])
    data_new = data_new.drop(columns=['proc_new'])

    # the new split
    x_train_2, x_test_2, y_train_2, y_test_2, train_w_2, test_w_2, proc_arr_train_2, proc_arr_test_2 = train_test_split(data_new, y_train_labels_num, weights_new, y_train_labels, test_size = test_split, shuffle = True)

    train_w_df = pd.DataFrame()
    train_w = 300 * train_w_2 # to make loss function O(1)
    train_w_df['weight'] = train_w
    train_w_df['proc'] = proc_arr_train_2
    signal_sum_w = train_w_df[train_w_df['proc'] == signal[i]]['weight'].sum()
    background_sum_w = train_w_df[train_w_df['proc'] == 'background']['weight'].sum()

    train_w_df.loc[train_w_df.proc == 'background','weight'] = (train_w_df[train_w_df['proc'] == 'background']['weight'] * signal_sum_w / background_sum_w)
    train_w_new = np.array(train_w_df['weight'])

    print (' Training classifier with Signal = ', signal[i])
    clf_2 = clf_2.fit(x_train_2, y_train_2, sample_weight=train_w_new)
    print (' Finished classifier with Signal = ', signal[i])

    y_pred_test_2 = clf_2.predict_proba(x_test_2) 
    y_pred_2 = y_pred_test_2.argmax(axis=1)

    cm_2 = confusion_matrix(y_true = y_test_2, y_pred = y_pred_2, sample_weight = test_w_2)  #weights result in decimal values <1 so not sure if right
    cm_2_no_weights = confusion_matrix(y_true = y_test_2, y_pred = y_pred_2)

    #print('cm_2:')
    #print(cm_2)

    # grabbing predicted label column
    #norm = cm_2[0][1] + cm_2[1][1]
    #conf_matrix[0][i] = (cm_2[0][1])/norm
    #conf_matrix[1][i] = (cm_2[1][1])/norm

    conf_matrix_w[0][i] = cm_2[0][1]
    conf_matrix_w[1][i] = cm_2[1][1]
    conf_matrix_no_w[0][i] = cm_2_no_weights[0][1]
    conf_matrix_no_w[1][i] = cm_2_no_weights[1][1]

    conf_matrix_w2[0][i] = (cm_2[0][0] + cm_2[1][0]) / np.sum(np.array(cm_2))
    conf_matrix_no_w2[0][i] = (cm_2_no_weights[0][0] + cm_2_no_weights[1][0])/ np.sum(np.array(cm_2_no_weights))


    # ERROR STUFF -----------
    # dear philipp, here is your christmas present:
    threshold = 0.4   # bckg efficiency threshold (manually set)

    # get output score
    x_test_2['proc'] = proc_arr_test_2
    x_test_2['weight'] = test_w_2

    x_test_2['output_score_background'] = y_pred_test_2[:,0]
    x_test_2[signal[i]] = y_pred_test_2[:,1]

    x_test_qqh1 = x_test_2[x_test_2['proc'] == signal[i]]
    x_test_qqh2 = x_test_2[x_test_2['proc'] == 'background']

    qqh1_w = x_test_qqh1['weight'] / x_test_qqh1['weight'].sum()
    qqh2_w = x_test_qqh2['weight'] / x_test_qqh2['weight'].sum()

    output_score_qqh2 = np.array(x_test_qqh2[signal[i]])
    counts, bins, _ = plt.hist(output_score_qqh2, bins=100, label='Background', histtype='step',weights=qqh2_w, density = True)
    #plt.savefig('plotting/TESTING', dpi = 1200)

    for j in range(len(bins)):
        bins_2 = bins[:j+1]
        counts_2 = counts[:j]
        area = sum(np.diff(bins_2)*counts_2)
        if area <= (1-threshold):
            bdt_score = bins_2[j]
    
    print('bdt_score: ', bdt_score)

    thresh = bdt_score

    y_pred_errors = []
    for k in range(len(y_test_2)):
        if y_pred_test_2[:,1][k]>thresh:
            y_pred_errors.append(1)
        else:
            y_pred_errors.append(0)
    y_pred_errors = np.array(y_pred_errors)
    cm_errors = np.zeros((2,2),dtype=int)
    cm_errors_weights = np.zeros((2,2),dtype=float)
    cm_errors_weights_squared = np.zeros((2,2),dtype=float)
    for l in range(len(y_test_2)):
        cm_errors[y_test_2[l]][y_pred_errors[l]] += 1
        cm_errors_weights[y_test_2[l]][y_pred_errors[l]] += test_w_2[l]
        cm_errors_weights_squared[y_test_2[l]][y_pred_errors[l]] += test_w_2[l]**2
    print(cm_errors)
    print(cm_errors_weights)
    print(cm_errors_weights_squared)

    s_in_2.append(cm_errors[1][1])
    s_in_w_2.append(cm_errors_weights[1][1])
    s_in_w_squared_2.append(cm_errors_weights_squared[1][1])
    s_tot_2.append(np.sum(cm_errors[1,:]))
    s_tot_w_2.append(np.sum(cm_errors_weights[1,:]))
    s_tot_w_squared_2.append(np.sum(cm_errors_weights_squared[1,:]))
    e_s_2.append(s_in_2[i]/s_tot_2[i])

    b_in_2.append(cm_errors[0][1])
    b_in_w_2.append(cm_errors_weights[0][1])
    b_in_w_squared_2.append(cm_errors_weights_squared[0][1])
    b_tot_2.append(np.sum(cm_errors[0,:]))
    b_tot_w_2.append(np.sum(cm_errors_weights[0,:]))
    b_tot_w_squared_2.append(np.sum(cm_errors_weights_squared[0,:]))
    e_b_2.append(b_in_2[i]/b_tot_2[i])

    signal_error = error_fn(s_in_w_2[i], s_tot_w_2[i], np.sqrt(s_in_w_squared_2[i]), np.sqrt(s_tot_w_squared_2[i]))
    #signal_error = 1 / np.sqrt(np.sum(cm_new[i,:]))
    print('Final Signal Efficiency: ', e_s_2[i])
    print('with error: ', signal_error)
    signal_error_list_2.append(signal_error)

    bckg_error = error_fn(b_in_w_2[i], b_tot_w_2[i], np.sqrt(b_in_w_squared_2[i]), np.sqrt(b_tot_w_squared_2[i]))
    #bckg_error = 1 / np.sqrt(np.sum(cm_new[i,:]))
    print('Final Signal Efficiency: ', e_b_2[i])
    print('with error: ', bckg_error)
    bckg_error_list_2.append(bckg_error)

    # adding the errors
    error_final_array.append(np.sqrt(signal_error_list_2[i]**2 + bckg_error_list_2[i]**2 + signal_error_list[i]**2 + bckg_error_list[i]**2))
    print('Error final: ', error_final_array[i])
    #exit(0)

    # --------------

    # ROC Curve
    sig_y_test  = np.where(y_test_2==1, 1, 0)
    #sig_y_test  = y_test_2
    y_pred_test_array = y_pred_test_2[:,1] # to grab the signal
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(sig_y_test, y_pred_test_array, sample_weight = test_w_2)
    fpr_keras.sort()
    tpr_keras.sort()
    name_fpr = 'csv_files/Cuts_binary_fpr_' + signal[i]
    name_tpr = 'csv_files/Cuts_binary_tpr_' + signal[i]
    np.savetxt(name_fpr, fpr_keras, delimiter = ',')
    np.savetxt(name_tpr, tpr_keras, delimiter = ',')
    auc_test = auc(fpr_keras, tpr_keras)
    ax.plot(fpr_keras, tpr_keras, label = 'AUC = {0}, {1}'.format(round(auc_test, 3), labelNames[i]), color = color[i])

ax.legend(loc = 'lower right', fontsize = 'small')
ax.set_xlabel('Background Efficiency', ha='right', x=1, size=9)
ax.set_ylabel('Signal Efficiency',ha='right', y=1, size=9)
ax.grid(True, 'major', linestyle='dotted', color='grey', alpha=0.5)
plt.tight_layout()
name = 'plotting/Cuts/Cuts_qqH_binary_Multi_ROC_curve'
plt.savefig(name, dpi = 1200)
print("Plotting ROC Curve")
plt.close()

print('Final conf_matrix:')
print(conf_matrix_w)

#Exporting final confusion matrix
name_cm = 'csv_files/Cuts_binary_cm'
np.savetxt(name_cm, cm_old, delimiter = ',')
name_cm_no_w = 'csv_files/Cuts_binary_cm_no_w'
np.savetxt(name_cm_no_w, cm_old_no_weights, delimiter = ',')

#Need a new function beause the cm structure is different
def plot_performance_plot_final(cm=conf_matrix_w,cm_old = cm_old,labels=labelNames, color = color, name = 'plotting/Cuts/Cuts_qqH_Sevenclass_Performance_Plot_final'):
    cm = cm.astype('float')/cm.sum(axis=0)[np.newaxis,:]
    cm_old = cm_old.astype('float')/cm_old.sum(axis=0)[np.newaxis,:]
    sig_old = []
    for k in range(cm_old.shape[0]):
        sig_old.append(cm_old[k][k])
    for i in range(len(cm[0])):
        for j in range(len(cm[:,1])):
            cm[j][i] = float("{:.3f}".format(cm[j][i]))
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize = (10,10))
    plt.rcParams.update({
    'font.size': 9})
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks,labels,rotation=45, horizontalalignment = 'right')
    bottom = np.zeros(len(labels))
    ax.bar(labels, cm[1,0],label='Signal',bottom=bottom,color=color[1])
    bottom += np.array(cm[1,:])
    ax.bar(labels, cm[0,:],label='Background',bottom=bottom,color=color[0])
    ax.bar(labels, sig_old, label = 'Signal before binary BDT',fill = False, ecolor = 'black')
    plt.legend()
    current_bottom, current_top = ax.get_ylim()
    ax.set_ylim(bottom=0, top=current_top*1.3)
    plt.ylabel('Fraction of events', size = 12)
    ax.set_xlabel('Events',size=12)
    plt.tight_layout()
    plt.savefig(name, dpi = 1200)
    plt.show()
# now to make our final plot of performance
#plot_performance_plot_final(cm = conf_matrix_w,labels = labelNames, name = 'plotting/Cuts/Cuts_qqH_Sevenclass_Performance_Plot_final')

y_true = y_test_2
y_pred = y_pred_2
cm_new = np.zeros((len(labelNames),len(labelNames)),dtype=float)
cm_weights_new_squared = np.zeros((len(labelNames),len(labelNames)),dtype=float)
cm_weights_new = np.zeros((len(labelNames),len(labelNames)),dtype=float)
for i in range(len(y_true)):
    cm_new[y_true[i]][y_pred[i]] += 1
    cm_weights_new_squared[y_true[i]][y_pred[i]] += test_w_2[i]**2
    cm_weights_new[y_true[i]][y_pred[i]] += test_w_2[i]

#Accuracy Score
"""
print 'Accuracy score - function: '
NNaccuracy = accuracy_score(y_true, y_pred, sample_weight = test_w)
print(NNaccuracy)
"""

num_correct = 0
num_all = 0
sigma_correct = 0
sigma_all = 0
yield_all = 0
yield_correct = 0
for i in range(cm_new.shape[0]):
    for j in range(cm_new.shape[1]):
        num_all += cm_new[i][j]
        sigma_all += cm_weights_new_squared[i][j]
        yield_all += cm_weights_new[i][j]
        if i == j:     # so diagonal
            num_correct += cm_new[i][j]
            sigma_correct += cm_weights_new_squared[i][j] 
            yield_correct += cm_weights_new[i][j]
sigma_all = sigma_all**0.5
sigma_correct = sigma_correct**0.5
accuracy = num_correct / num_all

acc_score_error = error_fn(yield_correct, yield_all, sigma_correct, sigma_all)
print('Final Accuracy Score after specialized BDT: ', accuracy)
print('with error: ', acc_score_error)

num_false = np.sum(conf_matrix_w[0,:])
num_correct = np.sum(conf_matrix_w[1,:])
accuracy = num_correct / (num_correct + num_false)
print('Cuts Final Accuracy Score with qqH:')
print(accuracy)

num_false = np.sum(conf_matrix_w[0,1:])
num_correct = np.sum(conf_matrix_w[1,1:])
accuracy = num_correct / (num_correct + num_false)
print('Cuts Final Accuracy Score without qqH:')
print(accuracy)




















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

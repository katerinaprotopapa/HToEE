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

#Define key quantities, use to tune BDT
num_estimators = 300
test_split = 0.15
learning_rate = 0.001

binNames = ['ggH','qqH','VH','ttH'] 
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
#train_vars.append('HTXS_stage1_2_cat_pTjet30GeV')

dataframes = []
dataframes.append(pd.read_csv('2017/MC/DataFrames/ggH_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/VBF_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/VH_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/ttH_VBF_BDT_df_2017.csv'))
df = pd.concat(dataframes, sort=False, axis=0 )

data = df[train_vars]

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
#num_categories = data['proc'].nunique()
#y_train_labels_num, y_train_labels_def = pd.factorize(data['proc'])

num_categories = data['proc_new'].nunique()
y_train_labels_num, y_train_labels_def = pd.factorize(data['proc_new'])

#Label definition:
print('Label Definition:')
label_def = []
for i in range(num_categories):
    label_def.append([i,y_train_labels_def[i]])
    print(i,y_train_labels_def[i])

data['proc_num'] = y_train_labels_num

y_train_labels = np.array(data['proc_new'])
#y_train_labels = np.array(data['proc'])
y_train_labels_num = np.array(data['proc_num'])
y_train_labels_hot = np_utils.to_categorical(y_train_labels_num, num_classes=num_categories)
weights = np.array(data['weight'])

data = data.drop(columns=['proc'])
data = data.drop(columns=['proc_num'])
data = data.drop(columns=['weight'])
data = data.drop(columns=['HTXS_stage_0'])
data = data.drop(columns=['proc_new'])
#data = data.drop(columns=['HTXS_stage1_2_cat_pTjet30GeV'])

#With num
x_train, x_test, y_train, y_test, train_w, test_w, proc_arr_train, proc_arr_test = train_test_split(data, y_train_labels_num, weights, y_train_labels, test_size = test_split, shuffle = True)
#With hot
#x_train, x_test, y_train, y_test, train_w, test_w, proc_arr_train, proc_arr_test = train_test_split(data, y_train_labels_hot, weights, y_train_labels, test_size = val_split, shuffle = True)

#Before n_estimators = 100, maxdepth=4, gamma = 1
#Improved n_estimators = 300, maxdepth = 7, gamme = 4
clf = xgb.XGBClassifier(objective='multi:softprob', n_estimators=100, 
                            eta=0.1, maxDepth=4, min_child_weight=0.01, 
                            subsample=0.6, colsample_bytree=0.6, gamma=1,
                            num_class=4)

#Equalizing weights
train_w_df = pd.DataFrame()
train_w = 300 * train_w # to make loss function O(1)
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

print (' Training classifier...')
clf = clf.fit(x_train, y_train, sample_weight=train_w)
print ('Finished Training classifier!')

#print('Saving Classifier...')
#pickle.dump(clf, open("models/Multi_BDT_clf.pickle.dat", "wb"))
#print('Finished Saving classifier!')

#print('loading classifier:')
#clf = pickle.load(open("models/Multi_BDT_clf.pickle.dat", "rb"))

y_pred_test = clf.predict_proba(x_test)

x_test['proc'] = proc_arr_test
x_test['weight'] = test_w
x_test['output_score_ggh'] = y_pred_test[:,0]
x_test['output_score_qqh'] = y_pred_test[:,1]
x_test['output_score_vh'] = y_pred_test[:,2]
x_test['output_score_tth'] = y_pred_test[:,3]

output_score_ggh = np.array(y_pred_test[:,0])
output_score_qqh = np.array(y_pred_test[:,1])
output_score_vh = np.array(y_pred_test[:,2])
output_score_tth = np.array(y_pred_test[:,3])

x_test_ggh = x_test[x_test['proc'] == 'ggH']
x_test_qqh = x_test[x_test['proc'] == 'qqH']
x_test_vh = x_test[x_test['proc'] == 'VH']
x_test_tth = x_test[x_test['proc'] == 'ttH']

ggh_w = x_test_ggh['weight'] / x_test_ggh['weight'].sum()
qqh_w = x_test_qqh['weight'] / x_test_qqh['weight'].sum()
vh_w = x_test_vh['weight'] / x_test_vh['weight'].sum()
tth_w = x_test_tth['weight'] / x_test_tth['weight'].sum()

total_w = x_test['weight'] / x_test['weight'].sum()

#Accuracy Score
y_pred = y_pred_test.argmax(axis=1)
#y_true = y_test.argmax(axis=1)
y_true = y_test
print 'Accuracy score: '
NNaccuracy = accuracy_score(y_true, y_pred)
print(NNaccuracy)

#Confusion Matrix
cm = confusion_matrix(y_true=y_true,y_pred=y_pred)

#Calculations for the ROC curve
y_true_ggh = np.where(y_true == 0, 1, 0)
y_pred_ggh = np.where(y_pred == 0, 1, 0)
y_pred_ggh_prob = []
for i in range(len(y_pred_ggh)):
    if y_pred_ggh[i] == 0:
        y_pred_ggh_prob.append(0)
    elif y_pred_ggh[i] == 1:
        y_pred_ggh_prob.append(output_score_ggh[i])
y_true_qqh = np.where(y_true == 1, 1, 0)
y_pred_qqh = np.where(y_pred == 1, 1, 0)
y_pred_qqh_prob = []
for i in range(len(y_pred_qqh)):
    if y_pred_qqh[i] == 0:
        y_pred_qqh_prob.append(0)
    elif y_pred_qqh[i] == 1:
        y_pred_qqh_prob.append(output_score_qqh[i])
y_true_vh = np.where(y_true == 2, 1, 0)
y_pred_vh = np.where(y_pred == 2, 1, 0)
y_pred_vh_prob = []
for i in range(len(y_pred_vh)):
    if y_pred_vh[i] == 0:
        y_pred_vh_prob.append(0)
    elif y_pred_vh[i] == 1:
        y_pred_vh_prob.append(output_score_vh[i])
y_true_tth = np.where(y_true == 3, 1, 0)
y_pred_tth = np.where(y_pred == 3, 1, 0)
y_pred_tth_prob = []
for i in range(len(y_pred_tth)):
    if y_pred_tth[i] == 0:
        y_pred_tth_prob.append(0)
    elif y_pred_tth[i] == 1:
        y_pred_tth_prob.append(output_score_tth[i])

#Plotting:
def roc_score(y_true = y_true, y_pred = y_pred_test):

    fpr_keras_ggh, tpr_keras_ggh, thresholds_keras_ggh = roc_curve(y_true_ggh, y_pred_ggh_prob,sample_weight=total_w)
    fpr_keras_ggh.sort()
    tpr_keras_ggh.sort()
    auc_keras_test_ggh = auc(fpr_keras_ggh,tpr_keras_ggh)
    print("Area under ROC curve for ggH (test): ", auc_keras_test_ggh)

    fpr_keras_qqh, tpr_keras_qqh, thresholds_keras_qqh = roc_curve(y_true_qqh, y_pred_qqh_prob,sample_weight=total_w)
    fpr_keras_qqh.sort()
    tpr_keras_qqh.sort()
    auc_keras_test_qqh = auc(fpr_keras_qqh,tpr_keras_qqh)
    print("Area under ROC curve for qqH (test): ", auc_keras_test_qqh)

    fpr_keras_vh, tpr_keras_vh, thresholds_keras_vh = roc_curve(y_true_vh, y_pred_vh_prob,sample_weight=total_w)
    fpr_keras_vh.sort()
    tpr_keras_vh.sort()
    auc_keras_test_vh = auc(fpr_keras_vh,tpr_keras_vh)
    print("Area under ROC curve for VH (test): ", auc_keras_test_vh)

    fpr_keras_tth, tpr_keras_tth, thresholds_keras_tth = roc_curve(y_true_tth, y_pred_tth_prob,sample_weight=total_w)
    fpr_keras_tth.sort()
    tpr_keras_tth.sort()
    auc_keras_test_tth = auc(fpr_keras_tth,tpr_keras_tth)
    print("Area under ROC curve for ttH (test): ", auc_keras_test_tth)

    print("Plotting ROC Score")
    fig, ax = plt.subplots()
    ax.plot(fpr_keras_ggh, tpr_keras_ggh, label = 'ggH (area = %0.2f)'%auc_keras_test_ggh)
    ax.plot(fpr_keras_qqh, tpr_keras_qqh, label = 'qqH (area = %0.2f)'%auc_keras_test_qqh)
    ax.plot(fpr_keras_vh, tpr_keras_vh, label = 'VH (area = %0.2f)'%auc_keras_test_vh)
    ax.plot(fpr_keras_tth, tpr_keras_tth, label = 'ttH (area = %0.2f)'%auc_keras_test_tth)
    ax.legend()
    ax.set_xlabel('Background Efficiency', ha='right', x=1, size=9)
    ax.set_ylabel('Signal Efficiency',ha='right', y=1, size=9)
    ax.grid(True, 'major', linestyle='solid', color='grey', alpha=0.5)
    name = 'plotting/BDT_plots/BDT_Multi_ROC_curve'
    plt.savefig(name, dpi = 200)

#Confusion Matrix
def plot_confusion_matrix(cm,classes,normalize=True,title='Confusion matrix',cmap=plt.cm.Blues):
    fig, ax = plt.subplots(1)
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
    name = 'plotting/BDT_plots/BDT_Multi_Confusion_Matrix'
    fig.savefig(name)


def plot_output_score(data='output_score_qqh', density=False,):
    #Can then change it to plotting proc
    print('Plotting',data)
    output_score_ggh = np.array(x_test_ggh[data])
    output_score_qqh = np.array(x_test_qqh[data])
    output_score_vh = np.array(x_test_vh[data])
    output_score_tth = np.array(x_test_tth[data])

    fig, ax = plt.subplots()
    ax.hist(output_score_ggh, bins=50, label='ggH', histtype='step',weights=ggh_w)#,density=True) 
    ax.hist(output_score_qqh, bins=50, label='qqH', histtype='step',weights=qqh_w) #density=True)
    ax.hist(output_score_vh, bins=50, label='VH', histtype='step',weights=vh_w) #density=True) 
    ax.hist(output_score_tth, bins=50, label='ttH', histtype='step',weights=tth_w) #density=True)
    plt.legend()
    plt.title('Output Score')
    plt.ylabel('Fraction of Events')
    plt.xlabel('BDT Score')
    name = 'plotting/BDT_plots/BDT_Multi_'+data
    plt.savefig(name, dpi = 200)


# Feature Importance

def feature_importance(num_plots='single',num_feature=20,imp_type='gain',values = False):
    if num_plots == 'single':
        plt.rcParams["figure.figsize"] = (14,7)
        xgb.plot_importance(clf, max_num_features=num_feature, grid = False, height = 0.4, importance_type = imp_type, title = 'Feature importance ({})'.format(imp_type), show_values = values, color ='blue')
        plt.savefig('plotting/BDT_plots/feature_importance_{0}.png'.format(imp_type), dpi = 200)
        print('saving: /plotting/BDT_plots/feature_importance_{0}.png'.format(imp_type))
        
    else:
        imp_types = ['weight','gain','cover']
        for i in imp_types:
            xgb.plot_importance(clf, max_num_features=num_feature, grid = False, height = 0.4, importance_type = imp_type, title = 'Feature importance ({})'.format(i), show_values = values, color ='blue')
            plt.savefig('/plotting/BDT_plots/feature_importance_{0}.png'.format(i), dpi = 200)
            print('saving: /plotting/BDT_plots/feature_importance_{0}.png'.format(i))

plot_output_score(data='output_score_qqh')
plot_output_score(data='output_score_ggh')
plot_output_score(data='output_score_vh')
plot_output_score(data='output_score_tth')

feature_importance()

roc_score()

plot_confusion_matrix(cm,binNames,normalize=True)
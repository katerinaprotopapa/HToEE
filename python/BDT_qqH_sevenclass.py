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
num_estimators = 400
test_split = 0.15
learning_rate = 0.001

#STXS mapping
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

binNames = ['QQ2HQQ_FWDH','rest','QQ2HQQ_GE2J_MJJ_60_120','QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200','QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25','QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25','QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25','QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_GT25','WH','ZH'] 
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

dataframes = []
#dataframes.append(pd.read_csv('2017/MC/DataFrames/ggH_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/VBF_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/VH_VBF_BDT_df_2017.csv'))
#dataframes.append(pd.read_csv('2017/MC/DataFrames/ttH_VBF_BDT_df_2017.csv'))
#dataframes.append(pd.read_csv('2017/MC/DataFrames/tHq_VBF_BDT_df_2017.csv'))
#dataframes.append(pd.read_csv('2017/MC/DataFrames/tHW_VBF_BDT_df_2017.csv'))
df = pd.concat(dataframes, sort=False, axis=0 )

data = df[train_vars]

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

data['proc_new'] = mapping(map_list=map_def_2,stage=data['HTXS_stage1_2_cat_pTjet30GeV'])

# now I only want to keep the qqH - 7class
data = data.drop(data[(data.proc_new == 'QQ2HQQ_FWDH') & (data.proc_new == 'WH') & (data.proc_new == 'ZH')].index)

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
data = data.drop(columns=['HTXS_stage1_2_cat_pTjet30GeV'])

#With num
x_train, x_test, y_train, y_test, train_w, test_w, proc_arr_train, proc_arr_test = train_test_split(data, y_train_labels_num, weights, y_train_labels, test_size = test_split, shuffle = True)
#With hot
#x_train, x_test, y_train, y_test, train_w, test_w, proc_arr_train, proc_arr_test = train_test_split(data, y_train_labels_hot, weights, y_train_labels, test_size = val_split, shuffle = True)

#Before n_estimators = 100, maxdepth=4, gamma = 1
#Improved n_estimators = 300, maxdepth = 7, gamme = 4
clf = xgb.XGBClassifier(objective='multi:softprob', n_estimators=100, 
                            eta=0.1, maxDepth=6, min_child_weight=0.01, 
                            subsample=0.6, colsample_bytree=0.6, gamma=4,
                            num_class=4)

#Equalizing weights
#Equalizing weights
train_w_df = pd.DataFrame()
train_w = 300 * train_w # to make loss function O(1)
train_w_df['weight'] = train_w
train_w_df['proc'] = proc_arr_train
qqh1_sum_w = train_w_df[train_w_df['proc'] == 'rest']['weight'].sum()
qqh2_sum_w = train_w_df[train_w_df['proc'] == 'QQ2HQQ_GE2J_MJJ_60_120']['weight'].sum()
qqh3_sum_w = train_w_df[train_w_df['proc'] == 'QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200']['weight'].sum()
qqh4_sum_w = train_w_df[train_w_df['proc'] == 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25']['weight'].sum()
qqh5_sum_w = train_w_df[train_w_df['proc'] == 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25']['weight'].sum()
qqh6_sum_w = train_w_df[train_w_df['proc'] == 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25']['weight'].sum()
qqh7_sum_w = train_w_df[train_w_df['proc'] == 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_GT25']['weight'].sum()

train_w_df.loc[train_w_df.proc == 'QQ2HQQ_GE2J_MJJ_60_120','weight'] = (train_w_df[train_w_df['proc'] == 'QQ2HQQ_GE2J_MJJ_60_120']['weight'] * qqH1_sum_w / qqh2_sum_w)
train_w_df.loc[train_w_df.proc == 'QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200','weight'] = (train_w_df[train_w_df['proc'] == 'QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200']['weight'] * qqH1_sum_w / qqh3_sum_w)
train_w_df.loc[train_w_df.proc == 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25','weight'] = (train_w_df[train_w_df['proc'] == 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25']['weight'] * qqH1_sum_w / qqh4_sum_w)
train_w_df.loc[train_w_df.proc == 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25','weight'] = (train_w_df[train_w_df['proc'] == 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25']['weight'] * qqH1_sum_w / qqh5_sum_w)
train_w_df.loc[train_w_df.proc == 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25','weight'] = (train_w_df[train_w_df['proc'] == 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25']['weight'] * qqH1_sum_w / qqh6_sum_w)
train_w_df.loc[train_w_df.proc == 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_GT25','weight'] = (train_w_df[train_w_df['proc'] == 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_GT25']['weight'] * qqH1_sum_w / qqh7_sum_w)
train_w = np.array(train_w_df['weight'])


print (' Training classifier...')
clf = clf.fit(x_train, y_train, sample_weight=train_w)
print ('Finished Training classifier!')

#print('Saving Classifier...')
#pickle.dump(clf, open("models/Multi_BDT_clf.pickle.dat", "wb"))
#print('Finished Saving classifier!')

#print('loading classifier:')
#clf = pickle.load(open("models/Multi_BDT_clf.pickle.dat", "rb"))
# Output Score
y_pred_test = clf.predict_proba(x_test)

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

x_test_qqh1 = x_test[x_test['proc'] == 'rest']
x_test_qqh2 = x_test[x_test['proc'] == 'QQ2HQQ_GE2J_MJJ_60_120']
x_test_qqh3 = x_test[x_test['proc'] == 'QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200']
x_test_qqh4 = x_test[x_test['proc'] == 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25']
x_test_qqh5 = x_test[x_test['proc'] == 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25']
x_test_qqh6 = x_test[x_test['proc'] == 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25']
x_test_qqh7 = x_test[x_test['proc'] == 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_GT25']

qqh1_w = x_test_qqh1['weight'] / x_test_qqh1['weight'].sum()
qqh2_w = x_test_qqh2['weight'] / x_test_qqh2['weight'].sum()
qqh3_w = x_test_qqh3['weight'] / x_test_qqh3['weight'].sum()
qqh4_w = x_test_qqh4['weight'] / x_test_qqh4['weight'].sum()
qqh5_w = x_test_qqh5['weight'] / x_test_qqh5['weight'].sum()
qqh6_w = x_test_qqh6['weight'] / x_test_qqh6['weight'].sum()
qqh7_w = x_test_qqh7['weight'] / x_test_qqh7['weight'].sum()
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

#Confusion Matrix
def plot_confusion_matrix(cm,classes,normalize=True,title='Confusion matrix',cmap=plt.cm.Blues):
    fig, ax = plt.subplots(figsize = (12,12))
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
    name = 'plotting/BDT_plots/BDT_qqH_Sevenclass_Confusion_Matrix'
    fig.savefig(name, dpi = 500)


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
    ax.hist(output_score_ggh, bins=50, label='ggH', histtype='step',weights=ggh_w)#,density=True) 
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
    plt.xlabel('BDT Score')
    name = 'plotting/BDT_plots/BDT_qqH_Sevenclass_'+data
    plt.savefig(name, dpi = 300)



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

plot_confusion_matrix(cm,binNames,normalize=True)

plot_output_score(data='output_score_qqh1')
plot_output_score(data='output_score_qqh2')
plot_output_score(data='output_score_qqh3')
plot_output_score(data='output_score_qqh4')
plot_output_score(data='output_score_qqh5')
plot_output_score(data='output_score_qqh6')
plot_output_score(data='output_score_qqh7')

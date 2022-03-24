import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import seaborn as sns
 

dataframes = []
#dataframes.append(pd.read_csv('2017/MC/DataFrames/ggH_VBF_BDT_df_2017.csv', nrows = 150000))
dataframes.append(pd.read_csv('2017/MC/DataFrames/VBF_VBF_BDT_df_2017.csv', nrows = 150000))
dataframes.append(pd.read_csv('2017/MC/DataFrames/VH_VBF_BDT_df_2017.csv'))
#dataframes.append(pd.read_csv('2017/MC/DataFrames/ttH_VBF_BDT_df_2017.csv'))
#dataframes.append(pd.read_csv('2017/MC/DataFrames/tHq_VBF_BDT_df_2017.csv', nrows = 150000))
#dataframes.append(pd.read_csv('2017/MC/DataFrames/tHW_VBF_BDT_df_2017.csv', nrows = 130900))
df = pd.concat(dataframes, sort=False, axis=0 )

map_def = [['ggH',10,11],['qqH',20,21,22,23],['VH',30,31,40,41],['ttH',60,61],['tH',80,81]]

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

df = df[train_vars]

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

df['proc_new'] = mapping(map_list=map_def,stage=df['HTXS_stage_0'])

df = df.drop(columns=['proc','weight','HTXS_stage_0'])

# -------------------------------

#Adapted code to plot a 2D matrix of Pearson Correlation coefficients
"""
# Background Grouping
# QCD
df['proc'] = np.where(df['proc'] == 'QCD30toinf', 'QCD', df['proc'])
df['proc'] = np.where(df['proc'] == 'QCD40toinf', 'QCD', df['proc'])
df['proc'] = np.where(df['proc'] == 'QCD30to40', 'QCD', df['proc'])

# Gjet
df['proc'] = np.where(df['proc'] == 'GJet20to40', 'Gjet', df['proc'])
df['proc'] = np.where(df['proc'] == 'GJet40toinf', 'Gjet', df['proc'])
df['proc'] = np.where(df['proc'] == 'GJet20toinf', 'Gjet', df['proc'])

# Diphoton
df['proc'] = np.where(df['proc'] == 'Diphoton40to80', 'Diphoton', df['proc'])
df['proc'] = np.where(df['proc'] == 'Diphoton80toinf', 'Diphoton', df['proc'])

#Creating the min and max_IDMVA variables
df['min_IDMVA'] = df[['leadPhotonIDMVA', 'subleadPhotonIDMVA']].min(axis=1)
df['max_IDMVA'] = df[['leadPhotonIDMVA', 'subleadPhotonIDMVA']].max(axis=1)

#Removing variables that we don't need
df = df.drop(labels='Unnamed: 0',axis=1)
#df = df.drop(labels='proc',axis=1)
df = df.drop(labels='year',axis=1)
df = df.drop(labels='weight',axis=1)
df = df.drop(labels='centralObjectWeight',axis=1)
df = df.drop(labels='genWeight',axis=1)
"""
#Introducing a split between the 4 different production modes
df_ggh = df[df['proc_new']=='ggH']
df_vbf = df[df['proc_new']=='qqH']
df_vh = df[df['proc_new']=='VH'] 
df_tth = df[df['proc_new']=='ttH']
df_th = df[df['proc_new']=='tH']

corr = df.corr()
corr_ggh = df_ggh.corr()
corr_vbf = df_vbf.corr()
corr_vh = df_vh.corr()
corr_tth = df_tth.corr()
corr_th = df_th.corr()

#Need to run the following command to display all rows
pd.set_option('display.max_rows', 1000)

corr_matrix = df.corr().abs()
corr_matrix_ggh = df_ggh.corr().abs()
corr_matrix_vbf = df_vbf.corr().abs()
corr_matrix_vh = df_vh.corr().abs()
corr_matrix_tth = df_tth.corr().abs()
corr_matrix_th = df_th.corr().abs()

corr_list = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)).stack().sort_values(ascending=False))
corr_list_ggh = (corr_matrix_ggh.where(np.triu(np.ones(corr_matrix_ggh.shape), k=1).astype(np.bool)).stack().sort_values(ascending=False))
corr_list_vbf = (corr_matrix_vbf.where(np.triu(np.ones(corr_matrix_vbf.shape), k=1).astype(np.bool)).stack().sort_values(ascending=False))
corr_list_vh = (corr_matrix_vh.where(np.triu(np.ones(corr_matrix_vh.shape), k=1).astype(np.bool)).stack().sort_values(ascending=False))
corr_list_tth = (corr_matrix_tth.where(np.triu(np.ones(corr_matrix_tth.shape), k=1).astype(np.bool)).stack().sort_values(ascending=False))
corr_list_th = (corr_matrix_th.where(np.triu(np.ones(corr_matrix_th.shape), k=1).astype(np.bool)).stack().sort_values(ascending=False))

"""
print('Total Signal')
print(corr_list)
print('ggH')
print(corr_list_ggh)
print('qqH')
print(corr_list_vbf)
print('VH')
print(corr_list_vh)
print('ttH')
print(corr_list_tth)
print('tH')
print(corr_list_th)
"""

#Plotting

fig, ax = plt.subplots(1,figsize=(17,17))
#ax.set_clim(-1, 1)
ax = sns.heatmap(corr, cmap='coolwarm',cbar=True,xticklabels=corr.columns.values,yticklabels=corr.columns.values,square=True, vmin=-1, vmax=1).set_title("Combined Signal")
name = 'plotting/plots/Pearson_Signal'
fig.savefig(name, dpi = 1200)

fig, ax = plt.subplots(1,figsize=(17,17))
#ax.set_clim(-1, 1)
ax = sns.heatmap(corr_ggh, cmap='coolwarm',cbar=True,xticklabels=corr_ggh.columns.values,yticklabels=corr_ggh.columns.values,square=True, vmin=-1, vmax=1).set_title("ggH")
name = 'plotting/plots/Pearson_ggh'
fig.savefig(name, dpi = 1200)

fig, ax = plt.subplots(1,figsize=(17,17))
#ax.set_clim(-1, 1)
ax = sns.heatmap(corr_vbf, cmap='coolwarm',cbar=True,xticklabels=corr_vbf.columns.values,yticklabels=corr_vbf.columns.values,square=True, vmin=-1, vmax=1).set_title("qqH")
name = 'plotting/plots/Pearson_vbf'
fig.savefig(name, dpi = 1200)

fig, ax = plt.subplots(1,figsize=(17,17))
#ax.set_clim(-1, 1)
ax = sns.heatmap(corr_vh, cmap='coolwarm',cbar=True,xticklabels=corr_vh.columns.values,yticklabels=corr_vh.columns.values,square=True, vmin=-1, vmax=1).set_title("VH leptonic")
name = 'plotting/plots/Pearson_vh'
fig.savefig(name, dpi = 1200)

fig, ax = plt.subplots(1,figsize=(17,17))
#ax.set_clim(-1, 1)
ax = sns.heatmap(corr_tth, cmap='coolwarm',cbar=True,xticklabels=corr_tth.columns.values,yticklabels=corr_tth.columns.values,square=True, vmin=-1, vmax=1).set_title("ttH")
name = 'plotting/plots/Pearson_tth'
fig.savefig(name, dpi = 1200)

fig, ax = plt.subplots(1,figsize=(17,17))
#ax.set_clim(-1, 1)
ax = sns.heatmap(corr_th, cmap='coolwarm',cbar=True,xticklabels=corr_th.columns.values,yticklabels=corr_th.columns.values,square=True, vmin=-1, vmax=1).set_title("tH")
name = 'plotting/plots/Pearson_th'
fig.savefig(name, dpi = 1200)

print('Done')
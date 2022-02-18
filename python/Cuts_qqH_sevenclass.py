import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



#Load the dataframe
dataframes = []
dataframes.append(pd.read_csv('2017/MC/DataFrames/ggH_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/VBF_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/VH_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/ttH_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/tHq_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/tHW_VBF_BDT_df_2017.csv'))
data = pd.concat(dataframes, sort=False, axis=0 )

# well also need to only keep the qqH 7-class btw

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
for i in range(data.shape[0]):
    #print('eeee')
    if njets[i] == 0 or njets[i] == 1:
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
data['proc_new'] = proc

# what I need to do
# qqH - 7 class cuts: to have a dataframe with only qqH [qqH proc from 1.2 and proc_new - our predictions from cuts] + subsub
# qqH - 7class NN and BDT
# confusion matrices of qqH 7-class NN, BDT, cuts
# check labeling




















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

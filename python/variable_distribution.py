import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob

map_def = [['ggH',10,11],['qqH',20,21,22,23],['VH',30,31,40,41],['ttH',60,61],['tH',80,81]]
colors = ['#54aaf8', '#f08633', '#1fcf57', '#cf1f9f', 'gold']

#Load the dataframe
dataframes = []
dataframes.append(pd.read_csv('2017/MC/DataFrames/ggH_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/VBF_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/VH_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/ttH_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/tHq_VBF_BDT_df_2017.csv', nrows = 254039))
dataframes.append(pd.read_csv('2017/MC/DataFrames/tHW_VBF_BDT_df_2017.csv', nrows = 130900))
df = pd.concat(dataframes, sort=False, axis=0)

print('Loaded dataframe')

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


list_variables = [#'diphotonPt'
                # 'leadPhotonEta'
                #, 'subleadPhotonEta' 
                #, 'leadPhotonIDMVA' 
                #, 'subleadPhotonIDMVA'
                #, 'diphotonMass' 
                #, 'weight' 
                #, 'centralObjectWeight' 
                #, 'leadPhotonPtOvM'
                #, 'dijetMass' 
                #, 'subleadPhotonPtOvM'
                # 'leadJetPt'
                #, 'subleadJetPt' #'leadElectronIDMVA', 'subleadElectronIDMVA',
                #, 'dijetAbsDEta' 
                #, 'dijetDPhi'
                #, 'diphotonCosPhi'
                #, 'leadJetPUJID'
                #, 'subleadJetPUJID'
                #, 'subsubleadJetEn'
                #, 'subsubleadJetPt'
                #, 'subsubleadJetEta'
                #, 'subsubleadJetPhi' 
                #, 'min_IDMVA'
                #, 'max_IDMVA'
                #,'dijetCentrality'
                #'leadJetBTagScore' 
                # 'subleadJetBTagScore', 'subsubleadJetBTagScore'
                #, 'leadJetMass' ,'leadPhotonEn', 'leadPhotonMass' ,
                # 'leadPhotonPt'
                #, 'subleadJetMass', 'subleadPhotonEn', 'subleadPhotonMass', 'subleadPhotonPt'
                #, 'DR' 
                #'leadPhotonPhi','leadJetDiphoDEta','leadJetDiphoDPhi','leadJetEn','leadJetEta','leadJetPhi'
                #'subleadPhotonPhi','subsubleadPhotonEn','subsubleadJetMass','subsubleadPhotonMass',
                #'subsubleadPhotonPt','subsubleadPhotonEta','subsubleadPhotonPhi','subleadJetDiphoDPhi',
                #'subleadJetDiphoDEta','subleadJetEn','subleadJetEta','subleadJetPhi','subsubleadPhotonIDMVA',				
                #'diphotonEta','diphotonPhi','dijetPt','dijetEta','dijetPhi','dijetMinDRJetPho','dijetDiphoAbsDEta',
                #'nSoftJets'
                #,
                #'leadElectronEn'
                #'leadElectronMass', 
                'leadElectronPt'
]

for variable in list_variables:
    
    print('Inside loop')
    # Signal 
    vbf_sig_0 = np.array(df[df['proc_new'] == 'qqH'][variable])
    vbf_sig_w = np.array(df[df['proc_new'] == 'qqH']['weight'])[(vbf_sig_0 > -10) & (vbf_sig_0 <2000)]
    #vbf_sig_w = vbf_sig_w / np.sum(vbf_sig_w)
    vbf_sig = vbf_sig_0[(vbf_sig_0 > -10) & (vbf_sig_0 < 2000)]
    print('vbf')
    vh_sig_0 = np.array(df[df['proc_new'] == 'VH'][variable])
    vh_sig_w = np.array(df[df['proc_new'] == 'VH']['weight'])[(vh_sig_0 > -10) & (vh_sig_0 <2000)]
    #vh_sig_w = vh_sig_w / np.sum(vh_sig_w)
    vh_sig = vh_sig_0[(vh_sig_0 > -10) & (vh_sig_0 <2000)]
    print('vh')
    ggh_sig_0 = np.array(df[df['proc_new'] == 'ggH'][variable])
    ggh_sig_w = np.array(df[df['proc_new'] == 'ggH']['weight'])[(ggh_sig_0 > -10) & (ggh_sig_0 <2000)]
    #ggh_sig_w = ggh_sig_w / np.sum(ggh_sig_w)
    ggh_sig = ggh_sig_0[(ggh_sig_0 > -10) & (ggh_sig_0 <2000)]
    print('ggh')
    tth_sig_0 = np.array(df[df['proc_new'] == 'ttH'][variable])
    tth_sig_w = np.array(df[df['proc_new'] == 'ttH']['weight'])[(tth_sig_0 > -10) & (tth_sig_0 <2000)]
    #tth_sig_w = tth_sig_w / np.sum(tth_sig_w)
    tth_sig = tth_sig_0[(tth_sig_0 > -10) & (tth_sig_0 <2000)]
    print('ttH')
    th_sig_0 = np.array(df[df['proc_new'] == 'tH'][variable])
    th_sig_w = np.array(df[df['proc_new'] == 'tH']['weight'])[(th_sig_0 > -10) & (th_sig_0 <2000)]
    th_sig = th_sig_0[(th_sig_0 > -10) & (th_sig_0 <2000)]
    print('tH')
    #thW_sig_0 = np.array(df[df['proc_new'] == 'tHW'][variable])
    #thW_sig_w = np.array(df[df['proc_new'] == 'tHW']['weight'])[(thW_sig_0 > -10) & (thW_sig_0 <2000)]
    #thW_sig = thW_sig_0[(thW_sig_0 > -10) & (thW_sig_0 <2000)]
    #print('tHW')
    #th_sig = np.concatenate((thq_sig, thW_sig))
    #print('tH: ', th_sig)
    #th_sig_w = np.concatenate((thq_sig_w, thW_sig_w))
    #th_sig_w = th_sig_w / np.sum(th_sig_w)
    #print('tH_w: ', th_sig_w)
    # Now let's plot the histogram

    scale = 100
    num_bins = 200
    normalize = True

    fig, ax = plt.subplots()
    plt.rcParams.update({'font.size': 9})
 
    # signal
    ax.hist(ggh_sig, bins = num_bins, density = normalize, color = colors[0], label = 'ggH', stacked = True, histtype = 'step', weights = scale * ggh_sig_w)
    ax.hist(vbf_sig, bins = num_bins, density = normalize, color = colors[1], label = 'qqH', histtype = 'step', weights = scale * vbf_sig_w, alpha = 1)
    ax.hist(vh_sig, bins = num_bins, density = normalize, color = colors[2], label = 'VH leptonic', stacked = True, histtype = 'step', weights = scale * vh_sig_w)
    ax.hist(tth_sig, bins = num_bins, density = normalize, color = colors[3], label = 'ttH', stacked = True, histtype = 'step', weights = scale * tth_sig_w)
    ax.hist(th_sig, bins = num_bins, density = normalize, color = colors[4], label = 'tH', stacked = True, histtype = 'step', weights = scale * th_sig_w)

    ax.set_xlim(0,100)
    #ax.set_ylim(0,20)
    #ax.set_xticks([0,50,100,150,200,250,300])
    #ax.set_yticks([0,0.005,0.01, 0.015,0.02, 0.025, 0.03, 0.035, 0.04])
    ax.legend(loc = 'upper right')
    ax.set_xlabel('Lead electron $p_T$ [GeV]', ha='center', size = 10)
    ax.set_ylabel('Fraction of events',ha='center', size = 10)
    #ax.grid(True, 'major', linestyle='solid', color='grey', alpha=0.5)
    plt.tight_layout()
    ax.grid(True, 'major', linestyle='dotted', color='grey', alpha=0.5)
    name = 'plotting/plots/' + variable 
    #print('Plotting leadJetPt plot')
    fig.savefig(name, dpi=1200)











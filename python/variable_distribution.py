import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob

#Load the dataframe
dataframes = []
dataframes.append(pd.read_csv('2017/MC/DataFrames/ggH_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/VBF_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/VH_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/ttH_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/tHq_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/tHW_VBF_BDT_df_2017.csv'))
df = pd.concat(dataframes, sort=False, axis=0)

print('Loaded dataframe')


list_variables = ['diphotonPt'
                #, 'leadPhotonEta'
                #, 'subleadPhotonEta' 
                #, 'leadPhotonIDMVA' 
                #, 'subleadPhotonIDMVA'
                #, 'diphotonMass' 
                #, 'weight' 
                #, 'centralObjectWeight' 
                #, 'leadPhotonPtOvM'
                , 'dijetMass' 
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
                 ,'dijetCentrality'
                # 'leadJetBTagScore' 
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
                ,'leadElectronEn', 'leadElectronMass', 'leadElectronPt'
]

for variable in list_variables:
    
    print('Inside loop')
    # Signal 
    vbf_sig_0 = np.array(df[df['proc'] == 'VBF'][variable])
    vbf_sig_w = np.array(df[df['proc'] == 'VBF']['weight'])[(vbf_sig_0 > -10) & (vbf_sig_0 <2000)]
    #vbf_sig_w = vbf_sig_w / np.sum(vbf_sig_w)
    vbf_sig = vbf_sig_0[(vbf_sig_0 > -10) & (vbf_sig_0 < 2000)]
    print('vbf')
    vh_sig_0 = np.array(df[df['proc'] == 'VH'][variable])
    vh_sig_w = np.array(df[df['proc'] == 'VH']['weight'])[(vh_sig_0 > -10) & (vh_sig_0 <2000)]
    #vh_sig_w = vh_sig_w / np.sum(vh_sig_w)
    vh_sig = vh_sig_0[(vh_sig_0 > -10) & (vh_sig_0 <2000)]
    print('vh')
    ggh_sig_0 = np.array(df[df['proc'] == 'ggH'][variable])
    ggh_sig_w = np.array(df[df['proc'] == 'ggH']['weight'])[(ggh_sig_0 > -10) & (ggh_sig_0 <2000)]
    #ggh_sig_w = ggh_sig_w / np.sum(ggh_sig_w)
    ggh_sig = ggh_sig_0[(ggh_sig_0 > -10) & (ggh_sig_0 <2000)]
    print('ggh')
    tth_sig_0 = np.array(df[df['proc'] == 'ttH'][variable])
    tth_sig_w = np.array(df[df['proc'] == 'ttH']['weight'])[(tth_sig_0 > -10) & (tth_sig_0 <2000)]
    #tth_sig_w = tth_sig_w / np.sum(tth_sig_w)
    tth_sig = tth_sig_0[(tth_sig_0 > -10) & (tth_sig_0 <2000)]
    print('ttH')
    thq_sig_0 = np.array(df[df['proc'] == 'tHq'][variable])
    thq_sig_w = np.array(df[df['proc'] == 'tHq']['weight'])[(thq_sig_0 > -10) & (thq_sig_0 <2000)]
    thq_sig = thq_sig_0[(thq_sig_0 > -10) & (thq_sig_0 <2000)]
    print('tHq')
    thW_sig_0 = np.array(df[df['proc'] == 'tHW'][variable])
    thW_sig_w = np.array(df[df['proc'] == 'tHW']['weight'])[(thW_sig_0 > -10) & (thW_sig_0 <2000)]
    thW_sig = thW_sig_0[(thW_sig_0 > -10) & (thW_sig_0 <2000)]
    print('tHW')
    th_sig = np.concatenate((thq_sig, thW_sig))
    print('tH: ', th_sig)
    th_sig_w = np.concatenate((thq_sig_w, thW_sig_w))
    #th_sig_w = th_sig_w / np.sum(th_sig_w)
    print('tH_w: ', th_sig_w)
    # Now let's plot the histogram

    scale = 100
    num_bins = 300
    normalize = True

    fig, ax = plt.subplots()

    # signal
    ax.hist(ggh_sig, bins = num_bins, density = normalize, color = '#24b1c9', label = 'ggH', stacked = True, histtype = 'step', weights = scale * ggh_sig_w)
    ax.hist(vbf_sig, bins = num_bins, density = normalize, color = '#e36b1e', label = 'VBF', histtype = 'step', weights = scale * vbf_sig_w, alpha = 1)
    ax.hist(vh_sig, bins = num_bins, density = normalize, color = '#1eb037', label = 'VH', stacked = True, histtype = 'step', weights = scale * vh_sig_w)
    ax.hist(tth_sig, bins = num_bins, density = normalize, color = '#c21bcf', label = 'ttH', stacked = True, histtype = 'step', weights = scale * tth_sig_w)
    ax.hist(th_sig, bins = num_bins, density = normalize, color = '#dbb104', label = 'tH', stacked = True, histtype = 'step', weights = scale * th_sig_w)

    #ax.set_xlim(0,300)
    ax.set_ylim(0,0.04)
    #ax.set_xticks([0,50,100,150,200,250,300])
    #ax.set_yticks([0,0.005,0.01, 0.015,0.02, 0.025, 0.03, 0.035, 0.04])
    ax.legend(loc = 'upper right')
    ax.set_xlabel(variable, ha='center',x=0.5, size = 12)
    ax.set_ylabel('Fraction of events',ha='center', y=0.5, size = 12)
    #ax.grid(True, 'major', linestyle='solid', color='grey', alpha=0.5)

    name = 'plotting/plots/' + variable 
    print('Plotting leadJetPt plot')
    fig.savefig(name, dpi=1200)











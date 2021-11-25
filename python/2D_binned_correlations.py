import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob


parser = argparse.ArgumentParser()
required_args = parser.add_argument_group('Required Arguments')
required_args.add_argument('-m','--mc', action='store', help='Input MC dataframe dir', required=True)
required_args.add_argument('-d','--data', action='store', help='Input data dataframe dir', required=True)
options=parser.parse_args()

files_mc_csv = glob.glob("%s/*.csv"%options.mc)
files_mc_data = glob.glob("%s/*.csv"%options.data)

dataframes = []
for f in files_mc_csv:
  dataframes.append( pd.read_csv(f) )
  #print " --> Read: %s"%f
for f in files_mc_data:
  dataframes.append( pd.read_csv(f) )
  #print " --> Read: %s"%f

df = pd.concat( dataframes, sort=False, axis=0 )
#print " --> Successfully read dataframes"


# -------------------------------

#Adapted code for 2D correlation plots
#For issues with plotting variables that have -999 values can implement the following code logic
#Find max and min of both variable arrays
#If none have -999 then no changes
#If variables have -999 then xlim and ylim from -1 to max of respective variable

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
df['DR'] = np.sqrt(df['diphotonEta']**2 + df['diphotonPhi']**2)

# -------------------------------
list_variables1 = [
#'leadPhotonIDMVA','subleadPhotonIDMVA',
#'min_IDMVA','max_IDMVA',
#'diphotonMass','diphotonPt',
'leadPhotonPtOvM','subleadPhotonPtOvM',
'leadPhotonEta','subleadPhotonEta',
'dijetMass',
'dijetAbsDEta',
'dijetDPhi'
#'leadJetPt','leadJetEta','leadJetPhi',
#'subleadJetEta','subleadJetPhi',
#'subsubleadJetPt','subsubleadJetEn','subsubleadJetEta','subsubleadJetPhi'
]

list_variables2 = [
#'leadPhotonIDMVA','subleadPhotonIDMVA',
#'min_IDMVA','max_IDMVA',
#'diphotonMass','diphotonPt',
'leadPhotonPtOvM','subleadPhotonPtOvM',
'leadPhotonEta','subleadPhotonEta',
'dijetMass',
'dijetAbsDEta',
'dijetDPhi'
#'leadJetPt','leadJetEta','leadJetPhi',
#'subleadJetEn','subleadJetEta','subleadJetPhi',
#'subsubleadJetPt','subsubleadJetEn','subsubleadJetEta','subsubleadJetPhi'
]

#list_plots is a list that contains all the variable combinations that have been previously plotted
#Variable is used to avoid the same pair to be plotted twice
list_plots = []
lower_lim = -1000
upper_lim = 50000
num_bins = 100

for variable1 in list_variables1:
  # Signal
  vbf_sig_00 = np.array(df[df['proc'] == 'VBF'][variable1])
  vbf_sig_1_w = np.array(df[df['proc'] == 'VBF']['weight'])[(vbf_sig_00 > lower_lim) & (vbf_sig_00 <upper_lim)]
  vbf_sig_1_final = vbf_sig_00[(vbf_sig_00 > lower_lim) & (vbf_sig_00 < upper_lim)]
  vh_sig_00 = np.array(df[df['proc'] == 'VH'][variable1])
  vh_sig_1_w = np.array(df[df['proc'] == 'VH']['weight'])[(vh_sig_00 > lower_lim) & (vh_sig_00 <upper_lim)]
  vh_sig_1_final = vh_sig_00[(vh_sig_00 > lower_lim) & (vh_sig_00 <upper_lim)]
  ggh_sig_00 = np.array(df[df['proc'] == 'ggH'][variable1])
  ggh_sig_1_w = np.array(df[df['proc'] == 'ggH']['weight'])[(ggh_sig_00 > lower_lim) & (ggh_sig_00 <upper_lim)]
  ggh_sig_1_final = ggh_sig_00[(ggh_sig_00 > lower_lim) & (ggh_sig_00 <upper_lim)]
  tth_sig_00 = np.array(df[df['proc'] == 'ttH'][variable1])
  tth_sig_1_w = np.array(df[df['proc'] == 'ttH']['weight'])[(tth_sig_00 > lower_lim) & (tth_sig_00 <upper_lim)]
  tth_sig_1_final = tth_sig_00[(tth_sig_00 > lower_lim) & (tth_sig_00 <upper_lim)]
  combined_sig_1 = np.concatenate((vbf_sig_1_final,vh_sig_1_final,ggh_sig_1_final,tth_sig_1_final),axis=0)
  combined_sig_1_w = np.concatenate((vbf_sig_1_w,vh_sig_1_w,ggh_sig_1_w,tth_sig_1_w),axis=0)

  for variable2 in list_variables2:
    vbf_sig_01 = np.array(df[df['proc'] == 'VBF'][variable2])
    vbf_sig_2_w = np.array(df[df['proc'] == 'VBF']['weight'])[(vbf_sig_01 > lower_lim) & (vbf_sig_01 <upper_lim)]
    vbf_sig_2_final = vbf_sig_01[(vbf_sig_01 > lower_lim) & (vbf_sig_01 < upper_lim)]
    vh_sig_01 = np.array(df[df['proc'] == 'VH'][variable2])
    vh_sig_2_w = np.array(df[df['proc'] == 'VH']['weight'])[(vh_sig_01 > lower_lim) & (vh_sig_01 <upper_lim)]
    vh_sig_2_final = vh_sig_01[(vh_sig_01 > lower_lim) & (vh_sig_01 <upper_lim)]
    ggh_sig_01 = np.array(df[df['proc'] == 'ggH'][variable2])
    ggh_sig_2_w = np.array(df[df['proc'] == 'ggH']['weight'])[(ggh_sig_01 > lower_lim) & (ggh_sig_01 <upper_lim)]
    ggh_sig_2_final = ggh_sig_01[(ggh_sig_01 > lower_lim) & (ggh_sig_01 <upper_lim)]
    tth_sig_01 = np.array(df[df['proc'] == 'ttH'][variable2])
    tth_sig_2_w = np.array(df[df['proc'] == 'ttH']['weight'])[(tth_sig_01 > lower_lim) & (tth_sig_01 <upper_lim)]
    tth_sig_2_final = tth_sig_01[(tth_sig_01 > lower_lim) & (tth_sig_01 <upper_lim)]
    combined_sig_2 = np.concatenate((vbf_sig_2_final,vh_sig_2_final,ggh_sig_2_final,tth_sig_2_final),axis=0)
    combined_sig_2_w = np.concatenate((vbf_sig_2_w,vh_sig_2_w,ggh_sig_2_w,tth_sig_2_w),axis=0)
    """
    if variable1 != variable2:
      cur_var_pair = [variable1,variable2]
      if cur_var_pair not in list_plots:
        #Loop through the differnt signal modes to include all pairs
        fig, ax = plt.subplots(1)
        plt.scatter(vbf_sig_1,vbf_sig_2,s=1,c='#1f77b4',alpha=0.6,label='VBF')
        plt.scatter(vh_sig_1,vh_sig_2,s=1,c='#ff7f0e',alpha=0.2,label='VH')
        name = 'plotting/plots/Correlation_' + variable1 + '_' + variable2 + '_VBF_VH'
        plt.title('Correlation Plot VBF and VH')
        plt.legend()
        plt.xlabel(variable1)
        plt.ylabel(variable2)
        fig.savefig(name)

        fig, ax = plt.subplots(1)
        plt.scatter(vbf_sig_1,vbf_sig_2,s=1,c='#1f77b4',alpha=0.6,label='VBF')
        plt.scatter(ggh_sig_1,ggh_sig_2,s=1,c='#ff7f0e',alpha=0.2,label='ggH')
        name = 'plotting/plots/Correlation_' + variable1 + '_' + variable2 + '_VBF_ggH'
        plt.title('Correlation Plot VBF and ggH')
        plt.legend()
        plt.xlabel(variable1)
        plt.ylabel(variable2)
        fig.savefig(name)

        fig, ax = plt.subplots(1)
        plt.scatter(vbf_sig_1,vbf_sig_2,s=1,c='#1f77b4',alpha=0.6,label='VBF')
        plt.scatter(tth_sig_1,tth_sig_2,s=1,c='#ff7f0e',alpha=0.2,label='ttH')
        name = 'plotting/plots/Correlation_' + variable1 + '_' + variable2 + '_VBF_ttH'
        plt.title('Correlation Plot VBF and ttH')
        plt.legend()
        plt.xlabel(variable1)
        plt.ylabel(variable2)
        fig.savefig(name)

        fig, ax = plt.subplots(1)
        plt.scatter(vh_sig_1,vh_sig_2,s=1,c='#1f77b4',alpha=0.6,label='VH')
        plt.scatter(ggh_sig_1,ggh_sig_2,s=1,c='#ff7f0e',alpha=0.2,label='ggH')
        name = 'plotting/plots/Correlation_' + variable1 + '_' + variable2 + '_VH_ggH'
        plt.title('Correlation Plot VH and ggH')
        plt.legend()
        plt.xlabel(variable1)
        plt.ylabel(variable2)
        fig.savefig(name)

        fig, ax = plt.subplots(1)
        plt.scatter(vh_sig_1,vh_sig_2,s=1,c='#1f77b4',alpha=0.6,label='VH')
        plt.scatter(tth_sig_1,tth_sig_2,s=1,c='#ff7f0e',alpha=0.2,label='ttH')
        name = 'plotting/plots/Correlation_' + variable1 + '_' + variable2 + '_VH_ttH'
        plt.title('Correlation Plot VH and ttH')
        plt.legend()
        plt.xlabel(variable1)
        plt.ylabel(variable2)
        fig.savefig(name)

        fig, ax = plt.subplots(1)
        plt.scatter(ggh_sig_1,ggh_sig_2,s=1,c='#1f77b4',alpha=0.6,label='ggH')
        plt.scatter(tth_sig_1,tth_sig_2,s=1,c='#ff7f0e',alpha=0.2,label='ttH')
        name = 'plotting/plots/Correlation_' + variable1 + '_' + variable2 + 'ggH_ttH'
        plt.title('Correlation Plot ggH and ttH')
        plt.legend()
        plt.xlabel(variable1)
        plt.ylabel(variable2)
        fig.savefig(name)

      list_plots.append([variable1,variable2])
      list_plots.append([variable2,variable1])
    """
    # new version?
    if variable1 != variable2:
      # Okay, this is about to get super ugly
      # VBF
      # removing the points where vbf_sig_1 is -999
      vbf_sig_1_a = vbf_sig_1_final[vbf_sig_1_final != -999]
      #print(vbf_sig_1.shape)
      #print('hello', vbf_sig_1_a.shape)
      #print('world', vbf_sig_2.shape)
      vbf_sig_2_a = vbf_sig_2_final[vbf_sig_1_final != -999]
      # removing the points where vbf_sig_2 is -999
      vbf_sig_1_b = vbf_sig_1_a[vbf_sig_2_a != -999]
      vbf_sig_2_b = vbf_sig_2_a[vbf_sig_2_a != -999]
      # removing the points where vbf_sig_1 is -1
      vbf_sig_1_c = vbf_sig_1_b[vbf_sig_1_b != -1]
      vbf_sig_2_c = vbf_sig_2_b[vbf_sig_1_b != -1]
      # removing the points where vbf_sig_2 is -1
      vbf_sig_1 = vbf_sig_1_c[vbf_sig_2_c != -1]
      vbf_sig_2 = vbf_sig_2_c[vbf_sig_2_c != -1]
      # VH
      # removing the points where vh_sig_1 is -999
      vh_sig_1_a = vh_sig_1_final[vh_sig_1_final != -999] 
      vh_sig_2_a = vh_sig_2_final[vh_sig_1_final != -999]
      # removing the points where vh_sig_2 is -999
      vh_sig_1_b = vh_sig_1_a[vh_sig_2_a != -999]
      vh_sig_2_b = vh_sig_2_a[vh_sig_2_a != -999]
      # removing the points where vh_sig_1 is -1
      vh_sig_1_c = vh_sig_1_b[vh_sig_1_b != -1]
      vh_sig_2_c = vh_sig_2_b[vh_sig_1_b != -1]
      # removing the points where vh_sig_2 is -1
      vh_sig_1 = vh_sig_1_c[vh_sig_2_c != -1]
      vh_sig_2 = vh_sig_2_c[vh_sig_2_c != -1]
      # ggH
      # removing the points where ggh_sig_1 is -999
      ggh_sig_1_a = ggh_sig_1_final[ggh_sig_1_final != -999] 
      ggh_sig_2_a = ggh_sig_2_final[ggh_sig_1_final != -999]
      # removing the points where ggh_sig_2 is -999
      ggh_sig_1_b = ggh_sig_1_a[ggh_sig_2_a != -999]
      ggh_sig_2_b = ggh_sig_2_a[ggh_sig_2_a != -999]
      # removing the points where ggh_sig_1 is -1
      ggh_sig_1_c = ggh_sig_1_b[ggh_sig_1_b != -1]
      ggh_sig_2_c = ggh_sig_2_b[ggh_sig_1_b != -1]
      # removing the points where ggh_sig_2 is -1
      ggh_sig_1 = ggh_sig_1_c[ggh_sig_2_c != -1]
      ggh_sig_2 = ggh_sig_2_c[ggh_sig_2_c != -1]
      # ttH
      # removing the points where tth_sig_1 is -999
      tth_sig_1_a = tth_sig_1_final[tth_sig_1_final != -999] 
      tth_sig_2_a = tth_sig_2_final[tth_sig_1_final != -999]
      # removing the points where tth_sig_2 is -999
      tth_sig_1_b = tth_sig_1_a[tth_sig_2_a != -999]
      tth_sig_2_b = tth_sig_2_a[tth_sig_2_a != -999]
      # removing the points where tth_sig_1 is -1
      tth_sig_1_c = tth_sig_1_b[tth_sig_1_b != -1]
      tth_sig_2_c = tth_sig_2_b[tth_sig_1_b != -1]
      # removing the points where tth_sig_2 is -1
      tth_sig_1 = tth_sig_1_c[tth_sig_2_c != -1]
      tth_sig_2 = tth_sig_2_c[tth_sig_2_c != -1]

      cur_var_pair = [variable1,variable2]
      if cur_var_pair not in list_plots:
        #Plot all 4 signal modes
        fig, ax = plt.subplots(2,2,sharex=True,sharey=True)
        ax[0][0].hist2d(vbf_sig_1,vbf_sig_2,bins=num_bins,cmap=plt.cm.jet,label='VBF')
        ax[0][0].set_title('VBF')
        ax[0][0].set(ylabel=variable2)
        ax[0][1].hist2d(vh_sig_1,vh_sig_2,bins=num_bins,cmap=plt.cm.jet,label='VH')
        ax[0][1].set_title('VH')
        ax[1][0].hist2d(ggh_sig_1,ggh_sig_2,bins=num_bins,cmap=plt.cm.jet,label='ggH')
        ax[1][0].set_title('ggH')
        ax[1][0].set(xlabel=variable1)
        ax[1][0].set(ylabel=variable2)
        ax[1][1].hist2d(tth_sig_1,tth_sig_2,bins=num_bins,cmap=plt.cm.jet,label='ttH')
        ax[1][1].set_title('ttH')
        ax[1][1].set(xlabel=variable1)
        plt.suptitle('Correlation Plot {}'.format(variable1 + variable2))
        #For some adding the colorbar doesn't work :(
        #plt.colorbar()
        name = 'plotting/plots/Correlation/' + variable1 + '_' + variable2
        fig.savefig(name)

      list_plots.append([variable1,variable2])
      list_plots.append([variable2,variable1])
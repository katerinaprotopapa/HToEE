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
#print " --> Successfully read dataframes. Printing first five events"
#print df.head()
#print 'HUUUUH'



# -------------------------------
# Background Grouping
# QCD
df['proc'] = np.where(df['proc'] == 'QCD_30to40', 'QCD', df['proc'])
df['proc'] = np.where(df['proc'] == 'QCD_30toInf', 'QCD', df['proc'])
df['proc'] = np.where(df['proc'] == 'QCD_40toInf', 'QCD', df['proc'])
df['proc'] = np.where(df['proc'] == 'QCD30to40', 'QCD', df['proc'])
df['proc'] = np.where(df['proc'] == 'QCD30toInf', 'QCD', df['proc'])
df['proc'] = np.where(df['proc'] == 'QCD40toInf', 'QCD', df['proc'])

# Gjet
df['proc'] = np.where(df['proc'] == 'Gjet_20to40', 'Gjet', df['proc'])
df['proc'] = np.where(df['proc'] == 'Gjet_20toInf', 'Gjet', df['proc'])
df['proc'] = np.where(df['proc'] == 'Gjet_40toInf', 'Gjet', df['proc'])
df['proc'] = np.where(df['proc'] == 'Gjet20to40', 'Gjet', df['proc'])
df['proc'] = np.where(df['proc'] == 'Gjet20toInf', 'Gjet', df['proc'])
df['proc'] = np.where(df['proc'] == 'Gjet40toInf', 'Gjet', df['proc'])

# Diphoton
df['proc'] = np.where(df['proc'] == 'Diphoton40to80', 'Diphoton', df['proc'])
df['proc'] = np.where(df['proc'] == 'Diphoton80toInf', 'Diphoton', df['proc'])
df['proc'] = np.where(df['proc'] == 'Diphoton_40to80', 'Diphoton', df['proc'])
df['proc'] = np.where(df['proc'] == 'Diphoton_80toInf', 'Diphoton', df['proc'])



# -------------------------------

# New columns
#df['difference_lead_sublead'] = df['leadPhotonIDMVA'] - df['subleadPhotonIDMVA']
#df['diff_lead_sublead_bool'] = np.where(df['difference_lead_sublead'] > 0, True, False)

# min_IDMVA: gives evidence of mostly being a jet
df['min_IDMVA'] = df[['leadPhotonIDMVA', 'subleadPhotonIDMVA']].min(axis=1)
# max_IDMVA: evidence of mostly having a photon
df['max_IDMVA'] = df[['leadPhotonIDMVA', 'subleadPhotonIDMVA']].max(axis=1)
# DeltaR
df['DR'] = np.sqrt(df['diphotonEta']**2 + df['diphotonPhi']**2)


list_variables = [#'diphotonPt'
                #, 'leadPhotonEta'
                #, 'subleadPhotonEta' 
                #, 'leadPhotonIDMVA' 
                #, 'subleadPhotonIDMVA'
                #, 'diphotonMass' 
                #, 'weight' 
                #, 'centralObjectWeight' 
                #, 'leadPhotonPtOvM'
                 'dijetMass' 
                #, 'subleadPhotonPtOvM'
                #, 'leadJetPt'
                #, 'subleadJetPt' #'leadElectronIDMVA', 'subleadElectronIDMVA',
                #, 'dijetMass'
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
                # 'dijetCentrality'
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
]

for variable in list_variables:

  # dealing with the NaNs in the dataframe
  #df[variable].fillna(-999, inplace=True)
  
  # Splitting into the different parts
  
  # Signal
  
  vbf_sig_0 = np.array(df[df['proc'] == 'VBF'][variable])
  vbf_sig_w = np.array(df[df['proc'] == 'VBF']['weight'])[(vbf_sig_0 > -10) & (vbf_sig_0 <2000)]
  vbf_sig = vbf_sig_0[(vbf_sig_0 > -10) & (vbf_sig_0 < 2000)]
  vh_sig_0 = np.array(df[df['proc'] == 'VH'][variable])
  vh_sig_w = np.array(df[df['proc'] == 'VH']['weight'])[(vh_sig_0 > -10) & (vh_sig_0 <2000)]
  vh_sig = vh_sig_0[(vh_sig_0 > -10) & (vh_sig_0 <2000)]
  ggh_sig_0 = np.array(df[df['proc'] == 'ggH'][variable])
  ggh_sig_w = np.array(df[df['proc'] == 'ggH']['weight'])[(ggh_sig_0 > -10) & (ggh_sig_0 <2000)]
  ggh_sig = ggh_sig_0[(ggh_sig_0 > -10) & (ggh_sig_0 <2000)]
  tth_sig_0 = np.array(df[df['proc'] == 'ttH'][variable])
  tth_sig_w = np.array(df[df['proc'] == 'ttH']['weight'])[(tth_sig_0 > -10) & (tth_sig_0 <2000)]
  tth_sig = tth_sig_0[(tth_sig_0 > -10) & (tth_sig_0 <2000)]
  combined_sig = np.concatenate((vbf_sig,vh_sig,ggh_sig,tth_sig),axis=0)
  combined_sig_w = np.concatenate((vbf_sig_w,vh_sig_w,ggh_sig_w,tth_sig_w),axis=0)
  combined_sig_exc_vbf = np.concatenate((vh_sig,ggh_sig,tth_sig),axis=0)
  combined_sig_w_exc_vbf = np.concatenate((vh_sig_w,ggh_sig_w,tth_sig_w),axis=0)
  combined_sig_exc_tth = np.concatenate((vbf_sig,vh_sig,ggh_sig),axis=0)
  combined_sig_w_exc_tth = np.concatenate((vbf_sig_w,vh_sig_w,ggh_sig_w),axis=0)

  # Background
  qcd_0 = np.array(df[df['proc'] == 'QCD'][variable])
  qcd_w = np.array(df[df['proc'] == 'QCD']['weight'])[(qcd_0 > -10) & (qcd_0 <300)]
  qcd = qcd_0[(qcd_0 > -10) & (qcd_0 <300)]
  gjet_0 = np.array(df[df['proc'] == 'Gjet'][variable])
  gjet_w = np.array(df[df['proc'] == 'Gjet']['weight'])[(gjet_0 > -10) & (gjet_0 <300)]
  gjet = gjet_0[(gjet_0 > -10) & (gjet_0 <300)]
  diphoton_0 = np.array(df[df['proc'] == 'Diphoton'][variable])
  diphoton_w = np.array(df[df['proc'] == 'Diphoton']['weight'])[(diphoton_0 > -10) & (diphoton_0 <300)]
  diphoton = diphoton_0[(diphoton_0 > -10) & (diphoton_0 <300)]
  combined_bkg = np.concatenate((qcd,gjet,diphoton),axis=0)
  combined_bkg_w = np.concatenate((qcd_w,gjet_w,diphoton_w),axis=0)

  # Data
  data = np.array(df[df['proc'] == 'Data'][variable])
  data_w = np.array(df[df['proc'] == 'Data']['weight'])

  # Now let's plot the histogram

  scale = 100
  num_bins = 40
  normalize = True

  fig, ax = plt.subplots(1)

   # background
  #plt.hist(qcd, bins = num_bins, density = normalize, color = 'lightgrey', label = 'QCD background', weights = qcd_w, histtype = 'step')
  #plt.hist(gjet, bins = num_bins, density = normalize, color = 'lightgreen', label = 'Gjet background', histtype = 'step', weights = gjet_w)
  #plt.hist(diphoton, bins = num_bins, density = normalize, color = 'lightskyblue', label = 'Diphoton background', histtype = 'step', weights = diphoton_w)
  #plt.hist(combined_bkg, bins = num_bins, density = normalize, color = 'lightskyblue', label = 'Background',  histtype = 'step', weights = combined_bkg_w)
  #area = sum(np.diff(bins)*values)
  #print("Bckg Area", area)

  # signal
  plt.hist(vbf_sig, bins = num_bins, density = normalize, color = 'orange', label = 'VBF', histtype = 'step', weights = scale * vbf_sig_w)
  #plt.hist(vh_sig, bins = num_bins, density = normalize, color = 'limegreen', label = 'VH', stacked = True, histtype = 'step', weights = scale * vh_sig_w)
  plt.hist(ggh_sig, bins = num_bins, density = normalize, color = 'deepskyblue', label = 'ggH', stacked = True, histtype = 'step', weights = scale * ggh_sig_w)
  #plt.hist(tth_sig, bins = num_bins, density = normalize, color = 'orchid', label = 'ttH', stacked = True, histtype = 'step', weights = scale * tth_sig_w)
  #plt.hist(combined_sig, bins = num_bins, density = normalize, color = 'blue', label = 'Signal',  histtype = 'step', weights = scale * combined_sig_w)
  #plt.hist(combined_sig_exc_tth, bins = num_bins, density = normalize, color = 'blue', label = 'VH, tth, ggh',  histtype = 'step', weights = scale * combined_sig_w_exc_tth)

  # data
  #plt.hist(data, bins = num_bins, density = normalize, color = 'black', label = 'Data', histtype = 'step', weights = data_w)

  plt.legend()
  plt.xlabel(variable)
  plt.ylabel('Events')
  if variable == 'diphotonMass':
    plt.xlim(100,150)
  #if variable == 'dijetMass':
  #  plt.xlim(0,300)
  elif variable == 'diphotonPt':
    plt.xlim(0,200)
  elif variable == 'leadPhotonIDMVA':
    plt.xlim(-1,1)
    plt.legend(loc='upper left')
  elif variable == 'subleadPhotonIDMVA':
    plt.xlim(-1,1)
    plt.legend(loc='upper left') 
  elif variable == 'min_IDMVA':
    plt.xlim(-1,1) 
  elif variable == 'max_IDMVA':
    plt.xlim(-1,1) 
    plt.legend(loc='upper left') 

  name = 'plotting/plots/' + variable 
  fig.savefig(name, dpi=300)








# ---------------------------------------------------


"""

# TRYING STUFF
  #counts = df['proc'].value_counts(dropna=False)
  #x= 0
  #plt.hist(df.replace(np.nan, x))


  #df['my_channel'] = np.where(df.my_channel > 20000, 0, df.my_channel)
  #qcd = np.nan_to_num(qcd)
  #gjet = np.nan_to_num(gjet)
  #diphoton = np.nan_to_num(diphoton)
  #qcd[np.isnan(qcd)] = 0
  #qcd[qcd == None] = 0
  #np.where(qcd == None, 10, qcd)
  #qcd[np.isnan(qcd)] = 10
  #qcd.replace(None, 10)
"""



"""print df.groupby('proc').size()
fig, ax = plt.subplots()
df['diphotonPt'].groupby(df['proc']).plot(kind = 'hist', legend = True)
fig.savefig('plotting/plots/signal_trial1')"""





"""
#get xrange from yaml config
        with open('plotting/var_to_xrange.yaml', 'r') as plot_config_file:
            plot_config   = yaml.load(plot_config_file)
            var_to_xrange = plot_config['var_to_xrange']
n_bins = 26 # just by default
bins = np.linspace(var_to_xrange[var][0], var_to_xrange[var][1], n_bins)

sig_labels   = np.unique(sig_df['proc'].values).tolist()
sig_scaler   = 5*10**2
def num_to_str(num):
        ''' 
        Convert basic number into scientific form e.g. 1000 -> 10^{3}.
        Not considering decimal inputs (see decimal_to_str). Also ignores first unit.
        '''
        str_rep = str(num) 
        if str_rep[0] == 0: return num 
        exponent = len(str_rep)-1
        return r'$\times 10^{%s}$'%(exponent)

sig_weights = self.sig_df['weight'].values


axes.hist(var_sig, bins=bins, label=sig_labels[0]+r' ($\mathrm{H}\rightarrow\mathrm{Gamma Gamma}$) '+num_to_str(sig_scaler), weights=sig_weights*(sig_scaler), histtype='step', color='forestgreen', zorder=10)


#axes.hist(sig_scores, bins=bins, label=self.sig_labels[0]+r' ($\mathrm{H}\rightarrow\mathrm{ee}$) '+self.num_to_str(self.sig_scaler), weights=sig_w_true*(self.sig_scaler), histtype='step', color=self.sig_colour)

#ax = vbf_sig.hist()
#fig = ax[0].get_figure()
# try one of the following
#fig = ax[0].get_figure()
#fig = ax[0][0].get_figure()

#fig.savefig('VBF_hist_trial1.pdf')

"""


#imports
import numpy as np
import pandas as pd

from NNs_class import NN_binary

#Define key quantities, use to tune NN
num_epochs = 4
batch_size = 400
test_split = 0.30
val_split = 0
learning_rate = 0.01 #0.001

epochs = np.linspace(1,num_epochs,num_epochs,endpoint=True).astype(int) #For plotting
binNames = ['ggH','VBF'] 
bins = 50

#Directories
modelDir = 'neural_networks/models/'
plotDir  = 'neural_networks/plots/'

#Define the variables, this is an arbitrary selection
# Philipp
"""
 train_vars = ['leadPhotonIDMVA','subleadPhotonIDMVA',
    'diphotonMass','diphotonPt',
    'leadPhotonPtOvM','subleadPhotonPtOvM',
    'leadPhotonEta','subleadPhotonEta',
    'dijetMass','dijetAbsDEta','dijetDPhi','diphotonCosPhi',
    'leadJetPUJID','subleadJetPUJID','subsubleadJetPUJID',
    'leadJetPt','leadJetEn','leadJetEta','leadJetPhi',
    'subleadJetPt','subleadJetEn','subleadJetEta','subleadJetPhi',
    'subsubleadJetPt','subsubleadJetEn','subsubleadJetEta','subsubleadJetPhi'
    ]
    """
# Kat
train_vars = ['diphotonPt', 'diphotonMass', 'diphotonCosPhi', 'diphotonEta','diphotonPhi', 'diphotonSigmaMoM',
     'dijetMass', 'dijetAbsDEta', 'dijetDPhi', 'dijetCentrality',
     'dijetPt','dijetEta','dijetPhi','dijetMinDRJetPho','dijetDiphoAbsDEta',
     'leadPhotonEta', 'leadPhotonPtOvM', 'leadPhotonEn', 'leadPhotonPt', 'leadPhotonPhi',
     'leadJetPt', 'leadJetPUJID', 'leadJetBTagScore', 'leadJetMass',
     'leadJetDiphoDEta','leadJetDiphoDPhi','leadJetEn','leadJetEta','leadJetPhi',
     'subleadPhotonEta', 'subleadPhotonPtOvM', 'subleadPhotonIDMVA', 'subleadPhotonPhi',
     'subleadPhotonEn','subleadPhotonPt', 
     'subleadJetDiphoDPhi','subleadJetDiphoDEta',
     'subleadJetPt', 'subleadJetPUJID', 'subleadJetBTagScore', 'subleadJetMass',
     'subleadJetEn','subleadJetEta','subleadJetPhi',
     'subsubleadJetEn','subsubleadJetPt','subsubleadJetEta','subsubleadJetPhi', 'subsubleadJetBTagScore', 
     'subsubleadJetMass',
     'metPt','metPhi','metSumET',
     'nSoftJets'
     ]

#Add proc and weight to shuffle with data 
train_vars.append('proc') 
train_vars.append('weight') 

#Load the dataframe
dataframes = []
dataframes.append(pd.read_csv('2017/MC/DataFrames/VBF_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/ggH_VBF_BDT_df_2017.csv'))
df = pd.concat(dataframes, sort=False, axis=0 )

#dataframe of train_vars
data = df[train_vars]

# call the class
class_NN = NN_binary(data, lr = learning_rate, test_split = test_split, val_split = val_split, batch_size = batch_size, num_epochs = num_epochs)
#class_NN.run(roc_curve_bool = True)
class_NN.nodes_analysis(num_nodes = 2, increase_nodes = 10)




























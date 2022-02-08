#imports
import numpy as np
import pandas as pd

from NNs_class import NN_binary

#Define key quantities, use to tune NN
num_epochs = 50
nodes = 120
batch_size = 64
test_split = 0.15
val_split = 0.10
learning_rate = 0.001 #0.001
activation = 'relu'
num_int_layers = 1

# For HP analysis
activation_list = ['relu', 'sigmoid', 'softmax', 'tanh', 'selu', 'elu']
num_intermediate_layers = 7

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
train_vars.append('HTXS_stage_0')
#train_vars.append('HTXS_stage1_2_cat_pTjet30GeV')

#Load the dataframe
dataframes = []
dataframes.append(pd.read_csv('2017/MC/DataFrames/VBF_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/ggH_VBF_BDT_df_2017.csv'))
df = pd.concat(dataframes, sort=False, axis=0 )

#dataframe of train_vars
data = df[train_vars]

# call the class
#class_NN = NN_binary(data, nodes = nodes, lr = learning_rate, test_split = test_split, val_split = val_split, batch_size = batch_size, activation = activation, num_layers = num_int_layers, num_epochs = num_epochs)   
#auc_test = class_NN.run(output_score = True, roc_curve_bool = True)

# optimization analysis
#class_NN.nodes_analysis(num_nodes = 10, increase_nodes = 5)
#class_NN.activation_analysis(activation_list)
#class_NN.layers_analysis(num_intermediate_layers)

#exit(0)

# Optimization - Grid approach
nodes = [50, 120, 150, 200]
layers = [1,2,4,10]
learn_rate = [0.1, 0.01, 0.001, 0.0001]
activation_value = 'relu'
#activation = ['relu', 'sigmoid', 'softmax']

param_comb = []
roc_values = []

for nodes_value in nodes:
     for layers_value in layers:
          for lr_value in learn_rate:
               class_NN = NN_binary(data, nodes = nodes_value, lr = lr_value, test_split = test_split, val_split = val_split, batch_size = batch_size, num_epochs = num_epochs, activation = activation_value, num_layers = layers_value)  
               auc_test = class_NN.run(output_score = True, roc_curve_bool = True)
               param_comb.append([nodes_value,layers_value,lr_value])
               roc_values.append(auc_test)
               print('Parameter combination: ', nodes_value,layers_value,lr_value)
               print('AUC ROC score: ', auc_test)

max_value = np.max(roc_values)
index_max_value = roc_values.index(max_value)
best_param_comb = param_comb[index_max_value]
print('Best parameter combination:',best_param_comb)
print('Best ROC Score:',max_value)





























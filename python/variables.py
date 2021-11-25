#photon vars
nominal_vars = ['diphotonPt','leadPhotonEta', 'subleadPhotonEta', 'leadPhotonIDMVA', 'subleadPhotonIDMVA', 'diphotonMass', 'weight', 'centralObjectWeight', 'leadPhotonPtOvM', 'dijetMass', 'subleadPhotonPtOvM','leadJetPt','subleadJetPt' #'leadElectronIDMVA', 'subleadElectronIDMVA',
                , 'dijetMass', 'dijetAbsDEta', 'dijetDPhi', 'diphotonCosPhi'
                , 'leadJetPUJID', 'subleadJetPUJID'
                ,'subsubleadJetEn','subsubleadJetPt','subsubleadJetEta','subsubleadJetPhi'
                , 'dijetCentrality', 'leadJetBTagScore', 'subleadJetBTagScore', 'subsubleadJetBTagScore'
                , 'leadJetMass', 'leadPhotonEn', 'leadPhotonMass', 'leadPhotonPt'
                , 'subleadJetMass', 'subleadPhotonEn', 'subleadPhotonMass', 'subleadPhotonPt'
                , 'diphotonEta', 'diphotonPhi',
                'leadPhotonPhi','leadJetDiphoDEta','leadJetDiphoDPhi','leadJetEn','leadJetEta','leadJetPhi',
                'subleadPhotonPhi','subsubleadPhotonEn','subsubleadJetMass','subsubleadPhotonMass',
                'subsubleadPhotonPt','subsubleadPhotonEta','subsubleadPhotonPhi','subleadJetDiphoDPhi',
                'subleadJetDiphoDEta','subleadJetEn','subleadJetEta','subleadJetPhi','subsubleadPhotonIDMVA',				
                'diphotonEta','diphotonPhi','dijetPt','dijetEta','dijetPhi','dijetMinDRJetPho','dijetDiphoAbsDEta',
                'nSoftJets',
                'leadTrkEn','leadTrkMass','leadTrkPt','leadTrkEta','leadTrkPhi','leadPhotonSigmaE',
                'leadPhotonHoE','leadPhotonPfRelIsoAll','leadPhotonPfRelIsoChg','leadPhotonR9','leadPhotonSieie',
                'leadPhotonElectronVeto','leadPhotonPixelSeed','leadPhotonElectronIdx','leadPhotonJetIdx','leadJetQGL',
                'leadJetID','leadTrkDxy','leadTrkDz','leadTrkPfRelIso03All','leadTrkPfRelIso03Chg','leadTrkMiniPFRelIsoAll',
                'leadTrkMiniPFRelIsoChg','leadTrkFromPV','leadTrkIsHighPurityTrack','leadTrkIsPFcand','leadTrkIsFromLostTrack',
                'subleadTrkEn','subleadTrkMass','subleadTrkPt','subleadTrkEta','subleadTrkPhi','subleadPhotonSigmaE','subleadPhotonHoE',
                'subleadPhotonPfRelIsoAll','subleadPhotonPfRelIsoChg','subleadPhotonR9','subleadPhotonSieie','subleadPhotonElectronVeto',
                'subleadPhotonPixelSeed','subleadPhotonElectronIdx','subleadPhotonJetIdx','subleadJetQGL','subleadJetID',
                'subleadTrkDxy','subleadTrkDz','subleadTrkPfRelIso03All','subleadTrkPfRelIso03Chg','subleadTrkMiniPFRelIsoAll',
                'subleadTrkMiniPFRelIsoChg','subleadTrkFromPV','subleadTrkIsHighPurityTrack','subleadTrkIsPFcand','subleadTrkIsFromLostTrack',
                'subsubleadTrkEn','subsubleadTrkMass','subsubleadTrkPt','subsubleadTrkEta','subsubleadTrkPhi','subsubleadPhotonSigmaE',
                'subsubleadPhotonHoE','subsubleadPhotonPfRelIsoAll','subsubleadPhotonPfRelIsoChg','subsubleadPhotonR9','subsubleadPhotonSieie',
                'subsubleadPhotonElectronVeto','subsubleadPhotonPixelSeed','subsubleadPhotonElectronIdx','subsubleadPhotonJetIdx',
                'subsubleadJetQGL','subsubleadJetID','subsubleadJetPUJID',#'subsubleadTrkDx',
                'subsubleadTrkDz','subsubleadTrkPfRelIso03All',
                'subsubleadTrkPfRelIso03Chg','subsubleadTrkMiniPFRelIsoAll','subsubleadTrkMiniPFRelIsoChg','subsubleadTrkFromPV',
                'subsubleadTrkIsHighPurityTrack','subsubleadTrkIsPFcand','subsubleadTrkIsFromLostTrack','diphotonSigmaMoM','dijetDiphoAbsDPhiTrunc',
                'metPt','metPhi','metSumET','metSignificance',#'PhotonIDSF','TriggerSF'
                # 'leadJetPt', 'subleadJetPt',
                #'leadElectronEta', #'leadElectronEn', 'leadElectronPt', 'leadElectronPhi', 'leadElectronMass',
                #'subleadElectronEta',#'subleadElectronPt', 'subleadElectronEn',  'subleadElectronPhi', 'subleadElectronMass',
                #'subsubleadElectronEta',#'subsubleadElectronPt', 'subsubleadElectronEn', 'subsubleadElectronPhi', 'subsubleadElectronMass',
                #'dielectronCosPhi','dielectronPt', 'dielectronMass', 'leadJetPt','subleadJetPt',#'leadJetMass', 'subleadJetMass', 
                #'leadJetEn', 'leadJetPt','leadJetEta', 'leadJetPhi','leadJetQGL', #add jet en
                #'subleadJetEn', 'subleadJetPt','subleadJetEta', 'subleadJetPhi', 'subleadJetQGL', #add sublead jet en
                #'subsubleadJetEn','subsubleadJetPt', 'subsubleadJetEta', 'subsubleadJetPhi', 'subsubleadJetQGL', #'subsubleadJetMass' add subsublead jet en
                #'dijetAbsDEta', 'dijetMass', 'dijetDieleAbsDEta', 'dijetDieleAbsDPhiTrunc', 
                #'dijetMinDRJetEle', 'dijetCentrality', 'dielectronSigmaMoM', 'dijetDPhi', #'dijetPt',
                #'leadJetDieleDPhi', 'leadJetDieleDEta', 'subleadJetDieleDPhi', 'subleadJetDieleDEta',
                #'leadElectronCharge', 'subleadElectronCharge',
                #'nSoftJets','metSumET','metPhi','metPt' , 'leadJetBTagScore', 'subleadJetBTagScore', 'subsubleadJetBTagScore',
                #'leadJetPUJID','subleadJetPUJID', 'subsubleadJetPUJID'#,'leadJetID','subleadJetID','subsubleadJetID'
               ]

#for MVA training, hence not including masses
gev_vars     = ['leadJetEn', 'leadJetPt', 'subleadJetEn', 'subleadJetPt', 'subsubleadJetEn', 'subsubleadJetPt', 
                'leadElectronEn', 'leadElectronPt', 'subleadElectronEn', 'subleadElectronPt',
                'leadElectronPToM', 'subleadElectronPToM', 'dijetMass', 'dielectronPt'
               ]

gen_vars     = ['genWeight'] 


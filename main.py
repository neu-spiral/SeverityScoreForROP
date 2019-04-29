"""
This script is to generate the file that compute the performance of combining absolute labels and comparison labels.
"""

import argparse
from scipy.io import loadmat, savemat
from cvxOpt import Log_Log, SVM_Log, Logistic
from cvxpyMdl import SVM_SVM, Log_SVM, SVM
from modelCVFunc import CVGlobalModel, CVExpertBiasModel, CVExpertModel
import sys
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combining absolute labels and comparison labels on multi-expert data',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--partition',default='../data/ropData/iROP_6DD_1st100_Partition.mat',help='the partition file contains the feature, label and cross-validation partition.')
    parser.add_argument('--alpha', default=1.0, type=float, help='The balance parameter in [0,1] controls weight on absolute labels. 1 means only using absolute labels and 0 means only using comparison labels.' )
    parser.add_argument('--lam', default=1.0, type=float, help='The regularization parameter for lasso loss. High value of lambda would show more sparsity.')

    segmentTypeGroup = parser.add_mutually_exclusive_group(required = True)
    segmentTypeGroup.add_argument('--segment',choices=['manual','auto', 'CNN'])
    
    expertModelGroup = parser.add_mutually_exclusive_group(required = True)
    expertModelGroup.add_argument('--expert',choices=['global','bias','expert','all'],default='all')
    
    lossFuncGroup = parser.add_mutually_exclusive_group(required=True)
    lossFuncGroup.add_argument('--loss',choices=['LogLog','LogSVM','SVMLog','SVMSVM','Boost'],default='LogLog')
    
    args = parser.parse_args()

    nameBase = '../result/rop/'+args.loss+'/MS_'+ args.loss +'_L1'+ args.expert + '_' + args.segment
    dataFile = loadmat(args.partition)
    labelPlusSet = dataFile['labelPlus']
    labelPrePSet = dataFile['labelPreP']
    labelCmp = dataFile['cmpLabel']
    repeatTimes = int(dataFile['repeatTimes'][0, :])
    K = int(dataFile['numOfFolds'][0, :])
    indexLblSinTrainPlus = dataFile['RSDTrainPlusPartition']
    indexLblSinTestPlus = dataFile['RSDTestPlusPartition']
    indexCmpSinTrainPlus = dataFile['cmpTrainPlusPartition']
    indexCmpSinTestPlus = dataFile['cmpTestPlusPartition']
    indexLblSinTrainPreP = dataFile['RSDTrainPrePPartition']
    indexLblSinTestPreP = dataFile['RSDTestPrePPartition']
    indexCmpSinTrainPreP = dataFile['cmpTrainPrePPartition']
    indexCmpSinTestPreP = dataFile['cmpTestPrePPartition']
    if args.segment == 'manual':
        featLblSin = dataFile['labelFeatManual']
        featCmpSin = dataFile['cmpFeatManual']
    elif args.segment == 'auto':
        featLblSin = dataFile['labelFeatAuto']
        featCmpSin = dataFile['cmpFeatAuto']
    elif args.segment == 'CNN':
        # cnn_file = pickle.load(open('../data/ropData/featuresOf100PredictedDiscCenters(DuplicatesRemoved)_ordered.p','rb'))
        cnn_file = pickle.load(open('../data/ropData/complexity.p'))
        featLblSin = cnn_file['class_feat']
        featCmpSin = cnn_file['cmp_feat']
    else:
        sys.exit('args.segment should be manual or auto')

    dataPlus = {'featLblSin': featLblSin, 'featCmpSin': featCmpSin,
                'labelRSD': labelPlusSet[:, -1], 'labelAbs': labelPlusSet[:, 0:-1], 'labelCmp': labelCmp}
    indexCVPlus = {'indexLblSinTrain': indexLblSinTrainPlus, 'indexLblSinTest': indexLblSinTestPlus,
                   'indexCmpSinTrain': indexCmpSinTrainPlus, 'indexCmpSinTest': indexCmpSinTestPlus,
                   'repeatTimes': repeatTimes, 'K': K}

    dataPreP = {'featLblSin': featLblSin, 'featCmpSin': featCmpSin,
                'labelRSD': labelPrePSet[:, -1], 'labelAbs': labelPrePSet[:, 0:-1], 'labelCmp': labelCmp}
    indexCVPreP = {'indexLblSinTrain': indexLblSinTrainPreP, 'indexLblSinTest': indexLblSinTestPreP,
                   'indexCmpSinTrain': indexCmpSinTrainPreP, 'indexCmpSinTest': indexCmpSinTestPreP,
                   'repeatTimes': repeatTimes, 'K': K}




    dictOutput = {}
    if args.expert == 'global' or 'all':
        betaPlusAbs, constPlusAbs, aucLblPlusAbs, aucCmpPlusAbs, aucRSDPlusAbs, scoreRSDPlusAbs = CVGlobalModel(args.loss, dataPlus, indexCVPlus, args.alpha, args.lam)
        betaPrePAbs, constPrePAbs, aucLblPrePAbs, aucCmpPrePAbs, aucRSDPrePAbs, scoreRSDPrePAbs = CVGlobalModel(args.loss, dataPreP, indexCVPreP, args.alpha, args.lam)
        dictGlobal = {'betaPlusAbs': betaPlusAbs, 'constPlusAbs': constPlusAbs, 'aucLblPlusAbs': aucLblPlusAbs,
                      'aucCmpPlusAbs': aucCmpPlusAbs, 'aucRSDPlusAbs': aucRSDPlusAbs, 'scoreRSDPlusAbs': scoreRSDPlusAbs,
                      'betaPrePAbs': betaPrePAbs, 'constPrePAbs': constPrePAbs, 'aucLblPrePAbs': aucLblPrePAbs,
                      'aucCmpPrePAbs': aucCmpPrePAbs, 'aucRSDPrePAbs': aucRSDPrePAbs,'scoreRSDPrePAbs': scoreRSDPrePAbs,}
        dictOutput.update(dictGlobal)

    if args.expert == 'bias' or 'all':
        betaPlusBias, constPlusBias, aucLblPlusBias, aucCmpPlusBias, aucRSDPlusBias, scoreRSDPlusBias = CVExpertBiasModel(args.loss, dataPlus, indexCVPlus, args.alpha, args.lam)
        betaPrePBias, constPrePBias, aucLblPrePBias, aucCmpPrePBias, aucRSDPrePBias, scoreRSDPrePBias = CVExpertBiasModel(args.loss, dataPreP, indexCVPreP, args.alpha, args.lam)
        dictBias = {'betaPlusBias': betaPlusBias, 'constPlusBias': constPlusBias, 'aucLblPlusBias': aucLblPlusBias,
                    'aucCmpPlusBias': aucCmpPlusBias, 'aucRSDPlusBias': aucRSDPlusBias,
                    'scoreRSDPlusBias': scoreRSDPlusBias,
                    'betaPrePBias': betaPrePBias, 'constPrePBias': constPrePBias, 'aucLblPrePBias': aucLblPrePBias,
                    'aucCmpPrePBias': aucCmpPrePBias, 'aucRSDPrePBias': aucRSDPrePBias,
                    'scoreRSDPrePBias': scoreRSDPrePBias,
                    }
        dictOutput.update(dictBias)

    if args.expert == 'expert' or 'all':
        betaPlusUnique, constPlusUnique, aucLblPlusUnique, aucCmpPlusUnique, aucRSDPlusUnique, scoreRSDPlusUnique = CVExpertModel(args.loss, dataPlus, indexCVPlus, args.alpha, args.lam, cmpExpOrder=[2, 6, 9, 10, 12])
        betaPrePUnique, constPrePUnique, aucLblPrePUnique, aucCmpPrePUnique, aucRSDPrePUnique, scoreRSDPrePUnique = CVExpertModel(args.loss, dataPreP, indexCVPreP, args.alpha, args.lam, cmpExpOrder=[2, 6, 9, 10, 12])
        dictExpert = {'betaPlusUnique': betaPlusUnique, 'constPlusUnique': constPlusUnique,
                  'aucLblPlusUnique': aucLblPlusUnique, 'aucCmpPlusUnique': aucCmpPlusUnique,
                  'aucRSDPlusUnique': aucRSDPlusUnique, 'scoreRSDPlusUnique': scoreRSDPlusUnique,
                  'betaPrePUnique': betaPrePUnique, 'constPrePUnique': constPrePUnique,
                  'aucLblPrePUnique': aucLblPrePUnique, 'aucCmpPrePUnique': aucCmpPrePUnique,
                  'aucRSDPrePUnique': aucRSDPrePUnique, 'scoreRSDPrePUnique': scoreRSDPrePUnique}
        dictOutput.update(dictExpert)

    savemat(nameBase + '_' + str(args.alpha) + '_' + str(args.lam) + '.mat', mdict=dictOutput)
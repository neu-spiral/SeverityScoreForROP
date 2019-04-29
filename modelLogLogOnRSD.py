"""
This script is to generate the label and comparison result based on training both diagnostic label data (RSD or 13 experts absoluete labels) and pairwise comparison data

Parameters :
-------------
dataType: 'auto' or 'manual'
    'auto' would load the automatic segmentation data. 'manual' would load the manual segmentation data.
lambdWeight: list
    contains the weights on L1 penalty parameter
alphaWeight: list
    contains the weights on label data. The weights on comparison data would be (1-alphaWeight). 0 - train label data only, 1 - train comparison data only.
penaltyTimes: float
    the number times on the expert bias avoid the penalty on these biases. Larger number would have less penalty on bias.

Return :
-------------
.mat file
        Two types for classification. Plus: Plus vs Not Plus. PreP : Not Normal vs Normal. All of the model belowing are all incorporate with comparison data.
        aucLExp132Exp13 : Using expert absolute label with bias predict expert absolute label with bias.
        aucLExp132RSD : Using expert absolute label with bias predict concensus RSD label.
        aucLExp132Exp13NoB : Usint expert absolute label with bias to trian but test without using bias.
        aucLRSD2Exp13 : Using the concensus RSD label to train and test on experts absolute labels.
        aucLRSD2RSD : Using the concensus RSD label to train and test on RSD labels.
        aucCExp13 : Using the expert absolute label with bias to train and test on the comparison labels.
        aucCRSD : Using concensus RSD label to train and text on comparison labels.


Author: Peng Tian
Date: Septempber 2016

"""


from sklearn import metrics
import scipy.io as sio
import numpy as np
from scipy.io import loadmat
from cvxOpt import Log_Log
import sys
import pickle

def TrainSinleExp(alpha, labelSin,labelAbs, labelCmp,labelTrainPartition, labelTestPartition, cmpTrainPartition, cmpTestPartition, num_iters=10000):
    Yl = np.reshape(labelAbs[:, :-1], [-1, ], order='F')
    YlSin = 1 * labelSin
    YlRSD = labelAbs[:, 13]
    Ntol = N * numOfExpts4Lbl
    Mtol = M * numOfExpts4Cmp
    # prepare Beta File
    betaSinTotal = np.zeros([d, repeatTimes * K])
    constSinTotal = np.zeros([1, repeatTimes * K])

    # Prepare Exp13 File score variables
    scoreSin2Abs = np.zeros([N * numOfExpts4Lbl, repeatTimes])

    # PrePare RSD File score variables
    scoreSin2RSD = np.zeros([N, repeatTimes])

    # Prepare comparison score variables
    scoreSin2Cmp = np.zeros([M * numOfExpts4Cmp, repeatTimes * K])  # Train RSD Labels
    locSin2Cmp = np.zeros([M * numOfExpts4Cmp, repeatTimes * K]) # Train RSD Labels
    aucSin2Abs = np.zeros([1, repeatTimes])
    aucSin2RSD = np.zeros([1, repeatTimes])
    aucSin2Cmp = np.zeros([1, repeatTimes])
    betaMat = np.zeros([repeatTimes * K, d])
    constMat = np.zeros([repeatTimes * K, 1])
    for repeatCount in range(repeatTimes):
        for countFold in range(K):
            trainLIndex = labelTrainPartition[repeatCount][countFold].copy()
            testLIndex = labelTestPartition[repeatCount][countFold].copy()
            trainCIndex = cmpTrainPartition[repeatCount][countFold].copy()
            testCIndex = cmpTestPartition[repeatCount][countFold].copy()
            trainLIndex = np.reshape(trainLIndex,[-1,])
            testLIndex = np.reshape(testLIndex,[-1,])
            trainCIndex = np.reshape(trainCIndex,[-1,])
            testCIndex = np.reshape(testCIndex,[-1,])
            trainFeatC = cmpFeat[trainCIndex, :]
            testFeatC = cmpFeat[testCIndex, :]
            trainFeatL = labelFeat[trainLIndex, :]
            testFeatL = labelFeat[testLIndex, :]

            # Prepare 13 Experts with experts bias training and testing feats labels
            YtrainExp13C = np.array([])
            for eC in range(numOfExpts4Cmp):
                YtrainExp13C = np.append(YtrainExp13C, labelCmp[trainCIndex, eC])

            # Prepare RSD training adn testing feats labels
            XtrainSinL = 1 * trainFeatL
            XtrainSinC = np.tile(trainFeatC, [numOfExpts4Cmp, 1])
            YtrainSinL = 1 * YlSin[trainLIndex]
            YtrainSinC = 1 * YtrainExp13C
            countLamda = 0
            for lamda in lamdaWeights:
                # Train RSD Model
                betaSin, constSin = Log_Log(XtrainSinL,YtrainSinL,XtrainSinC, YtrainSinC,alpha,lamda)
                betaSin = np.array(betaSin)
                constSin = np.array(constSin)
                # Save the Paramter Values
                betaMat[countFold + K * repeatCount,:] = np.array(betaSin.T)
                constMat[countFold + K * repeatCount,:] = constSin


                # Test on Exp13 Label
                for eLT in range(numOfExpts4Lbl):
                    scoreSin2Abs[eLT * N + testLIndex, repeatCount] = np.reshape(
                        np.dot(testFeatL, betaSin) + constSin, [-1, ])
                for eCT in range(numOfExpts4Cmp):
                    scoreSin2Cmp[eCT * M + testCIndex, K * repeatCount + countFold] = np.reshape(
                        np.dot(testFeatC, betaSin)+constSin, [-1, ])
                    locSin2Cmp[eCT * M + testCIndex, K * repeatCount + countFold] = 1
                # Test On RSD Label
                scoreSin2RSD[testLIndex, repeatCount] = np.reshape(np.dot(testFeatL, betaSin), [-1, ])
                countLamda += 1


    # Compute all the scores and auc for each repeat time.
        aucSin2Abs[0, repeatCount] = metrics.roc_auc_score(Yl, scoreSin2Abs[:, repeatCount])
        aucSin2RSD[0, repeatCount] = metrics.roc_auc_score(YlRSD, scoreSin2RSD[:, repeatCount])
        indexCmpValidTe = np.where(np.reshape(locSin2Cmp[:, repeatCount], [-1, ]) != 0)[0]
        aucSin2Cmp[0, repeatCount] = metrics.roc_auc_score(Yc[indexCmpValidTe],
                                                       scoreSin2Cmp[indexCmpValidTe, repeatCount])



    return aucSin2Abs, aucSin2RSD, aucSin2Cmp, betaMat, constMat, scoreSin2RSD





dataType = 'CNN'
nameBase = '../Data/result/rop/rsd/MS_RSD_LogLog_L1_CV1' + '_' +dataType
dataFile = loadmat('../data/ropData/iROP_6DD_1st100_Partition.mat')
aveLabelFile = loadmat('.../data/ropData/AveLabels.mat')
# alphaWeights = [1.0]
# lamdaWeights = [1e-10]
alphaWeights = [float(sys.argv[1])]
lamdaWeights = [float(sys.argv[2])]
numOfExpts4Lbl = 13
numOfExpts4Cmp = 5
penaltyTimes = 100
lenAlpha = len(alphaWeights)
lenLamda = len(lamdaWeights)
labelPlusSet = dataFile['labelPlus']
labelPrePSet = dataFile['labelPreP']
cmpLabel = dataFile['cmpLabel']
Yc = dataFile['cmpLabel1Column'][0,:]
repeatTimes = int(dataFile['repeatTimes'][0,:])
K = int(dataFile['numOfFolds'][0,:])
RSDTrainPlusPartition = dataFile['RSDTrainPlusPartition']
RSDTestPlusPartition = dataFile['RSDTestPlusPartition']
cmpTrainPlusPartition = dataFile['cmpTrainPlusPartition']
cmpTestPlusPartition = dataFile['cmpTestPlusPartition']
RSDTrainPrePPartition = dataFile['RSDTrainPrePPartition']
RSDTestPrePPartition = dataFile['RSDTestPrePPartition']
cmpTrainPrePPartition = dataFile['cmpTrainPrePPartition']
cmpTestPrePPartition = dataFile['cmpTestPrePPartition']
labelRSDPlus = labelPlusSet[:,-1]
labelRSDPreP = labelPrePSet[:,-1]
labelAvePlus = aveLabelFile['labelAvePlus']
labelAvePreP = aveLabelFile['labelAvePreP']


if dataType == 'manual':
    labelFeatOrigin = dataFile['labelFeatManual']
    cmpFeatOrigin = dataFile['cmpFeatManual']
elif dataType == 'auto':
    labelFeatOrigin = dataFile['labelFeatAuto']
    cmpFeatOrigin = dataFile['cmpFeatAuto']
elif dataType.segment == 'CNN':
    cnn_file = pickle.load(open('../data/ropData/featuresOf100PredictedDiscCenters(DuplicatesRemoved)_ordered.p','rb'))
    labelFeatOrigin = cnn_file['labelFeat']
    cmpFeatOrigin = cnn_file['cmpFeat']
else:
    assert('dataType should be manual or auto')
N, d = labelFeatOrigin.shape
M, _ = cmpFeatOrigin.shape
labelFeat = 1 * labelFeatOrigin
cmpFeat = 1 * cmpFeatOrigin



# main script
aucRSD2AbsPlus, aucRSD2RSDPlus, aucRSD2CmpPlus, betaRSDPlus, constRSDPlus, scoreRSD2RSDPlus = TrainSinleExp(alphaWeights[0],
                                                                                                         labelRSDPlus,labelPlusSet,
                                                                                                         cmpLabel,
                                                                                                         RSDTrainPlusPartition,
                                                                                                         RSDTestPlusPartition,
                                                                                                         cmpTrainPlusPartition,
                                                                                                         cmpTestPlusPartition)
aucRSD2AbsPreP, aucRSD2RSDPreP, aucRSD2CmpPreP, betaRSDPreP, constRSDPreP, scoreRSD2RSDPreP = TrainSinleExp(alphaWeights[0],
                                                                                                         labelRSDPreP,labelPrePSet,
                                                                                                         cmpLabel,
                                                                                                         RSDTrainPrePPartition,
                                                                                                         RSDTestPrePPartition,
                                                                                                         cmpTrainPrePPartition,
                                                                                                         cmpTestPrePPartition)
aucAve2AbsPlus, aucAve2RSDPlus, aucAve2CmpPlus, betaAvePlus, constAvePlus, scoreAve2RSDPlus = TrainSinleExp(alphaWeights[0],
                                                                                                         labelAvePlus[:,0],labelPlusSet,
                                                                                                         cmpLabel,
                                                                                                         RSDTrainPlusPartition,
                                                                                                         RSDTestPlusPartition,
                                                                                                         cmpTrainPlusPartition,
                                                                                                         cmpTestPlusPartition)
aucAve2AbsPreP, aucAve2RSDPreP, aucAve2CmpPreP, betaAvePreP, constAvePreP, scoreAve2RSDPreP = TrainSinleExp(alphaWeights[0],
                                                                                                         labelAvePreP[:,0],labelPrePSet,
                                                                                                         cmpLabel,
                                                                                                         RSDTrainPrePPartition,
                                                                                                         RSDTestPrePPartition,
                                                                                                         cmpTrainPrePPartition,
                                                                                                         cmpTestPrePPartition)

outputDict = {'aucRSD2AbsPlus': aucRSD2AbsPlus, 'aucRSD2RSDPlus':aucRSD2RSDPlus, 'aucRSD2CmpPlus':aucRSD2CmpPlus,
              'betaRSDPlus':betaRSDPlus, 'constRSDPlus':constRSDPlus, 'scoreRSD2RSDPlus':scoreRSD2RSDPlus,
              'aucRSD2AbsPreP':aucRSD2AbsPreP, 'aucRSD2RSDPreP':aucRSD2RSDPreP, 'aucRSD2CmpPreP':aucRSD2CmpPreP,
              'betaRSDPreP':betaRSDPreP, 'constRSDPreP':constRSDPreP, 'scoreRSD2RSDPreP':scoreRSD2RSDPreP,
              'aucAve2AbsPlus':aucAve2AbsPlus, 'aucAve2RSDPlus':aucAve2RSDPlus, 'aucAve2CmpPlus':aucAve2CmpPlus,
              'betaAvePlus':betaAvePlus, 'constAvePlus':constAvePlus, 'scoreAve2RSDPlus':scoreAve2RSDPlus,
              'aucAve2AbsPreP':aucAve2AbsPreP, 'aucAve2RSDPreP':aucAve2RSDPreP, 'aucAve2CmpPreP':aucAve2CmpPreP,
              'betaAvePreP':betaAvePreP, 'constAvePreP':constAvePreP, 'scoreAve2RSDPreP':scoreAve2RSDPreP}
sio.savemat(nameBase + '_' + str(alphaWeights[0]) + '_' + str(lamdaWeights[0]) + '.mat', outputDict)
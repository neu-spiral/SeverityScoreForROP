# This file is loading and preparing the data and labels for the cross-validation partitions
# Parameter
#  ----------
# K : scalar
#       number of folds
# numOfRep : scalar
#       number of re-partition times

# Return
# --------------
#
#

import numpy as np
from sklearn.cross_validation import StratifiedKFold
from loadFileLib import loadComparisonData
import scipy.io as sio
from scipy.io import loadmat
from sklearn.preprocessing import normalize

K = 5
numOfRep = 10
numOfExpts4Lbl = 13
numOfExpts4Cmp = 5
# ----------------------

labeldata = loadmat('../data/ropData/ExpertsLabel13.mat')
label13 = labeldata['ExpertsLabel13']
dataCmpFileManual = loadmat('../data/ropData/iROPcmpData_6DD_Norm.mat')
cmpFeatManual = dataCmpFileManual['cmpFeats']
dataManual = loadmat('.../data/ropData/iROPData_6DD.mat')
feat1stManual, classLabels1st = dataManual['feats1st'], dataManual['classLabels1st']
featOriginManual = feat1stManual*1
featManual = normalize(featOriginManual, norm='l2', axis=0)

dataCmpFileAuto = loadmat('.../data/ropData/iROPcmpData_6DD_Norm_Auto.mat')
cmpFeatAuto = dataCmpFileAuto['cmpFeats']
dataAuto = loadmat('../data/ropData/iROPData_6DD_Auto.mat')
feat1stAuto = dataAuto['feats']
featOriginAuto = feat1stAuto*1
featAuto = normalize(featOriginAuto, norm='l2', axis=0)

# Absolute label data is normalized here and comparison data has been normalized in advance.
N, d = featManual.shape
indRSDPlus = np.where(classLabels1st == 1)[0]
indRSDPreplus = np.where(classLabels1st == 2)[0]
indRSDNormal = np.where(classLabels1st == 3)[0]
indAllPlus = np.where(label13 == 2)
indAllPreplus = np.where(label13 == 1)
indAllNormal = np.where(label13 == 0)
indAllOut = np.where(label13 == 1.5)
RSDLPlus = -1 * np.ones([N, 1])
RSDLPlus[indRSDPlus] = 1
RSDLPreP = -1 * np.ones([N, 1])
RSDLPreP[indRSDPreplus] = 1
RSDLPreP[indRSDPlus] = 1
LPlus = -1 * np.ones([N, numOfExpts4Lbl])
LPlus[indAllPlus] = 1
LPreP = -1 * np.ones([N, numOfExpts4Lbl])
LPreP[indAllPreplus] = 1
LPreP[indAllPlus] = 1
LPreP[indAllOut] = 1
labelPlus = np.concatenate((LPlus, RSDLPlus), axis=1)
labelPreP = np.concatenate((LPreP, RSDLPreP), axis=1)

# Load Comparison Data
nameExperts = ['mike', 'paul', 'pete', 'susan']
IdOrder, cmpData = loadComparisonData('karyn')  # the order in the 13 experts is: 2, 6, 9, 10,12 (start from 0)
Ntol = N * numOfExpts4Lbl
M, _ = cmpData.shape
Mtol = M * numOfExpts4Cmp
cmpDataL = np.reshape(1 * cmpData[:, 2], [-1, 1])
for i in range(numOfExpts4Cmp - 1):
    _, cmpDataTmp = loadComparisonData(nameExperts[i])
    cmpDataL = np.concatenate((cmpDataL, np.reshape(cmpDataTmp[:, 2], [-1, 1])), axis=1)
Yc = np.reshape(cmpDataL, [-1, ], order='F')


def partitioning(label):
    RSDTrainPartition = [[None] * K] * numOfRep
    RSDTestPartition = [[None] * K] * numOfRep
    cmpTrainPartition = [[None] * K] * numOfRep
    cmpTestPartition = [[None] * K] * numOfRep
    for repCount in range(numOfRep):
        skf = StratifiedKFold(np.reshape(label, [-1, ]), n_folds=K, shuffle=True)
        CVIndex = list(skf)
        for KCount in range(K):
            trainLIndex = CVIndex[KCount][0].copy()
            testLIndex = CVIndex[KCount][1].copy()
            trainCIndex = np.array([])
            testCIndex = np.array([])
            for i in range(M):
                imgIndCi, imgIndCj = int(
                    IdOrder[float(cmpData[i, 0])]) - 1, int(IdOrder[float(cmpData[i, 1])]) - 1
                if imgIndCi in trainLIndex and imgIndCj in trainLIndex:
                    trainCIndex = np.append(trainCIndex, i)
                elif imgIndCi in testLIndex and imgIndCj in testLIndex:
                    testCIndex = np.append(testCIndex, i)
                # else:
                    # testCIndex = np.append(testCIndex, i)
            trainCIndex = np.int_(trainCIndex)
            testCIndex = np.int_(testCIndex)
            RSDTrainPartitionTemp = RSDTrainPartition[repCount][:]
            RSDTestPartitionTemp = RSDTestPartition[repCount][:]
            cmpTrainPartitionTemp = cmpTrainPartition[repCount][:]
            cmpTestPartitionTemp = cmpTestPartition[repCount][:]

            RSDTrainPartitionTemp[KCount] = 1 * trainLIndex
            RSDTestPartitionTemp[KCount] = 1 * testLIndex
            cmpTrainPartitionTemp[KCount] = 1 * trainCIndex
            cmpTestPartitionTemp[KCount] = 1 * testCIndex

            RSDTrainPartition[repCount] = RSDTrainPartitionTemp
            RSDTestPartition[repCount] = RSDTestPartitionTemp
            cmpTrainPartition[repCount] = cmpTrainPartitionTemp
            cmpTestPartition[repCount] = cmpTestPartitionTemp
    return RSDTrainPartition, RSDTestPartition, cmpTrainPartition, cmpTestPartition


RSDTrainPlusPartition, RSDTestPlusPartition, cmpTrainPlusPartition, cmpTestPlusPartition = partitioning(RSDLPlus)
RSDTrainPrePPartition, RSDTestPrePPartition, cmpTrainPrePPartition, cmpTestPrePPartition = partitioning(RSDLPreP)
outputDict = {'labelFeatManual': featManual, 'cmpFeatManual': cmpFeatManual,
              'labelFeatAuto': featAuto, 'cmpFeatAuto': cmpFeatAuto,
              'labelPlus': labelPlus, 'labelPreP': labelPreP,
              'cmpLabel': cmpDataL, 'cmpLabel1Column': Yc, 'repeatTimes': numOfRep, 'numOfFolds': K,
              'RSDTrainPlusPartition': RSDTrainPlusPartition, 'RSDTestPlusPartition': RSDTestPlusPartition,
              'cmpTrainPlusPartition': cmpTrainPlusPartition, 'cmpTestPlusPartition': cmpTestPlusPartition,
              'RSDTrainPrePPartition': RSDTrainPrePPartition, 'RSDTestPrePPartition': RSDTestPrePPartition,
              'cmpTrainPrePPartition': cmpTrainPrePPartition, 'cmpTestPrePPartition': cmpTestPrePPartition}
sio.savemat('../data/ropData/iROP_6DD_1st100_Partition.mat', outputDict)










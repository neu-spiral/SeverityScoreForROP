"""
This class is for data preparation for expert data, expert data with bias and expert unique data.

Parameter:
------------
data : N by d matrix. N is the number of samples and d is the dimension of features

Return:
------------


Author: Peng Tian
Date : November 2016

"""
import numpy as np
import sys

def dataLabelPrePare(positiveLabel,featSin,labelAbs,featCmp,labelCmp,labelRSD=None):
    dataType = 'manual'
    nameBase = '../../Data/Result/RSD&Bias2RSD&Exp13_L1_CV1' + '_' + dataType
    dataFile = loadmat('../data/ropData/iROP_6DD_1st100_Partition.mat')
    numOfExpts4Lbl = 13
    numOfExpts4Cmp = 5
    penaltyTimes = 100
    labelPlusSet = dataFile['labelPlus']
    labelPrePSet = dataFile['labelPreP']
    cmpLabel = dataFile['cmpLabel']
    Yc = dataFile['cmpLabel1Column'][0, :]
    repeatTimes = int(dataFile['repeatTimes'][0, :])
    K = int(dataFile['numOfFolds'][0, :])
    RSDTrainPlusPartition = dataFile['RSDTrainPlusPartition']
    RSDTestPlusPartition = dataFile['RSDTestPlusPartition']
    cmpTrainPlusPartition = dataFile['cmpTrainPlusPartition']
    cmpTestPlusPartition = dataFile['cmpTestPlusPartition']
    RSDTrainPrePPartition = dataFile['RSDTrainPrePPartition']
    RSDTestPrePPartition = dataFile['RSDTestPrePPartition']
    cmpTrainPrePPartition = dataFile['cmpTrainPrePPartition']
    cmpTestPrePPartition = dataFile['cmpTestPrePPartition']

    if dataType == 'manual':
        labelFeatOrigin = dataFile['labelFeatManual']
        cmpFeatOrigin = dataFile['cmpFeatManual']
    elif dataType == 'auto':
        labelFeatOrigin = dataFile['labelFeatAuto']
        cmpFeatOrigin = dataFile['cmpFeatAuto']
    else:
        sys.exit('dataType should be manual or auto')
    N, d = labelFeatOrigin.shape
    M, _ = cmpFeatOrigin.shape
    labelFeat = 1 * labelFeatOrigin
    cmpFeat = 1 * cmpFeatOrigin



class dataPrepare():

    def __init__(self,inputData):
        # featSin is the N by d matrix and N is the number of concensus labels or one expert.
        # featCmp is the M by d matrix and M is the the number of comparisons on singe expert.
        # labelRSD is the (N,) array contains +1 or -1.
        # labelAbs is the (N, numOfExpLbl) array contains +1 or -1, N is the number of samples for a singe expert and numOfExperLbl is the number of expert who labeled the N images.
        # labelCmp is the (M, numOfExpCmp) array contains +1 or -1, M is the M is the the number of comparisons for singe expert, numOfExpCmp is the number of expert who label the comparison data.
        # indexLblTrain is a list range in (0,N-1) that contains the training label data index for one expert or RSD. indexLbl Test is for testing.
        # indexCmpTrain is a list range in (0,M-1) that contains the training comparison data for one expert.

        self.featLblSin = np.array(inputData['featLblSin'])
        self.featCmpSin = np.array(inputData['featCmpSin'])
        self.labelRSD = np.array(inputData['labelRSD'])
        self.labelAbs = np.array(inputData['labelAbs'])
        self.labelCmp = np.array(inputData['labelCmp'])
        self.indexTrLblSin = np.array(inputData['indexLblSinTrain'])
        self.indexTeLblSin = np.array(inputData['indexLblSinTest'])
        self.indexTrCmpSin = np.array(inputData['indexCmpSinTrain'])
        self.indexTeCmpSin = np.array(inputData['indexCmpSinTest'])
        _, self.numOfExpLbl = self.labelAbs.shape
        _, self.numOfExpCmp = self.labelCmp.shape

    def ExpAbs(self):
        # This function is to generate the expert absolute labels and comparison training and testing set.
        _,numOfExpLbl = self.labelAbs.shape
        _,numOfExpCmp = self.labelCmp.shape
        featTrLblAbs = np.tile(self.featLblSin[self.indexTrLblSin,:],[numOfExpLbl,1])
        labelTrLblAbs = np.reshape(self.labelAbs[self.indexTrLblSin,:],[-1,],order='F')
        featTeLblSinAbs = self.featLblSin[self.indexTeLblSin,:]
        featTrCmpAbs = np.tile(self.featCmpSin[self.indexTrCmpSin,:], [numOfExpCmp,1])
        labelTrCmpAbs = np.reshape(self.labelCmp[self.indexTrCmpSin,:],[-1,],order='F')
        featTeCmpSinAbs = self.featCmpSin[self.indexTeCmpSin,:]
        expAbs = {'featTrainLblAbs':featTrLblAbs,'labelTrainLblAbs':labelTrLblAbs,
                  'featTestLblSinAbs':featTeLblSinAbs,
                  'featTrainCmpAbs':featTrCmpAbs, 'labelTrainCmpAbs':labelTrCmpAbs,
                  'featTestCmpSinAbs':featTeCmpSinAbs}

        return expAbs

    def ExpAbsBias(self, penaltyTimes):
        # generate the image feature with expert bias. The penaltyTims is a value that higher than the 1 to avoid the penalty on Bias.
        featTrLblAbs = np.tile(self.featLblSin[self.indexTrLblSin,:],[self.numOfExpLbl,1])
        featTeLblSin = self.featLblSin[self.indexTeLblSin,:]
        NTrLblSin, NTeLblSin = len(self.indexTrLblSin), len(self.indexTeLblSin)
        NTrCmpSin, NTeCmpSin = len(self.indexTrLblSin), len(self.indexTeLblSin)
        featBiasTrLbl = np.zeros([self.numOfExpLbl * NTrLblSin, self.numOfExpLbl])
        featTeLblBiasList = [None] * self.numOfExpLbl
        for expLbl in range(self.numOfExpLbl):
            featBiasTrLbl[expLbl*NTrLblSin:(expLbl+1)*NTrLblSin,expLbl]=penaltyTimes
            featBiasTeLbl = np.zeros([NTeLblSin,self.numOfExpLbl])
            featBiasTeLbl[:,expLbl] = penaltyTimes
            featTeLblBias = np.concatenate([featTeLblSin,featBiasTeLbl],axis=1)
            featTeLblBiasList[expLbl] = featTeLblBias
        featTrLblBias = np.concatenate([featTrLblAbs, featBiasTrLbl],axis=1)
        labelTrLblBias = np.reshape(self.labelAbs[self.indexTrLblSin, :], [-1, ], order='F')
        featTrCmpAbs = np.tile(self.featCmpSin[self.indexTrCmpSin,:], [self.numOfExpCmp,1])
        featTrCmpBias = np.concatenate([featTrCmpAbs, np.zeros([featTrCmpAbs.shape[0], self.numOfExpLbl])],axis=1)
        labelTrCmpBias = np.reshape(self.labelCmp[self.indexTrCmpSin, :], [-1, ], order='F')
        featTeCmpBiasSin = self.featCmpSin[self.indexTeCmpSin,:]
        featTeCmpSinBias = np.concatenate([featTeCmpBiasSin, np.zeros([featTeCmpBiasSin.shape[0],self.numOfExpLbl])],axis=1)

        expAbsBias = {'featTrainLblBias':featTrLblBias,'labelTrainLblBias':labelTrLblBias,
                      'featTestLblBiasList':featTeLblBiasList,
                      'featTrainCmpBias':featTrCmpBias, 'labelTrainCmpBias':labelTrCmpBias,
                      'featTestCmpSinBias':featTeCmpSinBias}
        return expAbsBias

    def ExpUnique(self,cmpExpOrder = [2, 6, 9, 10, 12]):
        # This function generate the feature and labels for training and testing with each expert data.
        featTrLblSin = self.featLblSin[self.indexTrLblSin,:]
        featTrLblList = [featTrLblSin] * self.numOfExpLbl
        labelTrLblList = list()
        for expLbl in range(self.numOfExpLbl):
            labelTrLblList.append(self.labelAbs[self.indexTrLblSin,expLbl])
        featTeLblSin = self.featLblSin[self.indexTeLblSin,:]
        featTeLblList = [featTeLblSin] * self.numOfExpLbl

        featTrCmpList = [None] * self.numOfExpLbl
        labelTrCmpList = [None] * self.numOfExpLbl
        featTeCmpList = [None] * self.numOfExpLbl
        for expCmp in range(self.numOfExpCmp):
            featTrCmpList[cmpExpOrder[expCmp]] = self.featCmpSin[self.indexTrCmpSin,:]
            labelTrCmpList[cmpExpOrder[expCmp]] = self.labelCmp[self.indexTrCmpSin,expCmp]
            featTeCmpList[cmpExpOrder[expCmp]] = self.featCmpSin[self.indexTeCmpSin,:]

        expUnique = {'featTrainLblUniqueList':featTrLblList,'labelTrainLblUniqueList':labelTrLblList,
                      'featTestLblUniqueList':featTeLblList,
                      'featTrainCmpUniqueList':featTrCmpList, 'labelTrainCmpUnique':labelTrCmpList,
                      'featTestCmpUniqueList':featTeCmpList}
        return expUnique








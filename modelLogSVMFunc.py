"""
This file contains the training function for expert
"""
from dataPreparation import dataPrepare
from cvxpyMdl import Log_SVM
from cvxOpt import Logistic
import numpy as np
import sys

def LogSVMTrTeExpAbs(inputData,alpha,lamda,penaltyTimes=100,penaltyType='l1',numMaxIters=10000):
    # Train and Test the expert absolute labels.
    # inputData is a dictionary and contains the following keys:
    # featSin is the N by d matrix and N is the number of concensus labels or one expert.
    # featCmp is the M by d matrix and M is the the number of comparisons on singe expert.
    # labelRSD is the (N,) array contains +1 or -1.
    # labelAbs is the (N, numOfExpLbl) array contains +1 or -1, N is the number of samples for a singe expert and numOfExperLbl is the number of expert who labeled the N images.
    # labelCmp is the (M, numOfExpCmp) array contains +1 or -1, M is the M is the the number of comparisons for singe expert, numOfExpCmp is the number of expert who label the comparison data.
    # indexLblTrain is a list range in (0,N-1) that contains the training label data index for one expert or RSD. indexLbl Test is for testing.
    # indexCmpTrain is a list range in (0,M-1) that contains the training comparison data for one expert.
    data = dataPrepare(inputData)
    expAbs = data.ExpAbs()
    featTrLblAbs = expAbs['featTrainLblAbs']
    labelTrLblAbs= expAbs['labelTrainLblAbs']
    featTeLblSinAbs = expAbs['featTestLblSinAbs']
    featTrCmpAbs = expAbs['featTrainCmpAbs']
    labelTrCmpAbs = expAbs['labelTrainCmpAbs']
    featTeCmpSinAbs = expAbs['featTestCmpSinAbs']

    if penaltyType=='l1':
        beta,const = Log_SVM(featTrLblAbs,labelTrLblAbs,featTrCmpAbs,labelTrCmpAbs,absWeight=alpha,lamda=lamda)
        scoreLblTestSin = np.dot(featTeLblSinAbs, np.array(beta))+const
        scoreCmpTestSin = np.dot(featTeCmpSinAbs, np.array(beta))+const
        beta = np.array(beta).T
    else: sys.exit('The penalty type must be either l1 or l2')

    return beta, const, scoreLblTestSin, scoreCmpTestSin

def LogSVMTrTeExpBias(inputData,alpha,lamda,penaltyTimes=100,penaltyType='l1',numMaxIters=10000):
    # Train and Test the expert absolute labels.
    # inputData is a dictionary and contains the following keys:
    # featSin is the N by d matrix and N is the number of concensus labels or one expert.
    # featCmp is the M by d matrix and M is the the number of comparisons on singe expert.
    # labelRSD is the (N,) array contains +1 or -1.
    # labelAbs is the (N, numOfExpLbl) array contains +1 or -1, N is the number of samples for a singe expert and numOfExperLbl is the number of expert who labeled the N images.
    # labelCmp is the (M, numOfExpCmp) array contains +1 or -1, M is the M is the the number of comparisons for singe expert, numOfExpCmp is the number of expert who label the comparison data.
    # indexLblTrain is a list range in (0,N-1) that contains the training label data index for one expert or RSD. indexLbl Test is for testing.
    # indexCmpTrain is a list range in (0,M-1) that contains the training comparison data for one expert.
    data = dataPrepare(inputData)
    expBias = data.ExpAbsBias(penaltyTimes=penaltyTimes)
    featTrLblBias = expBias['featTrainLblBias']
    labelTrLblBias= expBias['labelTrainLblBias']
    featTeLblListBias = expBias['featTestLblBiasList']
    featTrCmpBias = expBias['featTrainCmpBias']
    labelTrCmpBias = expBias['labelTrainCmpBias']
    featTeCmpSinBias = expBias['featTestCmpSinBias']
    numOfExpLbl = inputData['numOfExpLbl']
    numOfExpCmp = inputData['numOfExpCmp']
    scoreLblListTest = [None] * numOfExpLbl
    if penaltyType=='l1':
        beta,const = Log_SVM(featTrLblBias,labelTrLblBias,featTrCmpBias,labelTrCmpBias,absWeight=alpha,lamda=lamda)
        for expLbl in range(numOfExpLbl):
            scoreLblListTest[expLbl] = np.dot(featTeLblListBias[expLbl].copy(), np.array(beta))+const
        scoreCmpTestSin = np.dot(featTeCmpSinBias, np.array(beta))+const
        beta = np.array(beta).T
    else: sys.exit('The penalty type must be either l1 or l2')

    return beta, const, scoreLblListTest, scoreCmpTestSin


def LogSVMTrTeExpUnique(inputData,alpha,lamda,cmpExpOrder,penaltyType='l1',numMaxIters=10000):
    # Train and Test the each experts data.
    # inputData is a dictionary and contains the following keys:
    # featSin is the N by d matrix and N is the number of concensus labels or one expert.
    # featCmp is the M by d matrix and M is the the number of comparisons on singe expert.
    # labelRSD is the (N,) array contains +1 or -1.
    # labelAbs is the (N, numOfExpLbl) array contains +1 or -1, N is the number of samples for a singe expert and numOfExperLbl is the number of expert who labeled the N images.
    # labelCmp is the (M, numOfExpCmp) array contains +1 or -1, M is the M is the the number of comparisons for singe expert, numOfExpCmp is the number of expert who label the comparison data.
    # indexLblTrain is a list range in (0,N-1) that contains the training label data index for one expert or RSD. indexLbl Test is for testing.
    # indexCmpTrain is a list range in (0,M-1) that contains the training comparison data for one expert.
    data = dataPrepare(inputData)
    expUnique = data.ExpUnique()
    featTrLblUniqueList = expUnique['featTrainLblUniqueList']
    labelTrLblUniqueList = expUnique['labelTrainLblUniqueList']
    featTeLblUniqueList = expUnique['featTestLblUniqueList']
    featTrCmpUniqueList = expUnique['featTrainCmpUniqueList']
    labelTrCmpUniqueList = expUnique['labelTrainCmpUnique']
    featTeCmpUniqueList = expUnique['featTestCmpUniqueList']
    numOfExpLbl = inputData['numOfExpLbl']
    numOfExpCmp = inputData['numOfExpCmp']
    betaList = [None] * numOfExpLbl
    constList = [None] * numOfExpLbl
    scoreLblListTest = [None] * numOfExpLbl
    scoreCmpListTest = [None] * numOfExpLbl
    if penaltyType=='l1':
        if alpha != 0.0:
            for exp in range(numOfExpLbl):
                if not exp in cmpExpOrder:
                    beta,const = Logistic(featTrLblUniqueList[exp],labelTrLblUniqueList[exp],alpha=alpha,lamda=lamda)
                    betaList[exp] = np.array(beta).T
                    constList[exp] = const
                    scoreLblListTest[exp] = np.dot(featTeLblUniqueList[exp].copy(), np.array(beta)) + const
                elif exp in cmpExpOrder:
                    beta,const = Log_SVM(featTrLblUniqueList[exp],labelTrLblUniqueList[exp],featTrCmpUniqueList[exp],labelTrCmpUniqueList[exp],absWeight=alpha,lamda=lamda)
                    scoreLblListTest[exp] = np.dot(featTeLblUniqueList[exp].copy(), np.array(beta)) + const
                    scoreCmpListTest[exp] = np.dot(featTeCmpUniqueList[exp].copy(), np.array(beta)) + const
                    betaList[exp] = np.array(beta).T
                    constList[exp] = const
                else: sys.exit('The expert order is wrong.')
        else :
            for exp in cmpExpOrder:
                beta, const = Log_SVM(featTrLblUniqueList[exp], labelTrLblUniqueList[exp],
                                               featTrCmpUniqueList[exp], labelTrCmpUniqueList[exp], absWeight=alpha,
                                               lamda=lamda)
                scoreLblListTest[exp] = np.dot(featTeLblUniqueList[exp].copy(), np.array(beta)) + const
                scoreCmpListTest[exp] = np.dot(featTeCmpUniqueList[exp].copy(), np.array(beta)) + const
                betaList[exp] = np.array(beta).T
                constList[exp] = const
    else: sys.exit('The penalty type must be either l1')

    return betaList, constList, scoreLblListTest, scoreCmpListTest
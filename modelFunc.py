"""
This file contains the training function for expert
"""
from dataPreparation import dataPrepare
from cvxOpt import Log_Log, SVM_Log, Logistic
from cvxpyMdl import SVM_SVM, Log_SVM, SVM
import numpy as np
import sys
from sklearn.ensemble import GradientBoostingRegressor

def TrTeGlobalModel(lossFunc,inputData,alpha,lamda):
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
    if lossFunc =='LogLog':
        beta, const = Log_Log(featTrLblAbs,labelTrLblAbs,featTrCmpAbs,labelTrCmpAbs,absWeight=alpha,lamda=lamda)
    elif lossFunc =='LogSVM':
        beta, const = Log_SVM(featTrLblAbs,labelTrLblAbs,featTrCmpAbs,labelTrCmpAbs,absWeight=alpha,lamda=lamda)
    elif lossFunc == 'SVMLog':
        beta, const = SVM_Log(featTrLblAbs,labelTrLblAbs,featTrCmpAbs,labelTrCmpAbs,absWeight=alpha,lamda=lamda)
    elif lossFunc == 'SVMSVM':
        beta, const = SVM_SVM(featTrLblAbs,labelTrLblAbs,featTrCmpAbs,labelTrCmpAbs,absWeight=alpha,lamda=lamda)
    else:
        sys.exit('Please choose the correct loss function from one of {Log_Log,Log_SVM,SVM_Log,SVM_SVM}')
    scoreLblTestSin = np.dot(featTeLblSinAbs, np.array(beta))+const
    scoreCmpTestSin = np.dot(featTeCmpSinAbs, np.array(beta))+const
    beta = np.array(beta).T

    return beta, const, scoreLblTestSin, scoreCmpTestSin

def TrTeExpertBiasModel(lossFunc,inputData,alpha,lamda,penaltyTimes=100):
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
    if lossFunc =='LogLog':
        beta, const = Log_Log(featTrLblBias,labelTrLblBias,featTrCmpBias,labelTrCmpBias,absWeight=alpha,lamda=lamda)
    elif lossFunc =='LogSVM':
        beta, const = Log_SVM(featTrLblBias,labelTrLblBias,featTrCmpBias,labelTrCmpBias,absWeight=alpha,lamda=lamda)
    elif lossFunc == 'SVMLog':
        beta, const = SVM_Log(featTrLblBias,labelTrLblBias,featTrCmpBias,labelTrCmpBias,absWeight=alpha,lamda=lamda)
    elif lossFunc == 'SVMSVM':
        beta, const = SVM_SVM(featTrLblBias,labelTrLblBias,featTrCmpBias,labelTrCmpBias,absWeight=alpha,lamda=lamda)
    elif lossFunc == 'Boost':
        NAbs,_=featTrLblBias.shape()
        NCmp,_=featTrCmpBias.shape()
        est = GradientBoostingRegressor(n_estimators=100,learning_rate=0.1,max_depth=5,random_state=1,loss='ls').fit()
    else:
        sys.exit('Please choose the correct loss function from one of {Log_Log,Log_SVM,SVM_Log,SVM_SVM}')
    for expLbl in range(numOfExpLbl):
        scoreLblListTest[expLbl] = np.dot(featTeLblListBias[expLbl].copy(), np.array(beta))+const
    scoreCmpTestSin = np.dot(featTeCmpSinBias, np.array(beta))+const
    beta = np.array(beta).T

    return beta, const, scoreLblListTest, scoreCmpTestSin


def TrTeExpertModel(lossFunc, inputData,alpha,lamda,cmpExpOrder):
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
    if alpha != 0.0:
        for exp in range(numOfExpLbl):
            if not exp in cmpExpOrder:
                if lossFunc == 'LogLog':
                    beta, const = Logistic(featTrLblUniqueList[exp],labelTrLblUniqueList[exp],lamda,alpha=alpha)
                elif lossFunc == 'LogSVM':
                    beta, const = Logistic(featTrLblUniqueList[exp],labelTrLblUniqueList[exp],lamda,alpha=alpha)
                elif lossFunc == 'SVMLog':
                    beta, const = SVM(featTrLblUniqueList[exp],labelTrLblUniqueList[exp],lamda,alpha=alpha)
                elif lossFunc == 'SVMSVM':
                    beta, const = SVM(featTrLblUniqueList[exp],labelTrLblUniqueList[exp],lamda,alpha=alpha)
                else:
                    sys.exit('Please choose the correct loss function from one of {Log_Log,Log_SVM,SVM_Log,SVM_SVM}')
                betaList[exp] = np.array(beta).T
                constList[exp] = const
                scoreLblListTest[exp] = np.dot(featTeLblUniqueList[exp].copy(), np.array(beta)) + const
            elif exp in cmpExpOrder:
                if lossFunc == 'LogLog':
                    beta, const = Log_Log(featTrLblUniqueList[exp],labelTrLblUniqueList[exp],featTrCmpUniqueList[exp],labelTrCmpUniqueList[exp],absWeight=alpha,lamda=lamda)
                elif lossFunc == 'LogSVM':
                    beta, const = Log_SVM(featTrLblUniqueList[exp],labelTrLblUniqueList[exp],featTrCmpUniqueList[exp],labelTrCmpUniqueList[exp],absWeight=alpha,lamda=lamda)
                elif lossFunc == 'SVMLog':
                    beta, const = SVM_Log(featTrLblUniqueList[exp],labelTrLblUniqueList[exp],featTrCmpUniqueList[exp],labelTrCmpUniqueList[exp],absWeight=alpha,lamda=lamda)
                elif lossFunc == 'SVMSVM':
                    beta, const = SVM_SVM(featTrLblUniqueList[exp],labelTrLblUniqueList[exp],featTrCmpUniqueList[exp],labelTrCmpUniqueList[exp],absWeight=alpha,lamda=lamda)
                else:
                    sys.exit('Please choose the correct loss function from one of {Log_Log,Log_SVM,SVM_Log,SVM_SVM}')
                scoreLblListTest[exp] = np.dot(featTeLblUniqueList[exp].copy(), np.array(beta)) + const
                scoreCmpListTest[exp] = np.dot(featTeCmpUniqueList[exp].copy(), np.array(beta)) + const
                betaList[exp] = np.array(beta).T
                constList[exp] = const
            else: sys.exit('The expert order is wrong.')
    else :
        for exp in cmpExpOrder:
            if lossFunc == 'LogLog':
                beta, const = Log_Log(featTrLblUniqueList[exp], labelTrLblUniqueList[exp], featTrCmpUniqueList[exp], labelTrCmpUniqueList[exp], absWeight=alpha, lamda=lamda)
            elif lossFunc == 'LogSVM':
                beta, const = Log_SVM(featTrLblUniqueList[exp], labelTrLblUniqueList[exp], featTrCmpUniqueList[exp], labelTrCmpUniqueList[exp], absWeight=alpha, lamda=lamda)
            elif lossFunc == 'SVMLog':
                beta, const = SVM_Log(featTrLblUniqueList[exp], labelTrLblUniqueList[exp], featTrCmpUniqueList[exp], labelTrCmpUniqueList[exp], absWeight=alpha, lamda=lamda)
            elif lossFunc == 'SVMSVM':
                beta, const = SVM_SVM(featTrLblUniqueList[exp], labelTrLblUniqueList[exp], featTrCmpUniqueList[exp], labelTrCmpUniqueList[exp], absWeight=alpha, lamda=lamda)
            else:
                sys.exit('Please choose the correct loss function from one of {Log_Log,Log_SVM,SVM_Log,SVM_SVM}')
            scoreLblListTest[exp] = np.dot(featTeLblUniqueList[exp].copy(), np.array(beta)) + const
            scoreCmpListTest[exp] = np.dot(featTeCmpUniqueList[exp].copy(), np.array(beta)) + const
            betaList[exp] = np.array(beta).T
            constList[exp] = const

    return betaList, constList, scoreLblListTest, scoreCmpListTest
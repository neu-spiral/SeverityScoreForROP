
import numpy as np
from modelFunc import TrTeGlobalModel, TrTeExpertBiasModel, TrTeExpertModel
from sklearn import metrics

def CVGlobalModel(lossFunc, dataDict, indexCVDict, alpha, lamda, penaltyTimes=100, penaltyType='l1',numMaxIters= 10000):
    featLblSin,featCmpSin = dataDict['featLblSin'],dataDict['featCmpSin']
    N,d = featLblSin.shape
    M,_ = featCmpSin.shape
    labelRSD, labelAbs,labelCmp = dataDict['labelRSD'],dataDict['labelAbs'], dataDict['labelCmp']
    labelAbs1Col = np.reshape(labelAbs,[-1,],order='F')
    labelCmp1Col = np.reshape(labelCmp, [-1, ], order='F')
    numOfExpLbl, numOfExpCmp = labelAbs.shape[1], labelCmp.shape[1]
    indexListTrLblSin,indexListTeLblSin=indexCVDict['indexLblSinTrain'],indexCVDict['indexLblSinTest']
    indexListCmpSin,indexListTeCmpSin=indexCVDict['indexCmpSinTrain'],indexCVDict['indexCmpSinTest']
    repeatTimes,K = indexCVDict['repeatTimes'],indexCVDict['K']
    betaMat = np.zeros([repeatTimes * K,d])
    constMat = np.zeros([repeatTimes * K,1])
    scoreLbl = np.zeros([N*numOfExpLbl,repeatTimes])
    scoreCmp = np.zeros([M*numOfExpCmp,repeatTimes])
    scoreRSD = np.zeros([N, repeatTimes])
    locCmp = np.zeros([M*numOfExpCmp,repeatTimes])
    aucLbl = np.zeros([1,repeatTimes])
    aucCmp = np.zeros([1, repeatTimes])
    aucRSD = np.zeros([1, repeatTimes])

    for repeatCount in range(repeatTimes):
        for foldCount in range(K):
            indexLblSinTrain = np.reshape(indexListTrLblSin[repeatCount][foldCount].copy(),[-1,])
            indexLblSinTest = np.reshape(indexListTeLblSin[repeatCount][foldCount].copy(),[-1,])
            indexCmpSinTrain = np.reshape(indexListCmpSin[repeatCount][foldCount].copy(),[-1,])
            indexCmpSinTest = np.reshape(indexListTeCmpSin[repeatCount][foldCount].copy(),[-1,])
            inputData = {'featLblSin':featLblSin,'featCmpSin':featCmpSin,
                         'labelRSD':labelRSD,'labelAbs':labelAbs,'labelCmp':labelCmp,
                         'indexLblSinTrain':indexLblSinTrain, 'indexLblSinTest':indexLblSinTest,
                         'indexCmpSinTrain':indexCmpSinTrain, 'indexCmpSinTest':indexCmpSinTest}
            beta, const, scoreLblTestSin, scoreCmpTestSin = TrTeGlobalModel(lossFunc,inputData,alpha,lamda)
            betaMat[K*repeatCount+foldCount,:] = beta[:]
            constMat[K*repeatCount+foldCount,:] = const
            for expLbl in range(numOfExpLbl):
                scoreLbl[expLbl*N+indexLblSinTest,repeatCount] = scoreLblTestSin[:,0]
            for expCmp in range(numOfExpCmp):
                scoreCmp[expCmp * M + indexCmpSinTest, repeatCount] = scoreCmpTestSin[:,0]
                locCmp[expCmp * M + indexCmpSinTest, repeatCount] = 1
        aucLbl[0,repeatCount] = metrics.roc_auc_score(labelAbs1Col, scoreLbl[:,repeatCount])
        scoreRSD[:, repeatCount] = scoreLbl[0:N, repeatCount]
        aucRSD[0, repeatCount] = metrics.roc_auc_score(labelRSD, scoreRSD[:,repeatCount])
        indexCmpValidTe = np.where(np.reshape(locCmp[:,repeatCount],[-1,])!=0)[0]
        aucCmp[0,repeatCount] = metrics.roc_auc_score(labelCmp1Col[indexCmpValidTe],scoreCmp[indexCmpValidTe,repeatCount])
    return betaMat, constMat, aucLbl, aucCmp, aucRSD, scoreRSD

def CVExpertBiasModel(lossFunc, dataDict, indexCVDict, alpha, lamda, penaltyTimes=100, penaltyType='l1',numMaxIters= 10000):

    featLblSin,featCmpSin = dataDict['featLblSin'],dataDict['featCmpSin']
    N,d = featLblSin.shape
    M,_ = featCmpSin.shape
    labelRSD, labelAbs,labelCmp = dataDict['labelRSD'],dataDict['labelAbs'], dataDict['labelCmp']
    labelAbs1Col = np.reshape(labelAbs,[-1,],order='F')
    labelCmp1Col = np.reshape(labelCmp, [-1, ], order='F')
    numOfExpLbl, numOfExpCmp = labelAbs.shape[1], labelCmp.shape[1]
    indexListTrLblSin,indexListTeLblSin=indexCVDict['indexLblSinTrain'],indexCVDict['indexLblSinTest']
    indexListCmpSin,indexListTeCmpSin=indexCVDict['indexCmpSinTrain'],indexCVDict['indexCmpSinTest']
    repeatTimes,K = indexCVDict['repeatTimes'],indexCVDict['K']
    betaMat = np.zeros([repeatTimes * K,d+numOfExpLbl])
    constMat = np.zeros([repeatTimes * K,1])
    scoreLbl = np.zeros([N*numOfExpLbl,repeatTimes])
    scoreCmp = np.zeros([M*numOfExpCmp,repeatTimes])
    scoreRSD = np.zeros([N,repeatTimes])
    locCmp = np.zeros([M*numOfExpCmp,repeatTimes])
    aucLbl = np.zeros([1,repeatTimes])
    aucCmp = np.zeros([1,repeatTimes])
    aucRSD = np.zeros([1,repeatTimes])

    for repeatCount in range(repeatTimes):
        for foldCount in range(K):
            indexLblSinTrain = np.reshape(indexListTrLblSin[repeatCount][foldCount].copy(),[-1,])
            indexLblSinTest = np.reshape(indexListTeLblSin[repeatCount][foldCount].copy(),[-1,])
            indexCmpSinTrain = np.reshape(indexListCmpSin[repeatCount][foldCount].copy(),[-1,])
            indexCmpSinTest = np.reshape(indexListTeCmpSin[repeatCount][foldCount].copy(),[-1,])
            inputData = {'featLblSin':featLblSin,'featCmpSin':featCmpSin,
                         'labelRSD':labelRSD,'labelAbs':labelAbs,'labelCmp':labelCmp,
                         'indexLblSinTrain':indexLblSinTrain, 'indexLblSinTest':indexLblSinTest,
                         'indexCmpSinTrain':indexCmpSinTrain, 'indexCmpSinTest':indexCmpSinTest,
                         'numOfExpLbl':numOfExpLbl,'numOfExpCmp':numOfExpCmp}
            beta, const, scoreLblTestList, scoreCmpTestSin = TrTeExpertBiasModel(lossFunc,inputData,alpha,lamda,penaltyTimes)
            betaMat[K*repeatCount+foldCount,:] = beta[:]
            constMat[K*repeatCount+foldCount,:] = const
            for expLbl in range(numOfExpLbl):
                scoreLbl[expLbl*N+indexLblSinTest,repeatCount] = scoreLblTestList[expLbl][:,0]
            scoreRSD[indexLblSinTest, repeatCount] = np.reshape(
                np.dot(featLblSin[indexLblSinTest, :], beta[0, 0:d].T) + const, [-1, ])
            for expCmp in range(numOfExpCmp):
                scoreCmp[expCmp * M + indexCmpSinTest, repeatCount] = scoreCmpTestSin[:,0]
                locCmp[expCmp * M + indexCmpSinTest, repeatCount] = 1
        aucLbl[0,repeatCount] = metrics.roc_auc_score(labelAbs1Col, scoreLbl[:,repeatCount])
        aucRSD[0,repeatCount] = metrics.roc_auc_score(labelRSD, scoreRSD[:,repeatCount])
        indexCmpValidTe = np.where(np.reshape(locCmp[:,repeatCount],[-1,])!=0)[0]
        aucCmp[0,repeatCount] = metrics.roc_auc_score(labelCmp1Col[indexCmpValidTe],scoreCmp[indexCmpValidTe,repeatCount])
    return betaMat, constMat, aucLbl, aucCmp, aucRSD, scoreRSD

def CVExpertModel(lossFunc,dataDict, indexCVDict, alpha, lamda, cmpExpOrder=(2, 6, 9, 10, 12)):
    # This function uses the cross validation method to validate the result on  each expert.
    featLblSin,featCmpSin = dataDict['featLblSin'],dataDict['featCmpSin']
    N,d = featLblSin.shape
    M,_ = featCmpSin.shape
    labelRSD, labelAbs,labelCmp = dataDict['labelRSD'],dataDict['labelAbs'], dataDict['labelCmp']
    labelAbs1Col = np.reshape(labelAbs, [-1, ], order='F')
    labelCmp1Col = np.reshape(labelCmp, [-1, ], order='F')
    numOfExpLbl, numOfExpCmp = labelAbs.shape[1], labelCmp.shape[1]
    indexListTrLblSin,indexListTeLblSin=indexCVDict['indexLblSinTrain'],indexCVDict['indexLblSinTest']
    indexListCmpSin,indexListTeCmpSin=indexCVDict['indexCmpSinTrain'],indexCVDict['indexCmpSinTest']
    repeatTimes,K = indexCVDict['repeatTimes'],indexCVDict['K']
    repeatTimes = 1
    betaMat = [np.zeros([repeatTimes * K,d])]*numOfExpLbl
    constMat = [np.zeros([repeatTimes * K,1])]*numOfExpLbl
    scoreLbl = np.zeros([numOfExpLbl*N,repeatTimes])
    scoreRSD = np.zeros([N,repeatTimes])
    scoreCmp = np.zeros([numOfExpCmp*M,repeatTimes]) # if the expert doenst have comparison, the score would keep zero.
    locCmp = np.zeros([numOfExpCmp*M,repeatTimes])
    aucLbl = np.zeros([1, repeatTimes])
    aucRSD = np.zeros([1, repeatTimes])
    aucCmp = np.zeros([1, repeatTimes])
    if alpha != 0.0:
        betaOutMat = np.zeros([numOfExpLbl, 1], dtype=np.object)
        constOutMat = np.zeros([numOfExpLbl, 1], dtype=np.object)
        for repeatCount in range(repeatTimes):
            for foldCount in range(K):
                indexLblSinTrain = np.reshape(indexListTrLblSin[repeatCount][foldCount].copy(),[-1,])
                indexLblSinTest = np.reshape(indexListTeLblSin[repeatCount][foldCount].copy(),[-1,])
                indexCmpSinTrain = np.reshape(indexListCmpSin[repeatCount][foldCount].copy(),[-1,])
                indexCmpSinTest = np.reshape(indexListTeCmpSin[repeatCount][foldCount].copy(),[-1,])
                inputData = {'featLblSin':featLblSin,'featCmpSin':featCmpSin,
                             'labelRSD':labelRSD,'labelAbs':labelAbs,'labelCmp':labelCmp,
                             'indexLblSinTrain':indexLblSinTrain, 'indexLblSinTest':indexLblSinTest,
                             'indexCmpSinTrain':indexCmpSinTrain, 'indexCmpSinTest':indexCmpSinTest,
                             'numOfExpLbl':numOfExpLbl,'numOfExpCmp':numOfExpCmp}
                betaList, constList, scoreLblTestList, scoreCmpTestList = TrTeExpertModel(lossFunc,inputData,alpha,lamda,cmpExpOrder)
                for exp in range(numOfExpLbl):
                    expInCmp = 0
                    betaMatTemp = betaMat[exp].copy()
                    constMatTemp = constMat[exp].copy()
                    betaMatTemp[K*repeatCount+foldCount,:] = betaList[exp].copy()
                    constMatTemp[K*repeatCount+foldCount,0] = 1 * constList[exp]
                    scoreLbl[N*exp+indexLblSinTest,repeatCount] = scoreLblTestList[exp][:,0].copy()
                    if exp in cmpExpOrder:
                        scoreCmp[M*expInCmp+indexCmpSinTest, repeatCount] = scoreCmpTestList[exp][:, 0].copy()
                        locCmp[M*expInCmp+indexCmpSinTest, repeatCount] = 1
                        expInCmp += 1
                    betaMat[exp] = betaMatTemp
                    constMat[exp] = constMatTemp
            scoreRSD[:, repeatCount]= np.mean(np.reshape(scoreLbl[:,repeatCount],[N,numOfExpLbl],'F'),1)
            aucRSD[0,repeatCount] = metrics.roc_auc_score(labelRSD,scoreRSD[:,repeatCount])
            aucLbl[0,repeatCount] = metrics.roc_auc_score(labelAbs1Col, scoreLbl[:,repeatCount])
            indexCmpValidTe = np.where(np.reshape(locCmp[:, repeatCount], [-1, ]) != 0)[0]
            aucCmp[0, repeatCount] = metrics.roc_auc_score(labelCmp1Col[indexCmpValidTe],
                                                           scoreCmp[indexCmpValidTe, repeatCount])

        for expOutput in range(numOfExpLbl):
            betaOutMat[expOutput, 0] = betaMat[expOutput].copy()
            constOutMat[expOutput, 0] = constMat[expOutput].copy()
    else:
        locLbl= np.zeros([numOfExpLbl * N, repeatTimes])
        betaOutMat = np.zeros([numOfExpLbl, 1], dtype=np.object)
        constOutMat = np.zeros([numOfExpLbl, 1], dtype=np.object)
        for repeatCount in range(repeatTimes):
            for foldCount in range(K):
                indexLblSinTrain = np.reshape(indexListTrLblSin[repeatCount][foldCount].copy(), [-1, ])
                indexLblSinTest = np.reshape(indexListTeLblSin[repeatCount][foldCount].copy(), [-1, ])
                indexCmpSinTrain = np.reshape(indexListCmpSin[repeatCount][foldCount].copy(), [-1, ])
                indexCmpSinTest = np.reshape(indexListTeCmpSin[repeatCount][foldCount].copy(), [-1, ])
                inputData = {'featLblSin': featLblSin, 'featCmpSin': featCmpSin,
                             'labelRSD': labelRSD, 'labelAbs': labelAbs, 'labelCmp': labelCmp,
                             'indexLblSinTrain': indexLblSinTrain, 'indexLblSinTest': indexLblSinTest,
                             'indexCmpSinTrain': indexCmpSinTrain, 'indexCmpSinTest': indexCmpSinTest,
                             'numOfExpLbl': numOfExpLbl, 'numOfExpCmp': numOfExpCmp}
                betaList, constList, scoreLblTestList, scoreCmpTestList = TrTeExpertModel(lossFunc,inputData, alpha, lamda,
                                                                                        cmpExpOrder)
                for exp in range(numOfExpCmp):
                    betaMatTemp = betaMat[exp].copy()
                    constMatTemp = constMat[exp].copy()
                    betaMatTemp[K * repeatCount + foldCount, :] = betaList[cmpExpOrder[exp]].copy()
                    constMatTemp[K * repeatCount + foldCount, 0] = 1 * constList[cmpExpOrder[exp]]
                    scoreLbl[N * cmpExpOrder[exp] + indexLblSinTest, repeatCount] = scoreLblTestList[cmpExpOrder[exp]][:, 0].copy()
                    locLbl[N * cmpExpOrder[exp] + indexLblSinTest, repeatCount]=1
                    scoreCmp[M * exp + indexCmpSinTest, repeatCount] = scoreCmpTestList[cmpExpOrder[exp]][:, 0].copy()
                    locCmp[M * exp + indexCmpSinTest, repeatCount] = 1
                    betaMat[cmpExpOrder[exp]] = betaMatTemp
                    constMat[cmpExpOrder[exp]] = constMatTemp
            indexLblValidTe = np.where(np.reshape(locLbl[:, repeatCount], [-1, ]) != 0)[0]
            aucLbl[0, repeatCount] = metrics.roc_auc_score(labelAbs1Col[indexLblValidTe], scoreLbl[indexLblValidTe, repeatCount])
            scoreRSD[:, repeatCount] = np.mean(np.reshape(scoreLbl[indexLblValidTe, repeatCount], [N, numOfExpCmp], 'F'),1)
            aucRSD[0, repeatCount] = metrics.roc_auc_score(labelRSD, scoreRSD[:, repeatCount])
            indexCmpValidTe = np.where(np.reshape(locCmp[:, repeatCount], [-1, ]) != 0)[0]
            aucCmp[0, repeatCount] = metrics.roc_auc_score(labelCmp1Col[indexCmpValidTe],
                                                           scoreCmp[indexCmpValidTe, repeatCount])

        for expOutput in range(numOfExpLbl):
            betaOutMat[expOutput, 0] = betaMat[expOutput].copy()
            constOutMat[expOutput, 0] = constMat[expOutput].copy()

    return betaOutMat, constOutMat, aucLbl, aucCmp, aucRSD, scoreRSD
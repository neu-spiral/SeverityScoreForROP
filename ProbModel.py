# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 15:22:58 2016

@author: PengTian
"""

import cvxpyMdl as cp
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics,linear_model,svm
import xlrd as xr

def LogisticReg(trainingData,lambd,yTrain,penType='l2',max_iters=10000):
    ## The result of own code for logistic Regression
    NTraining,d=trainingData.shape
    beta = cp.Variable(d)      
    loss = sum(cp.logistic((trainingData[i,:] * beta * -yTrain[i])) for i in range(NTraining))
    if penType == 'l1':
        penalty = lambd * (cp.norm(beta,1))
    elif penType == 'l2':
        penalty = lambd * (cp.power(cp.norm(beta,2),2))
    else:
        print 'Only l1 or l2 penalty are supported'
        exit()
    objective = cp.Minimize(loss+penalty)
    prob = cp.Problem(objective)
    res = prob.solve(verbose=False, max_iters=max_iters)
    betaOwn = beta.value
    return betaOwn
    
def KFoldLogReg(data,labels,K,lambd,numOfRep=1000,penType='l2',max_iters=10000):
    N,d = data.shape
    scoreTotal = np.zeros([N,numOfRep])
    for numT in range(numOfRep):
        skf = StratifiedKFold(np.reshape(labels,[labels.shape[0],]), n_folds=K, shuffle=True)
        CVindex=list(skf)
        scoreKFold = np.zeros([N,1])
        for numK in range(K):       
            trainingIndex = CVindex[numK][0]
            testingIndex = CVindex[numK][1]
            yTrain = labels[trainingIndex]
            trainingData = np.matrix(data[trainingIndex,:])
            testingData=np.matrix(data[testingIndex,:])
            beta = LogisticReg(trainingData,lambd,yTrain,penType,max_iters)
            # Testing Procedure
            scoreKFold[testingIndex] = testingData * beta
        scoreTotal[:,numT]=np.reshape(scoreKFold,[N,])
    score = np.mean(scoreTotal,axis=1)
    aucScore = metrics.roc_auc_score(labels,score)
    return score, aucScore , scoreTotal
    
def KFoldLogRegAuto(data,labels,K,c,numOfRep=1000,penType='l2',max_iters=10000):
    reminderRatio = 0.2
    N,d = data.shape
    scoreTotal = np.zeros([N,numOfRep])
    for numT in range(numOfRep):
        skf = StratifiedKFold(np.reshape(labels,[labels.shape[0],]), n_folds=K, shuffle=True)
        CVindex=list(skf)
        scoreKFold = np.zeros([N,1])
        for numK in range(K):       
            trainingIndex = CVindex[numK][0]
            testingIndex = CVindex[numK][1]
            yTrain = np.reshape(labels[trainingIndex],[-1,])
            trainingData = np.matrix(data[trainingIndex,:])
            testingData=np.matrix(data[testingIndex,:])
            if penType == 'l1':
                LRm = linear_model.LogisticRegression(C=c,penalty='l1',fit_intercept=False,max_iter=max_iters)
            elif penType == 'l2':
                LRm = linear_model.LogisticRegression(C=c,penalty='l2',fit_intercept=False,solver='newton-cg',max_iter=max_iters)
            else:
                print 'Only l1 or l2 penalty are supported'
                exit() 
            mdl = LRm.fit(trainingData,yTrain)
            beta = mdl.coef_
            scoreKFold[testingIndex] = testingData * beta.T
        scoreTotal[:,numT]=np.reshape(scoreKFold,[N,])
        if (numT+1) % (reminderRatio * numOfRep) == 0 :
            print '%.1f%% complete' % (float(numT+1)/float(numOfRep)*100)
    score = np.mean(scoreTotal,axis=1)
    aucScore = metrics.roc_auc_score(labels,score)
    return score, aucScore , scoreTotal
    
def KFoldSVMAuto(data,labels,K,c,numOfRep=1000,penType='l2',max_iters=10000):
    reminderRatio = 0.5
    N,d = data.shape
    scoreTotal = np.zeros([N,numOfRep])
    for numT in range(numOfRep):
        skf = StratifiedKFold(np.reshape(labels,[labels.shape[0],]), n_folds=K, shuffle=True)
        CVindex=list(skf)
        scoreKFold = np.zeros([N,1])
        for numK in range(K):       
            trainingIndex = CVindex[numK][0]
            testingIndex = CVindex[numK][1]
            yTrain = np.reshape(labels[trainingIndex],[-1,])
            trainingData = np.matrix(data[trainingIndex,:])
            testingData=np.matrix(data[testingIndex,:])
            if penType == 'l2':
                SVMmodel = svm.SVC(C=c,kernel='linear',max_iter=max_iters)
            else:
                print 'Only l2 penalty are supported'
                exit()
            mdl = SVMmodel.fit(trainingData,yTrain)
            beta = mdl.coef_
            scoreKFold[testingIndex] = testingData * beta.T
        scoreTotal[:,numT]=np.reshape(scoreKFold,[N,])
        if (numT+1) % (reminderRatio * numOfRep) == 0 :
            print '%.1f%% complete' % (float(numT+1)/float(numOfRep)*100)
    score = np.mean(scoreTotal,axis=1)
    aucScore = metrics.roc_auc_score(labels,score)
    return score, aucScore , scoreTotal  
    
def loadComparisonData(name):
    IdOrderFile = xr.open_workbook('../data/ropData/100Images.xlsx')
    IdorderSheet = IdOrderFile.sheet_by_name(u'ID&Order')
    Ids = IdorderSheet.col_values(0)
    del Ids[0]
    imageOrder = IdorderSheet.col_values(3)
    del imageOrder[0]
    IdOrder = dict(zip(Ids,imageOrder))
    temp = np.zeros((1,4))
    comparisonData = np.int_(temp)
    fCSV = open('../data/ropData/results_ICL_third_set_hundred_r2_compare.csv')
    for row in fCSV:
        if not name in row: continue
        content = row.split(',')
        number = content[0:4]
        number = [int(x) for x in number]
        comparisonData = np.vstack((comparisonData,np.reshape(number,[1,-1])))
    comparisonData=np.delete(comparisonData,0,0)
    comparisonData=np.delete(comparisonData,0,1)
    return IdOrder, comparisonData
    
    
    
def ComparisonOnly(feats,labels,comparisonData,IdOrder,featDiff,indexDiff,typeCrossValidation,c,numOfRep=100,K=5,num_iters=10000):
    N,d = feats.shape
    if typeCrossValidation==1:
        N,d = feats.shape
        M,_ = comparisonData.shape
        scoreComTotal = np.zeros([M,numOfRep*K])
        locTotal = np.zeros([M,numOfRep*K])
        labelCom =  comparisonData[:,2]
        for numT in range(numOfRep):
            skf = StratifiedKFold(np.reshape(labels,[labels.shape[0],]), n_folds=K, shuffle=True)
            CVindex=list(skf)
            scoreKFold = np.zeros([M,K])
            LocKFold = np. zeros([M,K])
            for numK in range(K):       
                trainLIndex = CVindex[numK][0]
                testinLIndex = CVindex[numK][1]
                comDiffTrainIndex = np.array([]) # feature index from the difference matrix(Feature)
                comTrainLabelIndex = np.array([]) # Label index from the comparison data matrix
                comDiffTestIndex = np.array([])
                comTestLabelIndex = np.array([])
                for i in range(M):
                    imgIndCi,imgIndCj = int(IdOrder[float(comparisonData[i,0])]),int(IdOrder[float(comparisonData[i,1])])
                    featDiffIndexCurrent = np.where(indexDiff==N*imgIndCi+imgIndCj)[0]
                    if imgIndCi in trainLIndex and imgIndCj in trainLIndex:
                        comDiffTrainIndex = np.append(comDiffTrainIndex,featDiffIndexCurrent)
                        comTrainLabelIndex = np.append(comTrainLabelIndex,i)
                    else:
                        comDiffTestIndex = np.append(comDiffTestIndex,featDiffIndexCurrent)
                        comTestLabelIndex = np.append(comTestLabelIndex,i)
                comDiffTrainIndex,comTrainLabelIndex=np.int_(comDiffTrainIndex),np.int_(comTrainLabelIndex)
                comDiffTestIndex ,comTestLabelIndex= np.int_(comDiffTestIndex),np.int_(comTestLabelIndex)
                yTrain = np.reshape(comparisonData[comTrainLabelIndex,2],[-1,])
                trainingData = featDiff[comDiffTrainIndex,:]
                testingData = featDiff[comDiffTestIndex,:]
                LRm = linear_model.LogisticRegression(C=c,penalty='l2',fit_intercept=False,solver='newton-cg',max_iter=num_iters)
                mdl = LRm.fit(trainingData,yTrain)
                beta = np.matrix(mdl.coef_)
                scoreKFold[comTestLabelIndex,numK] = np.reshape(testingData * beta.T,[-1,])
                LocKFold[comTestLabelIndex,numK] = 1
            scoreComTotal[:,K*numT:(numT+1)*K]=scoreKFold
            locTotal[:,K*numT:(numT+1)*K] = LocKFold
            if (numT+1) % (reminderRatio * numOfRep) == 0 :
                print '%.1f%% complete' % (float(numT+1)/float(numOfRep)*100)
        numOfSampleTest = np.sum(locTotal,axis=1)
        validTestIndex = np.where(np.reshape(numOfSampleTest,[-1,])!=0)
        scoreSum = np.sum(scoreComTotal, axis =1)
        score = np.divide(scoreSum[validTestIndex],numOfSampleTest[validTestIndex])
        aucScore = metrics.roc_auc_score(labelCom[validTestIndex],score)
        
    elif typeCrossValidation==2:
        N,d = feats.shape
        M,_ = comparisonData.shape
        scoreComTotal = np.zeros([M,numOfRep*K])
        locTotal = np.zeros([M,numOfRep*K])
        labelCom =  comparisonData[:,2]
        for numT in range(numOfRep):
            skf = StratifiedKFold(np.reshape(labels,[labels.shape[0],]), n_folds=K, shuffle=True)
            CVindex=list(skf)
            scoreKFold = np.zeros([M,K])
            LocKFold = np. zeros([M,K])
            for numK in range(K):       
                trainLIndex = CVindex[numK][0]
                testinLIndex = CVindex[numK][1]
                comDiffTrainIndex = np.array([]) # feature index from the difference matrix(Feature)
                comTrainLabelIndex = np.array([]) # Label index from the comparison data matrix
                comDiffTestIndex = np.array([])
                comTestLabelIndex = np.array([])
                for i in range(M):
                    imgIndCi,imgIndCj = int(IdOrder[float(comparisonData[i,0])]),int(IdOrder[float(comparisonData[i,1])])
                    featDiffIndexCurrent = np.where(indexDiff==N*imgIndCi+imgIndCj)[0]
                    if imgIndCi in trainLIndex or imgIndCj in trainLIndex:
                        comDiffTrainIndex = np.append(comDiffTrainIndex,featDiffIndexCurrent)
                        comTrainLabelIndex = np.append(comTrainLabelIndex,i)
                    else:
                        comDiffTestIndex = np.append(comDiffTestIndex,featDiffIndexCurrent)
                        comTestLabelIndex = np.append(comTestLabelIndex,i)
                comDiffTrainIndex,comTrainLabelIndex=np.int_(comDiffTrainIndex),np.int_(comTrainLabelIndex)
                comDiffTestIndex ,comTestLabelIndex= np.int_(comDiffTestIndex),np.int_(comTestLabelIndex)
                yTrain = np.reshape(comparisonData[comTrainLabelIndex,2],[-1,])
                trainingData = featDiff[comDiffTrainIndex,:]
                testingData = featDiff[comDiffTestIndex,:]
                LRm = linear_model.LogisticRegression(C=c,penalty='l2',fit_intercept=False,solver='newton-cg',max_iter=num_iters)
                mdl = LRm.fit(trainingData,yTrain)
                beta = np.matrix(mdl.coef_)
                scoreKFold[comTestLabelIndex,numK] = np.reshape(testingData * beta.T,[-1,])
                LocKFold[comTestLabelIndex,numK] = 1
            scoreComTotal[:,K*numT:(numT+1)*K]=scoreKFold
            locTotal[:,K*numT:(numT+1)*K] = LocKFold
#            if (numT+1) % (reminderRatio * numOfRep) == 0 :
#                print '%.1f%% complete' % (float(numT+1)/float(numOfRep)*100)
        numOfSampleTest = np.sum(locTotal,axis=1)
        validTestIndex = np.where(np.reshape(numOfSampleTest,[-1,])!=0)
        scoreSum = np.sum(scoreComTotal, axis =1)
        score = np.divide(scoreSum[validTestIndex],numOfSampleTest[validTestIndex])
        aucScore = metrics.roc_auc_score(labelCom[validTestIndex],score)
        
        
    elif typeCrossValidation==3:
        M,_ = comparisonData.shape
        labelCom = comparisonData[:,2]
        scoreComTotal = np.zeros([M,numOfRep])
        for numT in range(numOfRep):
            skf = StratifiedKFold(np.reshape(comparisonData[:,2],[comparisonData[:,2].shape[0],]), n_folds=K, shuffle=True)
            CVindex=list(skf)
            scoreKFold = np.zeros([M,1])
            for numK in range(K):       
                trainingCIndex = CVindex[numK][0]
                NTrain = len(trainingCIndex)
                testingCIndex = CVindex[numK][1]
                NTest = len(testingCIndex)
                comTrainFeats = np.zeros([NTrain,d])
                comTestFeats =  np.zeros([NTest,d])
                for i in range(NTrain):
                    imgIndCi,imgIndCj = int(IdOrder[float(comparisonData[trainingCIndex[i],0])])-1,int(IdOrder[float(comparisonData[trainingCIndex[i],1])])-1
                    comTrainFeats[i,:] = feats[imgIndCi]-feats[imgIndCj]
                for i in range(NTest):
                    imgIndCi,imgIndCj = int(IdOrder[float(comparisonData[testingCIndex[i],0])])-1,int(IdOrder[float(comparisonData[testingCIndex[i],1])])-1
                    comTestFeats[i,:] = feats[imgIndCi]-feats[imgIndCj] 
                comTrainLabels=comparisonData[trainingCIndex,2]
                yTrain = np.reshape(comTrainLabels,[-1,])
                trainingData = comTrainFeats
                testingData = comTestFeats
                LRm = linear_model.LogisticRegression(C=c,penalty='l2',fit_intercept=False,solver='newton-cg',max_iter=num_iters)
                mdl = LRm.fit(trainingData,yTrain)
                beta = np.array(mdl.coef_)
                scoreKFold[testingCIndex,:] = np.dot(comTestFeats , beta.T)
                if (numT+1) % (reminderRatio * numOfRep) == 0 :
                    print '%.1f%% complete' % (float(numT+1)/float(numOfRep)*100)
            scoreComTotal[:,numT]=np.reshape(scoreKFold,[M,])
        score = np.mean(scoreComTotal,axis=1)
        aucScore = metrics.roc_auc_score(labelCom,score)
    else : 
        print('CrossValidation Type Error') 
        exit()
    return scoreComTotal,score,aucScore
    
def LogRegLTrainCTest(feats,labels,comparisonData,IdOrder,featDiff,indexDiff,K,c,numOfRep=50,penType='l2',num_iters=10000):
        N,d = feats.shape
        M,_ = comparisonData.shape
        scoreComTotal = np.zeros([M,numOfRep*K])
        scoreLTotal = np.zeros([N,numOfRep])
        locCTotal = np.zeros([M,numOfRep*K])
        labelCom =  comparisonData[:,2]
        for numT in range(numOfRep):
            skf = StratifiedKFold(np.reshape(labels,[labels.shape[0],]), n_folds=K, shuffle=True)
            CVindex=list(skf)
            scoreCKFold = np.zeros([M,K])
            scoreLKFold = np.zeros([N,1])
            LocCKFold = np. zeros([M,K])
            for numK in range(K):       
                trainLIndex = CVindex[numK][0]
                testinLIndex = CVindex[numK][1]
                comDiffTrainIndex = np.array([]) # feature index from the difference matrix(Feature)
                comTrainLabelIndex = np.array([]) # Label index from the comparison data matrix
                comDiffTestIndex = np.array([])
                comTestLabelIndex = np.array([])
                for i in range(M):
                    imgIndCi,imgIndCj = int(IdOrder[float(comparisonData[i,0])]),int(IdOrder[float(comparisonData[i,1])])
                    featDiffIndexCurrent = np.where(indexDiff==N*imgIndCi+imgIndCj)[0]
                    if imgIndCi in trainLIndex and imgIndCj in trainLIndex:
                        comDiffTrainIndex = np.append(comDiffTrainIndex,featDiffIndexCurrent)
                        comTrainLabelIndex = np.append(comTrainLabelIndex,i)
                    else:
                        comDiffTestIndex = np.append(comDiffTestIndex,featDiffIndexCurrent)
                        comTestLabelIndex = np.append(comTestLabelIndex,i)
                comDiffTrainIndex,comTrainLabelIndex=np.int_(comDiffTrainIndex),np.int_(comTrainLabelIndex)
                comDiffTestIndex ,comTestLabelIndex= np.int_(comDiffTestIndex),np.int_(comTestLabelIndex)
#                yTrainC = np.reshape(comparisonData[comTrainLabelIndex,2],[-1,])
                yTrainL = np.reshape(labels[trainLIndex],[-1,])
#                yTrain = np.append(yTrainL,yTrainC)
#                trainingDataC = featDiff[comDiffTrainIndex,:]
                testingDataC = featDiff[comDiffTestIndex,:]
                trainingDataL = feats[trainLIndex,:]
                testingDataL = feats[testinLIndex,:]
#                trainingData = np.concatenate((trainingDataL,trainingDataC),axis=0)
                LRm = linear_model.LogisticRegression(C=c,penalty='l2',fit_intercept=False,solver='newton-cg',max_iter=num_iters)
                mdl = LRm.fit(trainingDataL,yTrainL)
                beta = np.matrix(mdl.coef_)
                scoreCKFold[comTestLabelIndex,numK] = np.reshape(testingDataC * beta.T,[-1,])
                scoreLKFold[testinLIndex,0]= np.reshape(testingDataL*beta.T,[-1,])
                LocCKFold[comTestLabelIndex,numK] = 1
            scoreComTotal[:,K*numT:(numT+1)*K]=scoreCKFold
            locCTotal[:,K*numT:(numT+1)*K] = LocCKFold
            scoreLTotal[:,numT]=np.reshape(scoreLKFold,[-1,])
        numOfSampleTest = np.sum(locCTotal,axis=1)
        validTestIndex = np.where(np.reshape(numOfSampleTest,[-1,])!=0)
        scoreSumC = np.sum(scoreComTotal, axis =1)
        scoreC = np.divide(scoreSumC[validTestIndex],numOfSampleTest[validTestIndex])
        aucScoreC = metrics.roc_auc_score(labelCom[validTestIndex],scoreC)
        scoreL = np.mean(scoreLTotal,axis=1)
        aucScoreL = metrics.roc_auc_score(labels,scoreL)
        return scoreL, aucScoreL,scoreLTotal,  scoreC, aucScoreC, scoreComTotal
    
def ComparisonLabelCombine(feats,labels,comparisonData,IdOrder,featDiff,indexDiff,typeCrossValidation,c,numOfRep=10,K=5,num_iters=10000):
    N,d = feats.shape
    if typeCrossValidation==1:
        N,d = feats.shape
        M,_ = comparisonData.shape
        scoreComTotal = np.zeros([M,numOfRep*K])
        scoreLTotal = np.zeros([N,numOfRep])
        locCTotal = np.zeros([M,numOfRep*K])
        labelCom =  comparisonData[:,2]
        for numT in range(numOfRep):
            skf = StratifiedKFold(np.reshape(labels,[labels.shape[0],]), n_folds=K, shuffle=True)
            CVindex=list(skf)
            scoreCKFold = np.zeros([M,K])
            scoreLKFold = np.zeros([N,1])
            LocCKFold = np. zeros([M,K])
            for numK in range(K):       
                trainLIndex = CVindex[numK][0]
                testinLIndex = CVindex[numK][1]
                comDiffTrainIndex = np.array([]) # feature index from the difference matrix(Feature)
                comTrainLabelIndex = np.array([]) # Label index from the comparison data matrix
                comDiffTestIndex = np.array([])
                comTestLabelIndex = np.array([])
                for i in range(M):
                    imgIndCi,imgIndCj = int(IdOrder[float(comparisonData[i,0])]),int(IdOrder[float(comparisonData[i,1])])
                    featDiffIndexCurrent = np.where(indexDiff==N*imgIndCi+imgIndCj)[0]
                    if imgIndCi-1 in trainLIndex and imgIndCj-1 in trainLIndex:
                        comDiffTrainIndex = np.append(comDiffTrainIndex,featDiffIndexCurrent)
                        comTrainLabelIndex = np.append(comTrainLabelIndex,i)
                    else:
                        comDiffTestIndex = np.append(comDiffTestIndex,featDiffIndexCurrent)
                        comTestLabelIndex = np.append(comTestLabelIndex,i)
                comDiffTrainIndex,comTrainLabelIndex=np.int_(comDiffTrainIndex),np.int_(comTrainLabelIndex)
                comDiffTestIndex ,comTestLabelIndex= np.int_(comDiffTestIndex),np.int_(comTestLabelIndex)
                yTrainC = np.reshape(comparisonData[comTrainLabelIndex,2],[-1,])
                yTrainL = np.reshape(labels[trainLIndex],[-1,])
                yTrain = np.append(yTrainL,yTrainC)
                trainingDataC = featDiff[comDiffTrainIndex,:]
                testingDataC = featDiff[comDiffTestIndex,:]
                trainingDataL = feats[trainLIndex,:]
                testingDataL = feats[testinLIndex,:]
                trainingData = np.concatenate((trainingDataL,trainingDataC),axis=0)
                LRm = linear_model.LogisticRegression(C=c,penalty='l2',fit_intercept=False,solver='newton-cg',max_iter=num_iters)
                mdl = LRm.fit(trainingData,yTrain)
                beta = np.matrix(mdl.coef_)
                scoreCKFold[comTestLabelIndex,numK] = np.reshape(testingDataC * beta.T,[-1,])
                scoreLKFold[testinLIndex,0]= np.reshape(testingDataL*beta.T,[-1,])
                LocCKFold[comTestLabelIndex,numK] = 1
            scoreComTotal[:,K*numT:(numT+1)*K]=scoreCKFold
            locCTotal[:,K*numT:(numT+1)*K] = LocCKFold
            scoreLTotal[:,numT]=np.reshape(scoreLKFold,[-1,])
        numOfSampleTest = np.sum(locCTotal,axis=1)
        validTestIndex = np.where(np.reshape(numOfSampleTest,[-1,])!=0)
        scoreSumC = np.sum(scoreComTotal, axis =1)
        scoreC = np.divide(scoreSumC[validTestIndex],numOfSampleTest[validTestIndex])
        aucScoreC = metrics.roc_auc_score(labelCom[validTestIndex],scoreC)
        scoreL = np.mean(scoreLTotal,axis=1)
        aucScoreL = metrics.roc_auc_score(labels,scoreL)
#        return scoreComTotal, scoreC, aucScoreC, scoreLTotal, scoreL, aucScoreL
        
    elif typeCrossValidation==2:
        N,d = feats.shape
        M,_ = comparisonData.shape
        scoreComTotal = np.zeros([M,numOfRep*K])
        scoreLTotal = np.zeros([N,numOfRep])
        locCTotal = np.zeros([M,numOfRep*K])
        labelCom =  comparisonData[:,2]
        for numT in range(numOfRep):
            skf = StratifiedKFold(np.reshape(labels,[labels.shape[0],]), n_folds=K, shuffle=True)
            CVindex=list(skf)
            scoreCKFold = np.zeros([M,K])
            scoreLKFold = np.zeros([N,1])
            LocCKFold = np. zeros([M,K])
            for numK in range(K):       
                trainLIndex = CVindex[numK][0]
                testinLIndex = CVindex[numK][1]
                comDiffTrainIndex = np.array([]) # feature index from the difference matrix(Feature)
                comTrainLabelIndex = np.array([]) # Label index from the comparison data matrix
                comDiffTestIndex = np.array([])
                comTestLabelIndex = np.array([])
                for i in range(M):
                    imgIndCi,imgIndCj = int(IdOrder[float(comparisonData[i,0])]),int(IdOrder[float(comparisonData[i,1])])
                    featDiffIndexCurrent = np.where(indexDiff==N*imgIndCi+imgIndCj)[0]
                    if imgIndCi in trainLIndex or imgIndCj in trainLIndex:
                        comDiffTrainIndex = np.append(comDiffTrainIndex,featDiffIndexCurrent)
                        comTrainLabelIndex = np.append(comTrainLabelIndex,i)
                    else:
                        comDiffTestIndex = np.append(comDiffTestIndex,featDiffIndexCurrent)
                        comTestLabelIndex = np.append(comTestLabelIndex,i)
                comDiffTrainIndex,comTrainLabelIndex=np.int_(comDiffTrainIndex),np.int_(comTrainLabelIndex)
                comDiffTestIndex ,comTestLabelIndex= np.int_(comDiffTestIndex),np.int_(comTestLabelIndex)
                yTrainC = np.reshape(comparisonData[comTrainLabelIndex,2],[-1,])
                yTrainL = np.reshape(labels[trainLIndex],[-1,])
                yTrain = np.append(yTrainL,yTrainC)
                trainingDataC = featDiff[comDiffTrainIndex,:]
                testingDataC = featDiff[comDiffTestIndex,:]
                trainingDataL = feats[trainLIndex,:]
                testingDataL = feats[testinLIndex,:]
                trainingData = np.concatenate((trainingDataL,trainingDataC),axis=0)
                LRm = linear_model.LogisticRegression(C=c,penalty='l2',fit_intercept=False,solver='newton-cg',max_iter=num_iters)
                mdl = LRm.fit(trainingData,yTrain)
                beta = np.matrix(mdl.coef_)
                scoreCKFold[comTestLabelIndex,numK] = np.reshape(testingDataC * beta.T,[-1,])
                scoreLKFold[testinLIndex,0]= np.reshape(testingDataL*beta.T,[-1,])
                LocCKFold[comTestLabelIndex,numK] = 1
            scoreComTotal[:,K*numT:(numT+1)*K]=scoreCKFold
            locCTotal[:,K*numT:(numT+1)*K] = LocCKFold
            scoreLTotal[:,numT]=np.reshape(scoreLKFold,[-1,])
        numOfSampleTest = np.sum(locCTotal,axis=1)
        validTestIndex = np.where(np.reshape(numOfSampleTest,[-1,])!=0)
        scoreSumC = np.sum(scoreComTotal, axis =1)
        scoreC = np.divide(scoreSumC[validTestIndex],numOfSampleTest[validTestIndex])
        aucScoreC = metrics.roc_auc_score(labelCom[validTestIndex],scoreC)
        scoreL = np.mean(scoreLTotal,axis=1)
        aucScoreL = metrics.roc_auc_score(labels,scoreL)
        
        
        
    elif typeCrossValidation==3:
        M,_ = comparisonData.shape
        labelCom = comparisonData[:,2]
        scoreComTotal = np.zeros([M,numOfRep])
        scoreLTotal = np.zeros([N,numOfRep])
        for numT in range(numOfRep):
            skfC = StratifiedKFold(np.reshape(comparisonData[:,2],[comparisonData[:,2].shape[0],]), n_folds=K, shuffle=True)
            skfL = StratifiedKFold(np.reshape(labels,[labels.shape[0],]), n_folds=K, shuffle=True)
            CVindexC=list(skfC)
            CVindexL=list(skfL)
            scoreCKFold = np.zeros([M,1])
            scoreLKFold = np.zeros([N,1])
            for numK in range(K):       
                trainingCIndex = CVindexC[numK][0]
                NTrain = len(trainingCIndex)
                testingCIndex = CVindexC[numK][1]
                NTest = len(testingCIndex)
                comTrainFeats = np.zeros([NTrain,d])
                comTestFeats =  np.zeros([NTest,d])
                trainLIndex = CVindexL[numK][0]
                testinLIndex = CVindexL[numK][1]
                for i in range(NTrain):
                    imgIndCi,imgIndCj = int(IdOrder[float(comparisonData[trainingCIndex[i],0])])-1,int(IdOrder[float(comparisonData[trainingCIndex[i],1])])-1
                    comTrainFeats[i,:] = feats[imgIndCi]-feats[imgIndCj]
                for i in range(NTest):
                    imgIndCi,imgIndCj = int(IdOrder[float(comparisonData[testingCIndex[i],0])])-1,int(IdOrder[float(comparisonData[testingCIndex[i],1])])-1
                    comTestFeats[i,:] = feats[imgIndCi]-feats[imgIndCj] 
                comTrainLabels=comparisonData[trainingCIndex,2]
                yTrainC = np.reshape(comTrainLabels,[-1,])
                yTrainL = np.reshape(labels[trainLIndex],[-1,])
                yTrain = np.append(yTrainL,yTrainC)
                trainingDataC = comTrainFeats
                testingDataC = comTestFeats
                trainingDataL = feats[trainLIndex,:]
                testingDataL = feats[testinLIndex,:]
                trainingData = np.concatenate((trainingDataL,trainingDataC),axis=0)
                LRm = linear_model.LogisticRegression(C=c,penalty='l2',fit_intercept=False,solver='newton-cg',max_iter=num_iters)
                mdl = LRm.fit(trainingData,yTrain)
                beta = np.array(mdl.coef_)
                scoreCKFold[testingCIndex,:] = np.dot(comTestFeats , beta.T)
                scoreLKFold[testinLIndex,:]= np.dot(testingDataL, beta.T)
            scoreComTotal[:,numT]=np.reshape(scoreCKFold,[M,])
            scoreLTotal[:,numT]=np.reshape(scoreLKFold,[-1,])
        scoreC = np.mean(scoreComTotal,axis=1)
        aucScoreC = metrics.roc_auc_score(labelCom,scoreC)
        scoreL = np.mean(scoreLTotal,axis=1)
        aucScoreL = metrics.roc_auc_score(labels,scoreL)
    else : 
        print('CrossValidation Type Error') 
        exit()
    return  scoreL, aucScoreL,scoreLTotal,  scoreC, aucScoreC, scoreComTotal

    
    
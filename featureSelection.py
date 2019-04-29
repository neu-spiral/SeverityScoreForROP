"""
This script is to training all the data based on the given alpha and lambda value to select the feature on modelSelectionOnRSDL1.py.
6 figures included: 1,2)Predicting RSD at Plus vs Not Plus, Not Normal vs Normal.
3,4)Predicting Expert bias at Plus vs Not Plus, Not Normal vs Normal.
5,6)Predicting comparison label at Plus vs Not Plus, Not Normal vs Normal.

Parameters :
-------------
lambdWeight: scalar
    contains the weights on L1 penalty parameter
alphaWeight: scalar
    contains the weights on label data. The weights on comparison data would be (1-alphaWeight). 0 - train label data only, 1 - train comparison data only.
dataType : 'auto' or 'manual'
    the type of data is using. 'auto' - automatic segmented image features. 'manual' manugal segemented image features
-------------


Author : Peng Tian
Date : November 2016

"""


from sklearn import metrics
import scipy.io as sio
import numpy as np
from scipy.io import loadmat, savemat
from sklearn import linear_model
from cvxOpt import Log_Log
from cvxpyMdl import Log_SVM
import matplotlib.pyplot as plt
import seaborn as sns
import csv

def featurePlot(beta,figName,title,featsNameList,d=156,numOfFeatsShow=20,ylim=[-20, 70],xlabel=None,ylabel=None):
    # This function is to produce the figure and CSV file list the best numOfFeatsShow features with a bar plot.
    # figure's x axis is the index of feature and in the csv file has the corresponding feature names.
    beta = np.array(beta)
    betaNoB = beta[0:d]
    print np.count_nonzero(betaNoB)
    betaNoBInd = np.flipud(np.argsort(np.absolute(betaNoB), axis=0)[:])
    betaShow = betaNoB[betaNoBInd]
    x = range(0, 20)
    fig = plt.figure()
    index = [i for i in betaNoBInd[:numOfFeatsShow, 0]]
    plt.xticks(x, [i+1 for i in betaNoBInd[:numOfFeatsShow, 0]], fontsize=10)
    plt.bar(x, betaNoB[index, 0], width=0.7)
    if ylabel ==None :
        plt.ylabel('Value of Expert with bias Model Parameter')
    else:
        plt.ylabel(ylabel)
    if xlabel == None :
        plt.xlabel('Feature Index')
    else:
        plt.xlabel(xlabel)
    plt.title(title)
    axes = plt.gca()
    axes.set_ylim(ylim)
    plt.savefig(figName, bbox_inches='tight')
    outputCSV = open(title+'.csv', 'wt')
    writerCSV = csv.writer(outputCSV)
    writerCSV.writerow(('Feature Index', 'Feature Name' ))
    for featCount in range(numOfFeatsShow):
        writerCSV.writerow(( betaNoBInd[featCount,0]+1, featsNameList[betaNoBInd[featCount,0]]))


# main script
numOfFeatsShow = 20  # Top number of features shows up.
# dataType = 'manual'
dataType = 'other'
nameBase = '../../../Data/figure/'
loadDataBase = '../../../Data/Result/L1Weights/RSD&Bias2RSD&Exp13_L1_CV1_NWeights_manual_'
dataFile = loadmat('../../../data/ropData/iROP_6DD_1st100_Partition.mat')
featsNameFilePath = '../../../Data/FeatureExtraction/Manual/SetOf100/featsName.mat'
featsNameFile = loadmat(featsNameFilePath)
featsNameOrigin = featsNameFile['featsName']
featsNameOrigin = featsNameOrigin[1:,:]
featsN, stats = featsNameOrigin.shape
featsName = [None]*featsN
for i in range(featsN):
    featsName[i] = (featsNameOrigin[i,1]).tostring() + '_' + (featsNameOrigin[i,2]).tostring()
    featsName[i] = featsName[i].replace("\x00","")

numOfExpts4Lbl = 13
numOfExpts4Cmp = 5
penaltyTimes = 100
labelPlusSet = dataFile['labelPlus']
labelPrePSet = dataFile['labelPreP']
cmpLabel = dataFile['cmpLabel']
Yc = dataFile['cmpLabel1Column'][0,:]

featFile = loadmat('../../../Data/pythonFeat/py1st100Full6DDCmp.mat')

if dataType == 'manual':
    labelFeatOrigin = dataFile['labelFeatManual']
    cmpFeatOrigin = dataFile['cmpFeatManual']
elif dataType == 'auto':
    labelFeatOrigin = dataFile['labelFeatAuto']
    cmpFeatOrigin = dataFile['cmpFeatAuto']
elif dataType == 'other':
    labelFeatOrigin = dataFile['labelFeatAuto']
    cmpFeatOrigin = dataFile['cmpFeatAuto']
else:
    assert('dataType should be manual or auto')
N, d = labelFeatOrigin.shape
M, _ = cmpFeatOrigin.shape
# labelFeat = np.concatenate((labelFeatOrigin,penaltyTimes*np.ones([N,1])),axis=1)
labelFeat = 1 * labelFeatOrigin
# cmpFeat = np.concatenate((cmpFeatOrigin,np.zeros([M,1])),axis=1)
cmpFeat = 1 * cmpFeatOrigin
YlPlus = np.reshape(labelPlusSet[:, :-1], [-1, ], order='F')
YlRSDPlus = labelPlusSet[:, 13]
YlPreP = np.reshape(labelPrePSet[:, :-1], [-1, ], order='F')
YlRSDPreP = labelPrePSet[:, 13]
Ntol = N * numOfExpts4Lbl
Mtol = M * numOfExpts4Cmp
trainFeatC = 1 * cmpFeat
trainFeatL = 1 * labelFeat


lenTrainL = 1 * N
# Prepare 13 Experts with experts bias training and testing feats labels
YtrainExp13LPlus = np.array([])
YtrainExp13LPreP = np.array([])
YtrainExp13C = np.array([])
XtrainExprtsFeat = np.zeros([numOfExpts4Lbl * lenTrainL, numOfExpts4Lbl])
for eL in range(numOfExpts4Lbl):
    YtrainExp13LPlus = np.append(YtrainExp13LPlus, labelPlusSet[:, eL])
    YtrainExp13LPreP = np.append(YtrainExp13LPreP, labelPrePSet[:, eL])
    XtrainExprtsFeat[eL * lenTrainL: (eL + 1) * lenTrainL, eL] = penaltyTimes
for eC in range(numOfExpts4Cmp):
    YtrainExp13C = np.append(YtrainExp13C, cmpLabel[:, eC])
XtrainExp13Lo = np.tile(trainFeatL, [numOfExpts4Lbl, 1])
XtrainExp13L = np.concatenate([XtrainExp13Lo, XtrainExprtsFeat], axis=1)
XtrainExp13Co = np.tile(trainFeatC, [numOfExpts4Cmp, 1])
XtrainExp13C = np.concatenate([XtrainExp13Co, np.zeros([XtrainExp13Co.shape[0], numOfExpts4Lbl])], axis=1)
NlExp13, NcExp13 = len(YtrainExp13LPlus), len(YtrainExp13C)
XtrainExp13 = np.concatenate((XtrainExp13L, XtrainExp13C), axis=0)
YtrainExp13 = np.append(YtrainExp13LPlus, YtrainExp13C)
# weightsExp13 = np.concatenate((alpha  * (NlExp13 + NcExp13) * np.ones([NlExp13, ]),(1 - alpha)  * (NlExp13 + NcExp13) * np.ones([NcExp13, ])), axis=0)

# Prepare RSD training adn testing feats labels
XtrainRSDL = 1 * trainFeatL
XtrainRSDC = np.tile(trainFeatC, [numOfExpts4Cmp, 1])
YtrainRSDLPlus = 1 * YlRSDPlus[:]
YtrainRSDLPreP = 1 * YlRSDPreP[:]
YtrainRSDC = 1 * YtrainExp13C
NlRSD, NcRSD = len(YtrainRSDLPlus), len(YtrainRSDLPlus) # Plus and PreP have the same number of training samples
XtrainRSD = np.concatenate((XtrainRSDL, XtrainRSDC), axis=0)
YtrainRSDPlus = np.append(YtrainRSDLPlus, YtrainRSDC)
YtrainRSDPreP = np.append(YtrainRSDLPreP, YtrainRSDC)





# 1 Plus Predict RSD Here training Exp13Bias
alphaExp132RSDPlus = 1.0
lambdaExp132RSDPlus = 0.1
betaExp132RSDPlus, constExp132RSDPlus = Log_Log(XtrainExp13L,YtrainExp13LPlus,XtrainExp13C, YtrainExp13C,alphaExp132RSDPlus,lambdaExp132RSDPlus)
betaExp132RSDPlus = np.array(betaExp132RSDPlus[0:d])
# featurePlot(betaExp132RSDPlus,nameBase+'featureSelectionNormalizedRSDPlus.pdf', 'Predicting RSD Label Plus vs Not Plus',featsName)

## 2 PreP Predict RSD Here training Exp13Bias
alphaExp132RSDPreP = 1.0
lambdaExp132RSDPreP = 0.3
betaExp132RSDPreP, constExp132RSDPreP = Log_SVM(XtrainExp13L,YtrainExp13LPreP,XtrainExp13C, YtrainExp13C,alphaExp132RSDPreP,lambdaExp132RSDPreP)
betaExp132RSDPreP = np.array(betaExp132RSDPreP[0:d])
# featurePlot(betaExp132RSDPreP,nameBase+'featureSelectionNormalizedRSDPreP.pdf', 'Predicting RSD Label Not Normal vs Normal',featsName)

outputDict = {'RSDPlus':[betaExp132RSDPlus,constExp132RSDPlus], 'RSDPreP':[betaExp132RSDPreP,constExp132RSDPreP]}
savemat(dataType+'Beta.mat',outputDict)

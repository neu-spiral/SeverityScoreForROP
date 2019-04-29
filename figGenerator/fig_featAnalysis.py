
"""
This script is to draw the feature analysis figure;

Input:
		- fileFeat: load the beta file in data folder.
		- figName: string denoting the saved figure name.
		- title, xlabel, ylabel: are 3 strings denoting the title, xlabel, ylabel of the figure.
Author : Peng Tian
Date : Feb 2017
"""



from sklearn import metrics
import scipy.io as sio
import numpy as np
from scipy.io import loadmat, savemat


import matplotlib.pyplot as plt
import seaborn as sns
import csv


from sklearn import metrics
import scipy.io as sio
import numpy as np
from scipy.io import loadmat, savemat


import matplotlib.pyplot as plt
import seaborn as sns
import csv



fileFeat = loadmat('../../data/featAnalysis/BestScoreBeta.mat')
beta = np.array(fileFeat['beta'])
figName='../../pic/featAnaly.pdf'
featsNameFilePath = '../../data/featAnalysis/featsName.mat'
featsNameFile = loadmat(featsNameFilePath)
featsNameOrigin = featsNameFile['featsNameLetter']
featsNameOrigin = featsNameOrigin[1:,:]
featsN, stats = featsNameOrigin.shape
featsName = [None]*featsN
for i in range(featsN):
    featsName[i] = (featsNameOrigin[i,1]).tostring()+'_'+(featsNameOrigin[i,2]).tostring()
    featsName[i] = featsName[i].replace("\x00","")

d=156
numOfFeatsShow=10
ylim=[-5, 15]
xlabel='Feature Name'
ylabel='Value of Model Parameter '+r'$\beta$'
title='Feature Parameter Plot'
# This function is to produce the figure and CSV file list the best numOfFeatsShow features with a bar plot.
# figure's x axis is the index of feature and in the csv file has the corresponding feature names.
betaNoB = beta[0:d]
print len(betaNoB[betaNoB>1e-6])
betaNoBInd = np.flipud(np.argsort(np.absolute(betaNoB), axis=0)[:])
betaShow = betaNoB[betaNoBInd]
x = range(0, numOfFeatsShow)
xStick = [xs+0.5 for xs in x]
# fig = plt.figure()
# index = [i for i in betaNoBInd[:numOfFeatsShow, 0]]
# plt.xticks(xStick, [featsName[i] for i in betaNoBInd[:numOfFeatsShow, 0]], fontsize=8)
# plt.bar(x, betaNoB[index, 0], width=1.0)
# plt.ylabel(ylabel,fontsize=20)
# plt.xlabel(xlabel,fontsize=20)
# plt.title(title,fontsize=20)
# axes = plt.gca()
# axes.set_ylim(ylim)

#fig = plt.figure()
index = [i for i in betaNoBInd[:numOfFeatsShow, 0]]

plt.rcdefaults()
fig, ax = plt.subplots()
ax.barh(x,betaNoB[index, 0])
ax.set_yticks(x)
plt.yticks(x, [featsName[i] for i in betaNoBInd[:numOfFeatsShow, 0]] ,fontsize=10)
#ax.set_ytickLabels([featsName[i] for i in betaNoBInd[:numOfFeatsShow, 0]])
ax.invert_yaxis()
ax.set_xlabel(ylabel,fontsize=15)
ax.set_ylabel(xlabel,fontsize=15)

#plt.xticks(xStick, [featsName[i] for i in betaNoBInd[:numOfFeatsShow, 0]], rotation='vertical',fontsize=100)
#ax = fig.add_subplot(1,1,1)
#ax.bar(x, betaNoB[index, 0], width=1.0)
#ax.tick_params(axis='x',labelsize=15)
#plt.ylabel(ylabel,fontsize=20)
#plt.xlabel(xlabel,fontsize=20)
#plt.title(title,fontsize=20)
#axes = plt.gca()
##xlim = [0.5,9.5]
##axes.set_xlim(xlim)
#axes.set_ylim(ylim)




plt.savefig(figName, bbox_inches='tight')
#outputCSV = open('../data/featureIndex.csv', 'wt')
#writerCSV = csv.writer(outputCSV)
#writerCSV.writerow(('Feature Index', 'Feature Name' ))
#for featCount in range(numOfFeatsShow):
#    writerCSV.writerow(( betaNoBInd[featCount,0]+1, featsName[betaNoBInd[featCount,0]]))



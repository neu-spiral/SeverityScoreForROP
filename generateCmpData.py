# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 21:26:56 2016
This script is to generate commparison data features for the 5941 comparison data. The normalize function included.

@author: TianPeng
"""
from sklearn.preprocessing import normalize,PolynomialFeatures
import numpy as np
from ProbModel import loadComparisonData
import scipy.io as sio

IdOrder, cmpDataId = loadComparisonData('karyn')
M, _ = cmpDataId.shape
data = sio.loadmat('../data/ropData/iROPData_6DD.mat')
dataAuto = sio.loadmat('../data/ropData/pythonFeat.mat')
classLabels1st = data['classLabels1st']
feats1st = dataAuto['feat']
feats = 1*feats1st
feats = normalize(feats, norm='l2', axis=0)
##########################
# multi = feats**2
# feats = np.concatenate((feats,multi),axis=1)
#-----------------------------------------------
# poly = PolynomialFeatures(2,interaction_only=True)
# feats = poly.fit_transform(feats)
##########################
N, d = feats.shape
cmpFeats = np.zeros([M, d])
for i in range(M):
    imgIndCi, imgIndCj = int(
        IdOrder[float(cmpDataId[i, 0])]) - 1, int(IdOrder[float(cmpDataId[i, 1])]) - 1
    featsDiffCurrent = feats[imgIndCi, :] - feats[imgIndCj, :]
    cmpFeats[i, :] = featsDiffCurrent
# sio.savemat('../../Data/ProbalisticModel/iROPcmpData_6DD_Norm.mat',
#             mdict={'cmpFeats': cmpFeats})

# sio.savemat('../../../Data/unet/py1st100Full6DDCmp.mat',
#             mdict={'cmpFeatAuto': cmpFeats, 'featAuto': feats})
sio.savemat('../data/ropData/py1st100Full6DDCmp.mat',
            mdict={'cmpFeatAuto': cmpFeats, 'featAuto': feats})
# iROPMulti.mat contains the file that x, x^2
# iROPPoly.mat contains the polynomial features there.
# iROPPolyIntOnly.mat contaisn the polynomial with interacted terms only.
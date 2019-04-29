from scipy.io import loadmat
import numpy as np
import xlrd as xr


def loadComparisonData(name):
    IdOrderFile = xr.open_workbook('../../data/ropData/100Images.xlsx')
    IdorderSheet = IdOrderFile.sheet_by_name(u'ID&Order')
    Ids = IdorderSheet.col_values(0)
    del Ids[0]
    imageOrder = IdorderSheet.col_values(3)
    del imageOrder[0]
    IdOrder = dict(zip(Ids,imageOrder))
    temp = np.zeros((1,4))
    comparisonData = np.int_(temp)
    fCSV = open('../../data/ropData/results_ICL_third_set_hundred_r2_compare.csv')
    for row in fCSV:
        if not name in row: continue
        content = row.split(',')
        number = content[0:4]
        number = [int(x) for x in number]
        comparisonData = np.vstack((comparisonData,np.reshape(number,[1,-1])))
    comparisonData=np.delete(comparisonData,0,0)
    comparisonData=np.delete(comparisonData,0,1)
    return IdOrder, comparisonData

# The rank generated at MGH.
rankExpertFile = loadmat('../../data/ropData//ExpertRankElo.mat')
rankExpert = rankExpertFile['NewRankElo'][:,0]


# Load Comparison Data
# nameExperts = ['mike', 'paul', 'pete', 'susan']

nameExperts = ['pete', 'paul', 'susan','mike']
IdOrder, cmpData = loadComparisonData('karyn')  # the order in the 13 experts is: 2, 6, 9, 10,12 (start from 0)
M, _ = cmpData.shape
Mtol = M * 5
cmpDataL =  1 * cmpData
for i in range(5 - 1):
    _, cmpDataTmp = loadComparisonData(nameExperts[i])
    cmpDataL = np.concatenate((cmpDataL, cmpDataTmp), axis=0)
# Yc = np.reshape(cmpDataL, [-1, ], order='F')

def determinK(R):
    if R<2100 :
        K = 32
    elif R<=2400 and R>=2100:
        K = 24
    elif R>2400:
        K = 16
    return K



N =100
inital_score_elo = 2200
K = 10
n = 400
elo_score = inital_score_elo*np.ones((N,))
num_cmp_times = np.zeros((N,))
for iter in range(Mtol):
    i, j = int(
        IdOrder[float(cmpDataL[iter, 0])]) - 1, int(IdOrder[float(cmpDataL[iter, 1])]) - 1
#    featsDiffCurrent = feats[imgIndCi, :] - feats[imgIndCj, :]
    # Current rating of i and j
    R_i, R_j = elo_score[i],elo_score[j]
    if cmpDataL[iter,2] == 1:
        S_i, S_j = 1, 0
    elif cmpDataL[iter,2] == -1:
        S_i, S_j = 0, 1
    else:
        print "Label Wrong at iter "+str(iter)
    Q_i, Q_j = 10**(1.*R_i/n), 10**(1.*R_j/n)
    E_i, E_j = 1.*Q_i/(Q_i+Q_j), 1.*Q_j/(Q_i+Q_j)

    # K = determinK(R_i)
    R_i_new = R_i + K*(S_i-E_i)
    # K = determinK(R_j)
    R_j_new = R_j + K*(S_j-E_j)
    elo_score[i] = 1.*R_i_new
    elo_score[j] = 1.*R_j_new
    num_cmp_times[i] = num_cmp_times[i] + 1
    num_cmp_times[j] = num_cmp_times[j] + 1

two_scores = np.concatenate((elo_score[:,np.newaxis],rankExpertFile['NewRankElo'][:,[1]]),axis=1)
temp = elo_score.argsort()
elo_rank = np.empty_like(temp)
elo_rank[temp] = np.arange(1,len(elo_score)+1)
two_ranks = np.concatenate((elo_rank[:,np.newaxis],rankExpert[:,np.newaxis]),axis=1)

rank_diff =  np.sum(np.square(elo_rank-rankExpert))
print rank_diff


print "done"
from sklearn.metrics import roc_curve
import numpy as np
def get_acc_precison_recall(label,score):
    # This function returns the  accuracy precision and recall by setting the threshold of scores via the nearest point
    # to the upper left corner of ROC curve.
    fpr_matrix, tpr_matrix, threshold = roc_curve(label, score)
    dist = np.concatenate([fpr_matrix[:,np.newaxis]-0,tpr_matrix[:,np.newaxis]-1],axis=1)
    distance = np.sum(np.square(dist),axis=1)
    ind = np.argmin(distance,axis=0)
    fpr = fpr_matrix[ind]
    tpr = tpr_matrix[ind]
    fnr = 1 - tpr
    tnr = 1 - fpr
    precision = 1.*tpr / (tpr + fpr)
    recall = 1.*tpr / (tpr + fnr)
    accuracy = 1.*(tpr + tnr) / 2.
    return precision, recall, accuracy

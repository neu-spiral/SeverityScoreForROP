import numpy as np
import argparse
import pickle
from cvxOpt import Log_Log, SVM_Log
from cvxpyMdl import Log_SVM,SVM_SVM
import sys
from sklearn import metrics
from scipy.io import savemat


def train_predict_score(loss, abs_train_feat, abs_train_label, cmp_train_feat, cmp_train_label,
               abs_test_feat, cmp_test_feat, alpha, lamda):
    if loss == 'LogLog':
        beta, const = Log_Log(abs_train_feat, abs_train_label, cmp_train_feat, cmp_train_label, absWeight=alpha,
                              lamda=lamda)
    elif loss == 'LogSVM':
        beta, const = Log_SVM(abs_train_feat, abs_train_label, cmp_train_feat, cmp_train_label, absWeight=alpha,
                              lamda=lamda)
    elif loss == 'SVMLog':
        beta, const = SVM_Log(abs_train_feat, abs_train_label, cmp_train_feat, cmp_train_label, absWeight=alpha,
                              lamda=lamda)
    elif loss == 'SVMSVM':
        beta, const = SVM_SVM(abs_train_feat, abs_train_label, cmp_train_feat, cmp_train_label, absWeight=alpha,
                              lamda=lamda)
    else:
        sys.exit('Please choose the correct loss function from one of {Log_Log,Log_SVM,SVM_Log,SVM_SVM}')
    abs_score = np.dot(abs_test_feat, np.array(beta)) + const
    cmp_score = np.dot(cmp_test_feat, np.array(beta)) + const
    return abs_score, cmp_score, np.array(beta), const


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combining absolute labels and comparison labels on multi-expert data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--alpha', default=1.0, type=float,
                        help='The balance parameter in [0,1] controls weight on absolute labels. 1 means only using absolute labels and 0 means only using comparison labels.')
    parser.add_argument('--lam', default=1.0, type=float,
                        help='The regularization parameter for lasso loss. High value of lambda would show more sparsity.')

    lossFuncGroup = parser.add_mutually_exclusive_group(required=True)
    lossFuncGroup.add_argument('--loss', choices=['LogLog', 'LogSVM', 'SVMLog', 'SVMSVM'], default='LogLog')
    args = parser.parse_args()

    data_file = pickle.load(open('../../../data/feature/abs_cmp_feature_splits_FAC'+'.p','rb'))
    abs_feat, abs_label, abs_splits = data_file['abs_feat'], data_file['abs_label'], data_file['abs_splits']
    cmp_feat, cmp_label, cmp_splits = data_file['cmp_feat'], data_file['cmp_label'], data_file['cmp_splits']
    file_names, k_fold = data_file['abs_img_names'], data_file['k_fold']

    abs_auc = np.zeros((len(abs_splits)/k_fold,1))
    cmp_auc = np.zeros((len(abs_splits)/k_fold,1))
    k_fold_iter = 0
    repeat_count = 0
    abs_score_test_list = []
    abs_label_test_list = []
    cmp_score_test_list = []
    cmp_label_test_list = []
    beta_list = []
    const_list = []
    for iter in range(len(abs_splits)):
        print iter
        k_fold_iter += 1
        abs_train_ind, abs_test_ind = abs_splits[iter]
        cmp_train_ind, cmp_test_ind = cmp_splits[iter]
        abs_train_feat, abs_test_feat = abs_feat[abs_train_ind,:],abs_feat[abs_test_ind,:]
        abs_train_label, abs_test_label = abs_label[abs_train_ind],abs_label[abs_test_ind]
        cmp_train_feat, cmp_test_feat = cmp_feat[cmp_train_ind,:],cmp_feat[cmp_test_ind,:]
        cmp_train_label, cmp_test_label = cmp_label[cmp_train_ind],cmp_label[cmp_test_ind]
        abs_score, cmp_score,beta, const = train_predict_score(args.loss,abs_train_feat,abs_train_label,cmp_train_feat,
                                                               cmp_train_label,abs_test_feat,cmp_test_feat,args.alpha,
                                                               args.lam)
        beta_list.append(beta)
        const_list.append(const)
        abs_score_test_list.append(abs_score)
        abs_label_test_list.append(abs_test_label)
        cmp_score_test_list.append(cmp_score)
        cmp_label_test_list.append(cmp_test_label)
        if k_fold_iter == k_fold:
            k_fold_iter = 0
            abs_auc[repeat_count,0] = metrics.roc_auc_score(np.concatenate(abs_label_test_list), np.concatenate(abs_score_test_list))
            cmp_auc[repeat_count,0] = metrics.roc_auc_score(np.concatenate(cmp_label_test_list), np.concatenate(cmp_score_test_list))
            abs_score_test_list = []
            abs_label_test_list = []
            cmp_score_test_list = []
            cmp_label_test_list = []
            repeat_count += 1

    out_mat_dict = {'abs_auc':abs_auc, 'cmp_auc':cmp_auc,
                'beta_list':np.array(beta_list,dtype=np.object),'const_list':np.array(const_list,dtype=np.object)}

    savemat('../../../result/FAC'+str(args.loss)+'/FAC_'+'_'+str(args.loss)+'_'+str(args.alpha)+'_'+str(args.lam) + '.mat', out_mat_dict)

    print 'done'






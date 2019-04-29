import numpy as np
import argparse
import pickle
from cvxOpt import Log_Log, SVM_Log
from cvxpyMdl import Log_SVM,SVM_SVM
import sys
from sklearn import metrics
from scipy.io import savemat
from compute_statistics import get_acc_precison_recall


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

def recover_cv_from_multi_labelers(test_score_list,repeat_count,k_fold=5):
    tmp_test_score_labeler_list = test_score_list[k_fold * repeat_count:k_fold * (repeat_count + 1)]
    tmp_test_score_list = [np.concatenate(s) for s in tmp_test_score_labeler_list]
    score_test_cv = np.concatenate(tmp_test_score_list)
    return score_test_cv

def recover_expert_cv_from_multi_labelers(test_score_list,repeat_count,number_of_labeler,k_fold=5):
    # This recover each labeler's socore/label from test_score_list.
    labeler_score_list = [None]*num_of_labelers
    for labeler_ind in range(number_of_labeler):
        tmp_test_score_list = [ test_score_list[k_fold * repeat_count+i][labeler_ind] for i in range(k_fold)]
        labeler_score = np.concatenate(tmp_test_score_list)
        labeler_score_list[labeler_ind] = labeler_score[:]
    return labeler_score_list

def ave_auc_expert_cv(test_score_list,test_label_list,repeat_count,number_of_labeler,k_fold=5):
    score_list_multi_labelers = recover_expert_cv_from_multi_labelers(test_score_list,repeat_count,number_of_labeler,k_fold)
    label_list_multi_labelers = recover_expert_cv_from_multi_labelers(test_label_list,repeat_count,number_of_labeler,k_fold)
    auc_multi_labelers = []
    for i in range(number_of_labeler):
        if len(np.unique(label_list_multi_labelers[i]))!=1:
            auc_multi_labelers.append(metrics.roc_auc_score(label_list_multi_labelers[i],score_list_multi_labelers[i]))
    ave_expert_auc = np.mean(np.array(auc_multi_labelers))
    return ave_expert_auc


def generate_comparison_features_labels(abs_feat, cmp_list,d=30):
# Generate comparison feature and labels for one labeler.
# Input: -abs_feat: 17770x30 matrix.
#        - cmp_list: list of tuples in the format of (i,j, y_ij)
#        - d: dimensionality of abs_feat.
    cmp_feat = np.zeros((len(cmp_list),d))
    cmp_label = np.zeros((len(cmp_list),))
    ind = 0
    for cmp in cmp_list:
        feat_i, feat_j = abs_feat[cmp[0],:], abs_feat[cmp[1],:]
        label = cmp[2]
        cmp_feat[ind,:] = feat_i - feat_j
        cmp_label[ind] = label
        ind += 1
    return cmp_feat, cmp_label



def generate_features_labeler(abs_feat, labeler,fold_index):
# Generate absolute and comparison feature and labels for one labeler.
# Input: -abs_feat: 17770x30 matrix.
#        - labeler: string of labeler id.
    labelers_splits_path = '../../../data/feature_netflix_camra/netflix_user_splits/'
    split_file = pickle.load(open(labelers_splits_path + 'Netflix_' + labeler + '_splits' + '.p', 'rb'))
    abs_splits = split_file['abs_splits_user']
    abs_train_ind = abs_splits[fold_index][0].astype(dtype=np.int)
    abs_test_ind = abs_splits[fold_index][1].astype(dtype=np.int)
    abs_train_label = abs_splits[fold_index][2]
    abs_test_label = abs_splits[fold_index][3]
    abs_train_feat = abs_feat[abs_train_ind,:]
    abs_test_feat = abs_feat[abs_test_ind,:]
    # generate comparison feature and labels
    cmp_splits = split_file['cmp_splits_user']
    cmp_train_tuples = cmp_splits[fold_index][0]
    cmp_test_tuples = cmp_splits[fold_index][1]
    cmp_train_feat, cmp_train_label = generate_comparison_features_labels(abs_feat, cmp_train_tuples)
    cmp_test_feat, cmp_test_label = generate_comparison_features_labels(abs_feat, cmp_test_tuples)

    return abs_train_feat, abs_train_label, abs_test_feat, abs_test_label, \
cmp_train_feat, cmp_train_label, cmp_test_feat, cmp_test_label

def generate_global_test_features_labeler(abs_feat, labeler,fold_index):
# Generate absolute and comparison feature and labels for one labeler.
# Input: -abs_feat: 17770x30 matrix.
#        - labeler: string of labeler id.
    labelers_splits_path = '../../../data/feature_netflix_camra/netflix_user_splits/'
    split_file = pickle.load(open(labelers_splits_path + 'Netflix_' + labeler + '_splits' + '.p', 'rb'))
    abs_splits = split_file['abs_splits_user']
    abs_test_ind = abs_splits[fold_index][1].astype(dtype=np.int)
    abs_test_label = abs_splits[fold_index][3]
    abs_test_feat = abs_feat[abs_test_ind,:]
    # generate comparison feature and labels
    cmp_splits = split_file['cmp_splits_user']
    cmp_test_tuples = cmp_splits[fold_index][1]
    cmp_test_feat, cmp_test_label = generate_comparison_features_labels(abs_feat, cmp_test_tuples)
    return abs_test_feat, abs_test_label, cmp_test_feat, cmp_test_label





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

    model = 'EM'
    data_file = pickle.load(open('../../../data/feature_netflix_camra/netflix_abs_feature_labelers.p','rb'))
    abs_feat = data_file['abs_feat'] # Normalized features
    labelers = data_file['valid_labeler']
    num_repartitions = 50
    k_fold = 5

    abs_score_test_list = []
    abs_label_test_list = []
    cmp_score_test_list = []
    cmp_label_test_list = []
    beta_list = []
    const_list = []

    abs_auc = np.zeros((num_repartitions / k_fold, 1))
    cmp_auc = np.zeros((num_repartitions / k_fold, 1))
    abs_ave_expert_auc = np.zeros((num_repartitions / k_fold, 1))
    abs_precision = np.zeros((num_repartitions / k_fold, 1))
    abs_recall = np.zeros((num_repartitions / k_fold, 1))
    abs_accuracy = np.zeros((num_repartitions / k_fold, 1))
    k_fold_iter = 0
    repeat_count = 0
    from time import time
    start_time = time()
    for iter in range(num_repartitions):
        print iter
        k_fold_iter += 1
        num_of_labelers = len(labelers)
        abs_test_score_list, abs_test_label_list = [[None] * num_of_labelers]*num_repartitions, [[None] * num_of_labelers]*num_repartitions
        cmp_test_score_list, cmp_test_label_list = [[None] * num_of_labelers]*num_repartitions, [[None] * num_of_labelers]*num_repartitions
        beta_list = [[None] * num_of_labelers]*num_repartitions
        const_list = [[None] * num_of_labelers]*num_repartitions
        labeler_count = 0
        for labeler in labelers:
            # print labeler_count
            abs_train_feat, abs_train_label, abs_test_feat, abs_test_label, cmp_train_feat, cmp_train_label, cmp_test_feat, cmp_test_label = generate_features_labeler(
                abs_feat, labeler, iter)
            abs_score, cmp_score, beta, const = train_predict_score(args.loss, abs_train_feat, abs_train_label,
                                                                    cmp_train_feat,
                                                                    cmp_train_label, abs_test_feat, cmp_test_feat,
                                                                    args.alpha,
                                                                    args.lam)
            abs_test_score_list[iter][labeler_count] = 1.*abs_score
            cmp_test_score_list[iter][labeler_count] = 1.*cmp_score
            beta_list[iter][labeler_count] = 1.*beta
            const_list[iter][labeler_count] = 1.*const
            abs_test_label_list[iter][labeler_count] = 1.*abs_test_label
            cmp_test_label_list[iter][labeler_count] = 1*cmp_test_label
            labeler_count += 1
        if k_fold_iter == k_fold:
            k_fold_iter = 0
            abs_score_cv = recover_cv_from_multi_labelers(abs_test_score_list,repeat_count)
            abs_label_cv = recover_cv_from_multi_labelers(abs_test_label_list,repeat_count)
            abs_auc[repeat_count, 0] = metrics.roc_auc_score(abs_label_cv,abs_score_cv)
            abs_precision[repeat_count, 0], abs_recall[repeat_count, 0], abs_accuracy[repeat_count, 0] = \
                get_acc_precison_recall(abs_label_cv,abs_score_cv)
            abs_ave_expert_auc[repeat_count,0] = ave_auc_expert_cv(abs_test_score_list, abs_test_label_list, repeat_count, num_of_labelers, k_fold=5)
            cmp_score_cv = recover_cv_from_multi_labelers(cmp_test_score_list, repeat_count)
            cmp_label_cv = recover_cv_from_multi_labelers(cmp_test_label_list, repeat_count)
            cmp_auc[repeat_count, 0] = metrics.roc_auc_score(cmp_label_cv,cmp_score_cv)
            repeat_count += 1
        end_time = time()
        print "One Split takes "+str(end_time-start_time)+' seconds'
    # out_dict = {'beta_list':beta_list,'const':const_list,'abs_auc':abs_auc,'cmp_auc':cmp_auc}

    out_mat_dict = {'beta_list':np.array(beta_list,dtype=np.object),'const':np.array(const_list,dtype=np.object),'abs_auc':abs_auc,'cmp_auc':cmp_auc,
                    'abs_precision':abs_precision,'abs_recall':abs_recall,'abs_accuracy':abs_accuracy,
                    'abs_ave_expert_auc':abs_ave_expert_auc}

    # pickle.dump(out_dict,open('../result/netflix/expertModel/'+str(args.loss)+'/netflix_'+'_'+str(args.loss)+'_'+str(args.alpha)+'_'+str(args.lam) + '.p','wb'))
    savemat('../../../result/netflix/expertModel/'+str(args.loss)+'/netflix_'+'_'+str(args.loss)+'_'+str(args.alpha)+'_'+str(args.lam) + '.mat',out_mat_dict)
    print 'done'








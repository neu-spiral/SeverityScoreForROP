import numpy as np
import argparse
import pickle
from cvxOpt import Log_Log, SVM_Log
from cvxpyMdl import Log_SVM,SVM_SVM
import sys
from sklearn import metrics
from scipy.io import savemat
from compute_statistics import get_acc_precison_recall
import os


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
        cmp_feat[ind,:] = 1.*feat_i - 1.*feat_j
        cmp_label[ind] = 1*label
        ind += 1
    return cmp_feat, cmp_label

def incorporate_expert_bias_abs_feat(abs_feat, num_of_labelers,current_labeler_ind):
    train_bias_term = np.zeros((abs_feat.shape[0], num_of_labelers))
    train_bias_term[:, current_labeler_ind] = 100
    abs_feature_w_bias = np.concatenate((abs_feat, train_bias_term),axis=1)
    return abs_feature_w_bias

def incorporate_expert_bias_cmp_feat(cmp_feat, num_of_labelers):
    train_bias_term = np.zeros((cmp_feat.shape[0], num_of_labelers))
    cmp_feature_w_bias = np.concatenate((cmp_feat, train_bias_term),axis=1)
    return cmp_feature_w_bias



def generate_global_features_labeler(abs_feat, labeler,fold_index):
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


def generate_global_features(abs_feat, labelers, fold_index):
# Generate absolute and comparison feature and labels for all labelers
# Input: list of labelers.
    num_of_labelers = len(labelers)
    abs_train_feat_list, abs_train_label_list, abs_test_feat_list, abs_test_label_list = [None]*num_of_labelers, [None]*num_of_labelers, [None]*num_of_labelers, [None]*num_of_labelers
    cmp_train_feat_list, cmp_train_label_list, cmp_test_feat_list, cmp_test_label_list = [None]*num_of_labelers, [None]*num_of_labelers, [None]*num_of_labelers, [None]*num_of_labelers
    labeler_count = 0
    for labeler in labelers:
        print labeler_count
        abs_train_feat, abs_train_label, abs_test_feat, abs_test_label,cmp_train_feat, cmp_train_label, cmp_test_feat, cmp_test_label = generate_global_features_labeler(abs_feat,labeler,fold_index)
        abs_train_feat_list[labeler_count] = 1.*abs_train_feat
        abs_train_label_list[labeler_count] = 1*abs_train_label
        abs_test_feat_list[labeler_count] = 1.*abs_test_feat
        abs_test_label_list[labeler_count] = 1*abs_test_label
        cmp_train_feat_list[labeler_count] = 1.*cmp_train_feat
        cmp_train_label_list[labeler_count] = 1*cmp_train_label
        cmp_test_feat_list[labeler_count] = 1.*cmp_test_feat
        cmp_test_label_list[labeler_count] = 1*cmp_test_label
        labeler_count += 1
    abs_train_feat_all = np.concatenate(abs_train_feat_list,axis=0)
    abs_train_label_all = np.concatenate(abs_train_label_list,axis=0)
    abs_test_feat_all = np.concatenate(abs_test_feat_list,axis=0)
    abs_test_label_all = np.concatenate(abs_test_label_list,axis=0)
    cmp_train_feat_all = np.concatenate(cmp_train_feat_list,axis=0)
    cmp_train_label_all = np.concatenate(cmp_train_label_list,axis=0)
    cmp_test_feat_all = np.concatenate(cmp_test_feat_list,axis=0)
    cmp_test_label_all = np.concatenate(cmp_test_label_list,axis=0)
    return abs_train_feat_all, abs_train_label_all, abs_test_feat_all, abs_test_label_all, \
           cmp_train_feat_all,cmp_train_label_all,cmp_test_feat_all, cmp_test_label_all

def generate_global_expert_bias_features(abs_feat, labelers, fold_index):
# Generate absolute and comparison feature and labels for all labelers
# Input: list of labelers.
    num_of_labelers = len(labelers)
    abs_train_feat_list, abs_train_label_list, abs_test_feat_list, abs_test_label_list = [None]*num_of_labelers, [None]*num_of_labelers, [None]*num_of_labelers, [None]*num_of_labelers
    cmp_train_feat_list, cmp_train_label_list, cmp_test_feat_list, cmp_test_label_list = [None]*num_of_labelers, [None]*num_of_labelers, [None]*num_of_labelers, [None]*num_of_labelers
    labeler_count = 0
    for labeler in labelers:
        print labeler_count
        abs_train_feat, abs_train_label, abs_test_feat, abs_test_label,cmp_train_feat, cmp_train_label, cmp_test_feat, cmp_test_label = generate_global_features_labeler(abs_feat,labeler,fold_index)
        abs_train_feat_expert_bias = incorporate_expert_bias_abs_feat(abs_train_feat, num_of_labelers, labeler_count)
        abs_train_feat_list[labeler_count] = 1.*abs_train_feat_expert_bias
        abs_train_label_list[labeler_count] = 1*abs_train_label
        abs_test_feat_expert_bias = incorporate_expert_bias_abs_feat(abs_test_feat, num_of_labelers, labeler_count)
        abs_test_feat_list[labeler_count] = 1.*abs_test_feat_expert_bias
        abs_test_label_list[labeler_count] = 1*abs_test_label
        cmp_train_feat_expert_bias = incorporate_expert_bias_cmp_feat(cmp_train_feat, num_of_labelers)
        cmp_train_feat_list[labeler_count] = 1.*cmp_train_feat_expert_bias
        cmp_train_label_list[labeler_count] = 1*cmp_train_label
        cmp_test_feat_expert_bias = incorporate_expert_bias_cmp_feat(cmp_test_feat, num_of_labelers)
        cmp_test_feat_list[labeler_count] = 1.*cmp_test_feat_expert_bias
        cmp_test_label_list[labeler_count] = 1*cmp_test_label
        labeler_count += 1
    abs_train_feat_all = np.concatenate(abs_train_feat_list,axis=0)
    abs_train_label_all = np.concatenate(abs_train_label_list,axis=0)
    abs_test_feat_all = np.concatenate(abs_test_feat_list,axis=0)
    abs_test_label_all = np.concatenate(abs_test_label_list,axis=0)
    cmp_train_feat_all = np.concatenate(cmp_train_feat_list,axis=0)
    cmp_train_label_all = np.concatenate(cmp_train_label_list,axis=0)
    cmp_test_feat_all = np.concatenate(cmp_test_feat_list,axis=0)
    cmp_test_label_all = np.concatenate(cmp_test_label_list,axis=0)
    return abs_train_feat_all, abs_train_label_all, abs_test_feat_all, abs_test_label_all, \
           cmp_train_feat_all,cmp_train_label_all,cmp_test_feat_all, cmp_test_label_all

def generate_global_test_features(abs_feat, labelers, fold_index):
# Generate absolute and comparison feature and labels for all labelers
# Input: list of labelers.
    num_of_labelers = len(labelers)
    abs_test_feat_list, abs_test_label_list = [None]*num_of_labelers, [None]*num_of_labelers
    cmp_test_feat_list, cmp_test_label_list = [None]*num_of_labelers, [None]*num_of_labelers
    labeler_count = 0
    for labeler in labelers:
        print labeler_count
        abs_test_feat, abs_test_label, cmp_test_feat, cmp_test_label = generate_global_test_features_labeler(abs_feat,labeler,fold_index)
        abs_test_feat_list[labeler_count] = 1.*abs_test_feat
        abs_test_label_list[labeler_count] = 1*abs_test_label
        cmp_test_feat_list[labeler_count] = 1.*cmp_test_feat
        cmp_test_label_list[labeler_count] = 1*cmp_test_label
        labeler_count += 1
    abs_test_feat_all = np.concatenate(abs_test_feat_list,axis=0)
    abs_test_label_all = np.concatenate(abs_test_label_list,axis=0)
    cmp_test_feat_all = np.concatenate(cmp_test_feat_list,axis=0)
    cmp_test_label_all = np.concatenate(cmp_test_label_list,axis=0)
    return abs_test_feat_all, abs_test_label_all, cmp_test_feat_all, cmp_test_label_all

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combining absolute labels and comparison labels on multi-expert data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', default='GM', choices=['GM','GMEB'],
                        help='choose which dataset works on',type=str)
    parser.add_argument('--alpha', default=1.0, type=float,
                        help='The balance parameter in [0,1] controls weight on absolute labels. 1 means only using absolute labels and 0 means only using comparison labels.')
    parser.add_argument('--lam', default=1.0, type=float,
                        help='The regularization parameter for lasso loss. High value of lambda would show more sparsity.')
    lossFuncGroup = parser.add_mutually_exclusive_group(required=True)
    lossFuncGroup.add_argument('--loss', choices=['LogLog', 'LogSVM', 'SVMLog', 'SVMSVM'], default='LogLog')
    args = parser.parse_args()
    if args.model == "GM":
        folder = 'globalModel/' + str(args.loss) + '/'
    elif args.model == "GMEB":
        folder = 'globalModelExpertBias/' + str(args.loss) + '/'
    # pickle.dump(out_dict,open('../result/netflix/'+folder+'/netflix_'+'_'+str(args.loss)+'_'+str(args.alpha)+'_'+str(args.lam) + '.p','wb'))
    if not os.path.isfile('../../../result/netflix/' + folder + '/netflix_' + '_' + str(args.loss) + '_' + str(args.alpha) + '_' + str(
        args.lam) + '.mat'):


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
        abs_precision = np.zeros((num_repartitions / k_fold, 1))
        abs_recall = np.zeros((num_repartitions / k_fold, 1))
        abs_accuracy = np.zeros((num_repartitions / k_fold, 1))
        k_fold_iter = 0
        repeat_count = 0
        for iter in range(num_repartitions):
            print iter
            k_fold_iter += 1
            if args.model == "GM":
                abs_train_feat, abs_train_label, abs_test_feat, abs_test_label, \
                cmp_train_feat, cmp_train_label, cmp_test_feat, cmp_test_label = generate_global_features(
                    abs_feat, labelers, iter)
            elif args.model == "GMEB":
                abs_train_feat, abs_train_label, abs_test_feat, abs_test_label, \
                cmp_train_feat, cmp_train_label, cmp_test_feat, cmp_test_label = generate_global_expert_bias_features(
                    abs_feat, labelers, iter)

            abs_score, cmp_score, beta, const = train_predict_score(args.loss, abs_train_feat, abs_train_label,
                                                                    cmp_train_feat,
                                                                    cmp_train_label, abs_test_feat, cmp_test_feat,
                                                                    args.alpha,
                                                                    args.lam)

            beta_list.append(1.*beta)
            const_list.append(1.*const)
            abs_score_test_list.append(1.*abs_score)
            cmp_score_test_list.append(1.*cmp_score)
            abs_label_test_list.append(1*abs_test_label)
            cmp_label_test_list.append(1*cmp_test_label)


            if k_fold_iter == k_fold:
                k_fold_iter = 0
                abs_auc[repeat_count, 0] = metrics.roc_auc_score(np.concatenate(abs_label_test_list[k_fold*repeat_count:k_fold*(repeat_count+1)]),
                                                                 np.concatenate(abs_score_test_list[k_fold*repeat_count:k_fold*(repeat_count+1)]))
                abs_precision[repeat_count,0], abs_recall[repeat_count,0], abs_accuracy[repeat_count,0] = get_acc_precison_recall(np.concatenate(abs_label_test_list[k_fold*repeat_count:k_fold*(repeat_count+1)]),
                                                                 np.concatenate(abs_score_test_list[k_fold*repeat_count:k_fold*(repeat_count+1)]))
                cmp_auc[repeat_count, 0] = metrics.roc_auc_score(np.concatenate(cmp_label_test_list[k_fold*repeat_count:k_fold*(repeat_count+1)]),
                                                                 np.concatenate(cmp_score_test_list[k_fold*repeat_count:k_fold*(repeat_count+1)]))
                repeat_count += 1
        # out_dict = {'beta_list':beta_list,'const':const_list,'abs_auc':abs_auc,'cmp_auc':cmp_auc,
        #             'abs_precision':abs_precision,'abs_recall':abs_recall,'abs_accuracy':abs_accuracy}
        out_mat_dict = {'beta_list':np.array(beta_list,dtype=np.object),'const':np.array(const_list,dtype=np.object),'abs_auc':abs_auc,'cmp_auc':cmp_auc,
                        'abs_precision': abs_precision, 'abs_recall': abs_recall, 'abs_accuracy': abs_accuracy}
        if args.model == "GM":
            folder = 'globalModel/'+str(args.loss)+'/'
        elif args.model == "GMEB":
            folder = 'globalModelExpertBias/'+str(args.loss)+'/'
        # pickle.dump(out_dict,open('../result/netflix/'+folder+'/netflix_'+'_'+str(args.loss)+'_'+str(args.alpha)+'_'+str(args.lam) + '.p','wb'))
        savemat('../../../result/netflix/'+folder+'/netflix_'+'_'+str(args.loss)+'_'+str(args.alpha)+'_'+str(args.lam) + '.mat',out_mat_dict)

    print 'done'







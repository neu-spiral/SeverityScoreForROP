import numpy as np
import pickle
from sklearn.model_selection import RepeatedStratifiedKFold


pickle_file = pickle.load(open('../../../data/feature_FAC/abs_cmp_feature_category_0.p', 'rb'))

abs_img_names = pickle_file['abs_img_names']
abs_feat = pickle_file['abs_feature']
abs_label = pickle_file['abs_label']
cmp_data = pickle_file['cmp_data']
cmp_feat = pickle_file['cmp_feature']
cmp_label = pickle_file['cmp_label']

skf =  RepeatedStratifiedKFold(n_splits=5,n_repeats=10,random_state=1)
abs_splits = list(skf.split(abs_feat,abs_label)) # abs has 50 (train,test pairs) Consecutive 5 paris is one repartition.
cmp_splits = []
for abs_train_ind, abs_test_ind in abs_splits:
    cmp_train_ind = []
    cmp_test_ind = []
    abs_file_names_train = list(np.array(abs_img_names)[abs_train_ind])
    abs_file_names_test = list(np.array(abs_img_names)[abs_test_ind])
    for row_ind in range(len(cmp_data)):
        img_left, img_right, _ = cmp_data[row_ind]
        if img_left in abs_file_names_train and img_right in abs_file_names_train:
            cmp_train_ind.append(row_ind)
        elif img_left in abs_file_names_test and img_right in abs_file_names_test:
            cmp_test_ind.append(row_ind)
    cmp_splits.append((cmp_train_ind[:],cmp_test_ind[:]))

out_dict = {'abs_feat':abs_feat,'abs_label':abs_label, 'abs_splits':abs_splits,
            'abs_img_names':abs_img_names, 'k_fold':5,
            'cmp_data':cmp_data,'cmp_feat':cmp_feat, 'cmp_label':cmp_label,
            'cmp_splits':cmp_splits}

out_mat_dict = {'abs_feat':np.array(abs_feat,dtype=np.double),'abs_label':abs_label, 'abs_splits':abs_splits,
            'abs_img_names':np.array(abs_img_names,dtype=np.object), 'k_fold':5,
            'cmp_data':np.array(cmp_data,dtype=np.object),'cmp_feat':np.array(cmp_feat,dtype=np.double), 'cmp_label':cmp_label,
            'cmp_splits':np.array(cmp_splits,dtype=np.object)}

from scipy.io import savemat
savemat('../../../data/feature_FAC/abs_cmp_feature_splits_FAC'+'.mat',out_mat_dict)

pickle.dump( out_dict, open( '../../../data/feature_FAC/abs_cmp_feature_splits_FAC'+'.p', "wb" ) )

print "done"


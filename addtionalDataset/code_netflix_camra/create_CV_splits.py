# This file creates 5 fold cross-validation 10 times with randomnization for netflix and camera dataset. The cross validation separtes both
# # absolute labels and comparison labels and guarantee that no images in the test set shows in the training set, even so in the comparison splits.


import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler


def generate_abs_labels(movie_scores):
    # This function generate absolute labels for each user by using her average score. + 1 means above average -1 means below.
    # Input: - movie_scores: list of tupels. In each tuple, there is (movie_id, score by user).
    # Output - m x 3 ndarray, abs_label_user, the first column is the movie id and the second column is the absolute labels, third column is the score.
    movie_ids = []
    absolute_labels = []
    scores = []
    average_score = 1.*sum(score for _, score in movie_scores)/len(movie_scores)
    for movie_id, movie_score in movie_scores:
        movie_ids.append(movie_id)
        scores.append(movie_score)
        if movie_score>= average_score:
            absolute_labels.append(1)
        else:
            absolute_labels.append(-1)
    abs_label_user = np.array([movie_ids,absolute_labels,scores]).T
    return abs_label_user


def generate_cmp_labels(movie_scores):
    # This function generate absolute labels for each user by using her average score. + 1 means above average -1 means below.
    # Input: - movie_scores: list of tupels. In each tuple, there is (movie_id, score by user).
    # Output - list of tuples, cmp_data_user, In each tuple, (i,j,cmp_label), i ,j are movie ids and cmp_label is in {-1, +1}.
    num_of_movies = len(movie_scores)
    cmp_data_user =[]
    for movie_i_ind in range(num_of_movies-1):
        id_movie_i, score_movie_i = movie_scores[movie_i_ind]
        for movie_j_ind in range(movie_i_ind+1,num_of_movies):
            id_movie_j, score_movie_j = movie_scores[movie_j_ind]
            if score_movie_i>score_movie_j:
                cmp_data_user.append((id_movie_i,id_movie_j,1))
            elif score_movie_i<score_movie_j:
                cmp_data_user.append((id_movie_i,id_movie_j,-1))
    return cmp_data_user



data_file = pickle.load(open('../../../data/feature_netflix_camra/NetflixFeature.p','rb'))
labelers = data_file.keys()#
labelers.remove('Feature')
abs_feat_unnormalized = data_file['Feature']
scaler = StandardScaler()
# Fit your data on the scaler object
abs_feat = scaler.fit_transform(abs_feat_unnormalized)
num_of_movies = abs_feat.shape[0]
movie_scores =  [[] for _ in xrange(num_of_movies)]
num_of_movies_valid_labeler_labeled = 1000
valid_labeler = []
for labeler in labelers:
    movies_labeler = data_file[labeler]
    if len(movies_labeler) >= num_of_movies_valid_labeler_labeled:
        valid_labeler.append(labeler)
    for movie_id, movie_score in movies_labeler:
        movie_scores[movie_id].append(movie_score)

ave_movie_score_times = np.zeros((num_of_movies,2)) #First column is the ave grade score, the second the number of grading times

movie_id_no_score = []   # The row number in the 17770x30 abs_feat matrix.
movie_id_has_score = []
for movie_id in range(num_of_movies):
    if len(movie_scores[movie_id]) == 0:
        movie_id_no_score.append(movie_id)
    else:
        movie_id_has_score.append(movie_id)
        ave_movie_score_times[movie_id,0] = np.mean(np.array(movie_scores[movie_id]))
        ave_movie_score_times[movie_id,1] = len(movie_scores[movie_id])
ave_score_all_movies = np.mean(ave_movie_score_times[np.array(movie_id_has_score),0])
valid_movie_id = np.array(movie_id_has_score)
invalid_movie_id = np.array(movie_id_no_score) # each number corresponds to a row in 17770x30 matrix.

# Filter out movies that at least graded 10 times.
ind_sort_time = ave_movie_score_times[:,1].argsort()
ind_sort_time = ind_sort_time[::-1]
movies_min_score_times = 350
num_filtered_movies = np.where(ave_movie_score_times[ind_sort_time,1]>=movies_min_score_times)[0].shape[0]
filtered_movie_id = ind_sort_time[:num_filtered_movies]


abs_label = ave_movie_score_times[:,0]-ave_score_all_movies
abs_label[abs_label>=0] = 1
abs_label[abs_label<0] = -1
abs_label[np.array(movie_id_no_score)] = 0

valid_abs_feat = abs_feat[filtered_movie_id,:]  # valid means the movies has received a score from some one and graded many times.
valid_abs_label = abs_label[filtered_movie_id]

skf =  RepeatedStratifiedKFold(n_splits=5,n_repeats=10,random_state=1)
abs_splits = list(skf.split(valid_abs_feat,valid_abs_label)) # abs has 50 (train,test pairs) Consecutive 5 paris is one repartition.
# splits contains the feature and label in the valid movie_id, which is 12744x30
labeler_train_test_splits = {}

#--------------------------------------------
# This is used to compute the number of absolute labels and comparison labels in the dataset. Usually commented
# num_total_abs_labels = 0
# num_total_cmp_labels = 0
# for labeler in valid_labeler:
#     abs_label_user = generate_abs_labels(data_file[labeler])
#     abs_movie_id, _, ind_label_user_array = np.intersect1d(filtered_movie_id, abs_label_user[:, 0],
#                                      return_indices=True)
#     num_total_abs_labels += len(abs_movie_id)
#     abs_movie_id_score = abs_label_user[:, [0, 2]][ind_label_user_array, :]
#     cmp_train_tuples = generate_cmp_labels(list(map(tuple, abs_movie_id_score.astype(dtype=np.int))))
#     num_total_cmp_labels += len(cmp_train_tuples)
# print "num of absolute labels:" +str(num_total_abs_labels)
# print "num of comparison labels:" +str(num_total_cmp_labels)
#--------------------------------------------



count = 0
for labeler in valid_labeler:
    print str(count)+'/'+str(len(valid_labeler))
    count+=1
    abs_label_user = generate_abs_labels(data_file[labeler])
    # cmp_data_user = generate_cmp_labels(data_file[labeler])
    abs_splits_user = []
    cmp_splits_user = []
    splits = 0
    for abs_train_ind_valid_set, abs_test_ind_valid_set in abs_splits:
        print str(splits)+'/50'
        splits += 1
        abs_train_ind = filtered_movie_id[abs_train_ind_valid_set] # This is the movie id in 17770x30 matrix
        abs_test_ind = filtered_movie_id[abs_test_ind_valid_set]
        abs_train_movie_id_fold,_,train_ind_label_user_array = np.intersect1d(abs_train_ind,abs_label_user[:,0],return_indices=True)
        abs_train_label = abs_label_user[train_ind_label_user_array,1]
        abs_test_movie_id_fold,_,test_ind_label_user_array = np.intersect1d(abs_test_ind,abs_label_user[:,0],return_indices=True)
        abs_test_label = abs_label_user[test_ind_label_user_array, 1]
        # abs_feat[abs_train_movie_id_fold,:] is the training feature in this fold.
        abs_splits_user.append((abs_train_movie_id_fold, abs_test_movie_id_fold, abs_train_label, abs_test_label))

        abs_train_movie_id_score = abs_label_user[:, [0,2]][train_ind_label_user_array,:]
        abs_test_movie_id_score = abs_label_user[:, [0,2]][test_ind_label_user_array,:]
        cmp_train_tuples = generate_cmp_labels(list(map(tuple,abs_train_movie_id_score.astype(dtype=np.int))))
        cmp_test_tuples = generate_cmp_labels(list(map(tuple,abs_test_movie_id_score.astype(dtype=np.int))))
        cmp_splits_user.append((cmp_train_tuples,cmp_test_tuples))
    out_labeler_dict = {'abs_splits_user':abs_splits_user,'cmp_splits_user':cmp_splits_user}
    pickle.dump(out_labeler_dict, open('../../../data/feature_netflix_camra/netflix_user_splits/Netflix_'+labeler+'_splits' + '.p', "wb"))
    # Each user's labeler_train_test_splits: abs_splits_user 50 tuples (train_id,test_id,train_label,test_label), each tuple is the movie id in the 17770x30 matrix.
    # cmp_splits_user: 50 tuples (cmp_train_tuples,cmp_test_tuples). cmp_train_tuples/cmp_test_tuples contains many tuples in (i,j,cmp_label) form.
    #
out_dict = {'abs_feat':abs_feat,'valid_labeler':valid_labeler,'filtered_movie_id':filtered_movie_id}

pickle.dump( out_dict, open( '../../../data/feature_netflix_camra/netflix_abs_feature_labelers'+'.p', "wb" ))

print "done"
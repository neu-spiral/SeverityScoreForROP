import pickle
import numpy as np
from keras.layers import Input
from keras.models import Model
from googlenet_functional import create_googlenet
from os import listdir
from keras.preprocessing.image import load_img, img_to_array
from scipy.misc import imresize
from sklearn.preprocessing import StandardScaler



def get_abs_cmp_data(category=0,dir="../IMAGE_QUALITY_DATA"):
    abs_label_file = dir+"/image_score.pkl"
    comp_label_file = dir+"/pairwise_comparison.pkl"
    with open(comp_label_file, 'rb') as f:
        comp_label_matrix = pickle.load(f)
    # data = pickle.load(open(partition_file_6000, 'rb'))
    with open(abs_label_file, 'rb') as f:
        abs_label_matrix = pickle.load(f)
    # category = 0  # Only use one category in the dataset

    image_name_list = []
    np.random.seed(1)
    # get all unique images in category
    for row in comp_label_matrix:
        # category, f1, f2, workerID, passDup, imgId, ans
        if row['category'] == category:
            image1_name = '/' + row['f1'] + '/' + row['imgId'] + '.jpg'
            if image1_name not in image_name_list:
                image_name_list.append(image1_name)
            image2_name = '/' + row['f2'] + '/' + row['imgId'] + '.jpg'
            if image2_name not in image_name_list:
                image_name_list.append(image2_name)

    abs_img_names = []
    abs_label = []
    for row in abs_label_matrix:
        # filterName, imgId, class, score
        image_name = '/' + row['filterName'] + '/' + row['imgId'] + '.jpg'
        if image_name in image_name_list:
            abs_img_names.append(image_name)
            if row['class'] == '1':
                abs_label.append(1)
            elif row['class'] == '0':
                abs_label.append(-1)
    abs_label = np.array(abs_label)

    cmp_data = []
    for row in comp_label_matrix:
        if row['category'] == category:
            # category, f1, f2, workerID, passDup, imgId, ans
            image1_name = '/' + row['f1'] + '/' + row['imgId'] + '.jpg'
            image2_name = '/' + row['f2'] + '/' + row['imgId'] + '.jpg'
            # test
            if image1_name in abs_img_names and image2_name in abs_img_names:
                # save comparison label
                if row['ans'] == 'left':
                    cmp_data.append((image1_name, image2_name, +1))
                elif row['ans'] == 'right':
                    cmp_data.append((image1_name, image2_name, -1))

    return abs_img_names, abs_label,cmp_data

def create_network():
    input1 = Input(shape=(3, 224, 224))
    input2 = Input(shape=(3, 224, 224))
    feature1, _ = create_googlenet(input1, input2)
    base_net = Model(input1, feature1)
    base_net.load_weights('./googlenet_weights.h5', by_name=True)
    base_net.compile(loss='mean_squared_error', optimizer='sgd')
    return base_net

def extract_feature(base_net,abs_img):
    abs_feature = base_net.predict(abs_img)
    return abs_feature

def extract_feat_based_on_path(files,file_path="../IMAGE_QUALITY_DATA"):
    base_net = create_network()
    input_shape = (3,224,224)

    img_mtx = img_to_array(load_img(file_path+files[0])).astype(np.uint8)
    img_mtx = np.reshape(imresize(img_mtx,input_shape[1:]),input_shape)
    img_input = img_mtx[np.newaxis,:,:,:]
    file_names = [files[0]]
    for file_ind in range(1,len(files)):
        img_mtx = img_to_array(load_img(file_path + files[file_ind])).astype(np.uint8)
        img_mtx = np.reshape(imresize(img_mtx, input_shape[1:]), input_shape)[np.newaxis,:,:,:]
        img_input = np.concatenate((img_input,img_mtx),axis=0)
        file_names.append(files[file_ind])
    features = extract_feature(base_net,img_input)
    return features,file_names


if __name__ == "__main__":
    dir = "../IMAGE_QUALITY_DATA"
    abs_img_names, abs_label, cmp_data = get_abs_cmp_data()
    abs_feature_unnormalized, abs_file_names = extract_feat_based_on_path(abs_img_names,file_path=dir)
    scaler = StandardScaler()
    # Fit your data on the scaler object
    abs_feature = scaler.fit_transform(abs_feature_unnormalized)
    cmp_feature_list = []
    cmp_label_list = []
    n_cmp = len(cmp_data)
    n_cmp_count = 0
    for img_1,img_2,cmp_pair_label in cmp_data:
        print str(n_cmp_count)+'/'+str(n_cmp)
        n_cmp_count += 1
        ind_1,ind_2 = abs_img_names.index(img_1), abs_img_names.index(img_2)
        cmp_feature_list.append(abs_feature[[ind_1],:]-abs_feature[[ind_2],:])
        cmp_label_list.append(1.*cmp_pair_label)

    cmp_feature = np.concatenate(cmp_feature_list,axis=0)
    cmp_label = np.array(cmp_label_list)
    out_dict = {'abs_img_names':abs_img_names,'abs_feature':abs_feature,'abs_label':abs_label,
                'cmp_data':cmp_data,'cmp_feature':cmp_feature,'cmp_label':cmp_label}
    pickle.dump(out_dict,open('../../../data/feature_FAC/abs_cmp_feature_category_0.p','wb'))





    print "Done"
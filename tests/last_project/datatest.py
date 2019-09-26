# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import global_feature_extraction as fe
import numpy as np
import cv2
import os
import mahotas
import h5py

train_path = 'dataset/train'
test_path  = 'dataset/test'
fixed_size = tuple((500, 500))
images_per_class = 100
h5_data          = 'output/data.h5'
h5_labels        = 'output/labels.h5'
# empty lists to hold feature vectors and labels
global_features = []
labels          = []

#object of feature extractor
fe_obj = fe.Global_feature_extraction()

# get the training labels
train_labels = os.listdir(train_path)
# sort the training labels
train_labels.sort()

# loop over the training data sub-folders
for training_name in train_labels:
    # join the training data path and each species training folder
    dir = os.path.join(train_path, training_name)

    # get the current training label
    current_label = training_name

    # loop over the images in each sub-folder
    for x in range(1,images_per_class+1):
        # get the image file name
        # you should rename all data in the numbers on 1to End.
        file = dir + "/" + str(x) + ".jpg"
        #file = 'dataset/train/alalog-watch/analog-watch1.jpg'
        # read the image and resize it to a fixed-size
        image = cv2.imread(file)
        print(file+'   readed   ')
        image = cv2.resize(image, fixed_size)
        print(file+'   resized   ')

        ####################################
        # Global Feature extraction
        ####################################
        shape = fe_obj.fd_hu_moments(image)
        texture   = fe_obj.fd_haralick(image)
        color  = fe_obj.fd_histogram(image)

        ###################################
        # Concatenate global features
        ###################################
        global_feature = np.hstack([color, texture, shape])

        # update the list of labels and feature vectors
        labels.append(current_label)
        global_features.append(global_feature)

    print("[STATUS] processed folder: {}".format(current_label))
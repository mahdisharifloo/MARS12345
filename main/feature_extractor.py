# -*- coding: utf-8 -*-
#_______________________________________
#this file calls utils to extract features of dataset 
#this file save results on h5 files
#_______________________________________

#libraties
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import h5py

# some_file.py
import sys
import os
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../utils/')
#import utils and toolbox
import global_feature_extraction as fe

#_____________________________________________
#set global values
#---------------------------------------------
train_path = 'dataset/train'
test_path  = 'dataset/test'
fixed_size = tuple((500, 500))
images_per_class = 440
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
#---------------------------------------------

#_________________________________________________________________
# loop over the training data sub-folders
#-----------------------------------------------------------------
for training_name in train_labels:
    # join the training data path and each species training folder
    dir = os.path.join(train_path, training_name)

    # get the current training label
    current_label = training_name

    # loop over the images in each sub-folder
    for x in range(1,images_per_class+1):
        # get the image file name
        # you should rename all data in the numbers on 1to End.
        file = dir + "/" + str(x) + ".png"
        #file = 'dataset/train/alalog-watch/analog-watch1.jpg'
        # read the image and resize it to a fixed-size
        image = cv2.imread(file)
        image = cv2.resize(image, fixed_size)

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
    
#------------------------------------------------------------------

#________________________
#print some result
#------------------------
print("[STATUS] completed Global Feature Extraction...")
# get the overall feature vector size
print("[STATUS] feature vector size {}".format(np.array(global_features).shape))
# get the overall training label size
print("[STATUS] training Labels {}".format(np.array(labels).shape))
#-------------------------

#_________________________________________________
# encode the target labels
#-------------------------------------------------
targetNames = np.unique(labels)
le          = LabelEncoder()
target      = le.fit_transform(labels)
print("[STATUS] training labels encoded...")

# scale features in the range (0-1)
scaler            = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)
print("[STATUS] feature vector normalized...")

print("[STATUS] target labels: {}".format(target))
print("[STATUS] target labels shape: {}".format(target.shape))

# save the feature vector using HDF5
h5f_data = h5py.File(h5_data, 'w')
h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

h5f_label = h5py.File(h5_labels, 'w')
h5f_label.create_dataset('dataset_1', data=np.array(target))

h5f_data.close()
h5f_label.close()

print("[STATUS] end of training..")
#-------------------------------------------------
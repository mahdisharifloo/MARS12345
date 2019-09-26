# -*- coding: utf-8 -*-

import os
import glob
import datetime
import tarfile
import urllib.request

train_path='dataset/train'

# empty lists to hold feature vectors and labels
global_features = []
labels          = []
images_per_class = 10


#to finding just jpg files
def jpg_files(members):
  for tarinfo in members:
    if os.path.splitext(tarinfo.name)[1] == ".jpg":
      yield tarinfo
      
      
# get the training labels
train_labels = os.listdir(train_path)
# sort the training labels
train_labels.sort()
print(train_labels)

# loop over the training data sub-folders
for training_name in train_labels:
    # join the training data path and each species training folder
    dir = os.path.join(train_path, training_name)

    # get the current training label
    current_label = training_name

    # loop over the images in each sub-folder
    for x in range(1,images_per_class+1):
        # get the image file name
        file = dir + "/" + str(x) + ".jpg"

        # read the image and resize it to a fixed-size
        image = cv2.imread(file)
        image = cv2.resize(image, fixed_size)

        ####################################
        # Global Feature extraction
        ####################################
        fv_hu_moments = fd_hu_moments(image)
        fv_haralick   = fd_haralick(image)
        fv_histogram  = fd_histogram(image)

        ###################################
        # Concatenate global features
        ###################################
        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

        # update the list of labels and feature vectors
        labels.append(current_label)
        global_features.append(global_feature)

    print("[STATUS] processed folder: {}".format(current_label))

print("[STATUS] completed Global Feature Extraction...")
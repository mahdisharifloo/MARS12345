# -*- coding: utf-8 -*-
import os
# Function to rename multiple files

train_path = 'dataset/train' 
test_path = 'dataset/test'
        

def train_rename(train_path):
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
        i = 1
        for filename in os.listdir(dir): 
                dst = str(i) + ".png"
                src = os.path.normpath(dir+'/'+ filename)

                # rename() function will 
                # rename all the files 
                os.rename(src, os.path.join(dir+'/'+dst)) 
                i += 1


def test_rename(test_path):
    # get the training labels
    test_labels = os.listdir(test_path)
    # sort the training labels
    test_labels.sort()

    # loop over the training data sub-folders
    for testing_name in test_labels:
        # join the training data path and each species training folder
        dir = os.path.join(test_path, testing_name)

        # get the current training label
        current_label = testing_name
        i = 1
        for filename in os.listdir(dir): 
                dst = str(i) + ".png"
                src = os.path.normpath(dir+'/'+ filename)

                # rename() function will 
                # rename all the files 
                os.rename(src, os.path.join(dir+'/'+dst)) 
                i += 1

if __name__ == "__main__":
    train_rename(train_path)
    test_rename(test_path)


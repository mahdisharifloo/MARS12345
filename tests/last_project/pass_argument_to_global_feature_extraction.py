# -*- coding: utf-8 -*-

import global_feature_extraction as fe
import numpy as np
import cv2


fe_obj = fe.Global_feature_extraction()

# read the image and resize it to a fixed-size
file = 'dataset/train/analog-watch/1.jpg'
fixed_size = tuple((500, 500))

image = cv2.imread(file)
image = cv2.resize(image, fixed_size)
####################################
# Global Feature extraction
####################################
shape = fe_obj.fd_hu_moments(image)
texture   = fe_obj.fd_haralick(image)
color  = fe_obj.fd_histogram(image)
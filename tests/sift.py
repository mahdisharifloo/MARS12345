# -*- coding: utf-8 -*-

import cv2
import numpy as np

img = cv2.imread('/home/mahdi/9.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT()
kp = sift.detect(gray,None)

img=cv2.drawKeypoints(gray,kp)

cv2.imwrite('./images/sift_keypoints.jpg',img)
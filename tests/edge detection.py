# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 10:04:28 2019

@author: MCA
"""



import cv2
import numpy as np

img = cv2.imread("19_100.jpg", 0)
cv2.imwrite("canny.jpg", cv2.Canny(img, 200,300))
cv2.imshow("canny", cv2.imread("canny.jpg"))
cv2.waitKey()
cv2.destroyAllWindows()


# -*- coding: utf-8 -*-

#########################
# import libraries
#########################
import cv2
import numpy as np
from matplotlib import pyplot as plt

#########################
# operations
#########################
class local_feature_extraction:
    def __init__(self,img_path):
        self.img_path = img_path
    
    def SIFT(self):
        #https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_surf_intro/py_surf_intro.html
        img = cv2.imread(image)
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT()
        kp = sift.detect(gray,None)
        
        img=cv2.drawKeypoints(gray,kp)
        
        cv2.imwrite('sift_keypoints.jpg',img)
        
        
        
    def SURF(self):
        img = cv2.imread('fly.png',0)
        # Create SURF object. You can specify params here or later.
        # Here I set Hessian Threshold to 400
        surf = cv2.SURF(400)
        kp, des = surf.detectAndCompute(img,None)
        # Find keypoints and descriptors directly
        print( surf.hessianThreshold)
        # We set it to some 50000. Remember, it is just for representing in picture.
        # In actual cases, it is better to have a value 300-500
        surf.hessianThreshold = 50000
        img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
        plt.imshow(img2),plt.show()
        
        
    def ORB(self):
        img = cv2.imread('ORB.jpg',0)
        # Initiate STAR detector
        orb = cv2.ORB()
        # find the keypoints with ORB
        kp = orb.detect(img,None)
        # compute the descriptors with ORB
        kp, des = orb.compute(img, kp)
        # draw only keypoints location,not size and orientation
        img2 = cv2.drawKeypoints(img,kp,color=(0,255,0), flags=0)
        plt.imshow(img2),plt.show()
        
    def BRIEF(self):
        img = cv2.imread('BRIEF.jpg',0)
        # Initiate STAR detector
        star = cv2.FeatureDetector_create("STAR")
        # Initiate BRIEF extractor
        brief = cv2.DescriptorExtractor_create("BRIEF")
        # find the keypoints with STAR
        kp = star.detect(img,None)
        # compute the descriptors with BRIEF
        kp, des = brief.compute(img, kp)
        print (brief.getInt('bytes'))
        print (des.shape)
        
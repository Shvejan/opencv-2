# -*- coding: utf-8 -*-

import cv2
import numpy as np

image = cv2.imread('images/house.jpg')
orig_image = cv2.pyrDown(image)

gray = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
frame1=np.zeros_like(orig_image)
frame2=np.zeros_like(orig_image)
for c in contours:
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(orig_image,(x,y),(x+w,y+h),(0,0,255),2)    
    accuracy = 0.03*cv2.arcLength(c, True)
    points=cv2.convexHull(c)
    points1 = cv2.approxPolyDP(c,accuracy,True)
    
    cv2.drawContours(frame1,[points1],-1,(0,0,255),3)
    cv2.drawContours(frame2,[points],-1  ,(0,0,255),3)
    cv2.imshow('approx',frame1)
    cv2.imshow('hull',frame2)
    cv2.waitKey(0)



cv2.destroyAllWindows()
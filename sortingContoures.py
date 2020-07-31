# -*- coding: utf-8 -*-
import cv2
import numpy as np

def findConto(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(grey,50,200)
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    return contours

def drawContByArea(image,contours):
    srted_contours = sorted(contours,key=cv2.contourArea)
    cv2.imshow('new',image)
    cv2.waitKey(0)
    for cnt in srted_contours:
        cv2.drawContours(image,cnt,-1,(0,0,255),3)
        cv2.imshow('new',image)
        cv2.waitKey(0)

image = cv2.imread('images/bunchofshapes.jpg')
smaller = cv2.pyrDown(image)
copy = smaller.copy()
blank = np.zeros_like(copy)
contours = findConto(copy)
drawContByArea(blank,contours)
cv2.destroyAllWindows()

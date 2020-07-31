# -*- coding: utf-8 -*-

import cv2
import numpy as np

def get_cont(image):
    
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cannny=cv2.Canny(grey,50,200)
    contours, hierarchy = cv2.findContours(cannny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours

def x_cord(cnt):
    if(cv2.contourArea(cnt)>10):
        M=cv2.moments(cnt)
        return (int(M['m10']/M['m00']))
    else:
        pass

def y_cord(cnt):
    if(cv2.contourArea(cnt)>10):
        M=cv2.moments(cnt)
        return (int(M['m01'] / M['m00']))
    else:
        pass
    
        

def draw(contours,img):
    for i,cnt in enumerate(contours):
        cv2.drawContours(img,cnt,-1,(0,10,200),5)
        cv2.putText(img,str(i+1),(x_cord(cnt),y_cord(cnt)),cv2.FONT_HERSHEY_COMPLEX,1,(30,40,90),4)
        cv2.imshow('sorted',img)
        
        cv2.waitKey(0)


image = cv2.imread('images/bunchofshapes.jpg')
smaller = cv2.pyrDown(image)
copy = smaller.copy()
blank = np.zeros_like(copy)
c=get_cont(copy)

sorted = sorted(c,key=x_cord)

draw(sorted,smaller)
cv2.destroyAllWindows()
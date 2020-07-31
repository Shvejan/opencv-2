import cv2
import numpy as np

# Load the shape template or reference image
template = cv2.imread('images/4star.jpg',0)

# Load the target image with the shapes we're trying to match
target = cv2.imread('images/shapestomatch.jpg')
#target = cv2.imread('images/someshapes.jpg')

target_gray = cv2.cvtColor(target,cv2.COLOR_BGR2GRAY)

# Threshold both images first before using cv2.findContours
ret, thresh1 = cv2.threshold(template, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh2 = cv2.threshold(target_gray, 127, 255,  cv2.THRESH_BINARY_INV)
cv2.imshow('1',thresh1)
cv2.imshow('2',thresh2)
cv2.waitKey(0)
    

contours1, hierarchy=cv2.findContours(thresh1, cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
contours2, hierarchy = cv2.findContours(thresh2,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

matched_list = []

for c in contours2:
    percentage = cv2.matchShapes(contours1[0],c,3,0.0)
    if(percentage<0.15):
        matched_list.append(c)
        

cv2.drawContours(target, matched_list, -1, (0,0,255), 3)
cv2.imshow('2',target)

cv2.waitKey(0)
cv2.destroyAllWindows()
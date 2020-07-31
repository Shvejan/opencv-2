import cv2
import numpy as np


# Load the target image with the shapes we're trying to match
target = cv2.imread('images/someshapes.jpg') 

grey = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
ret,threshold = cv2.threshold(grey,50,255,cv2.THRESH_BINARY_INV)

contours,hierarchy = cv2.findContours(threshold,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
blank = np.zeros_like(target)
    
for c in contours:
    cv2.drawContours(blank, c, -1, (0,0,255), 3)
    accuracy = 0.01*cv2.arcLength(c, True)
    points = cv2.approxPolyDP(c,accuracy,True)
    M = cv2.moments(c)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    
    if(len(points)==3):
        cv2.putText(blank,'Triangle',(cx,cy),cv2.FONT_HERSHEY_COMPLEX,1,(30,40,90),4)
    elif(len(points)==4):
        x,y,w,h = cv2.boundingRect(c)
        if(w-h<3):
            cv2.putText(blank,'Square',(cx,cy),cv2.FONT_HERSHEY_COMPLEX,1,(30,40,90),4)
        else:
            cv2.putText(blank,'Rectangle',(cx,cy),cv2.FONT_HERSHEY_COMPLEX,1,(30,40,90),4)
    elif(len(points)==10):
        cv2.putText(blank,'Star',(cx,cy),cv2.FONT_HERSHEY_COMPLEX,1,(30,40,90),4)
        
    elif(len(points)==15):
        cv2.putText(blank,'Circle',(cx,cy),cv2.FONT_HERSHEY_COMPLEX,1,(30,40,90),4)
    
    cv2.imshow('thresh',blank)
    cv2.waitKey(0)



cv2.destroyAllWindows()
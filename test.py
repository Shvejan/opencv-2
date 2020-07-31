# -*- coding: utf-8 -*-

import cv2
import requests
video = cv2.VideoCapture(0)

while True:
    _,frame = video.read()


    canney = cv2.Canny(frame,80,100)
    bcanny = cv2.Canny(cv2.GaussianBlur(frame,(7,7), 0)  ,80,100)
    _,thresh = cv2.threshold(bcanny, 70, 255, cv2.THRESH_BINARY_INV)
    
    cv2.imshow('video1',bcanny)
    cv2.imshow('video',thresh)
    
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break

video.release()
cv2.destroyAllWindows()

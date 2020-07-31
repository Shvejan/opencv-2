import cv2
import numpy as np;
 
image = cv2.imread("images/blobs.jpg")
detector= cv2.SimpleBlobDetector_create()

keypoints = detector.detect(image)

blobs=cv2.drawKeypoints(image, keypoints,np.zeros((1,1)),(255,0,0),cv2.DRAW_MATCHES_FLAGS_DEFAULT)

cv2.imshow("Blobs", blobs)
cv2.waitKey(0)
cv2.destroyAllWindows()
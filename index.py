# pip install opencv-contrib-python==3.4.0.12
import cv2
import numpy as np
import matplotlib.pyplot as plt


# example from https://docs.opencv.org/3.3.0/dc/dc3/tutorial_py_matcher.html

def imshow(img):
    print("* show image...")
    cv2.imshow("image", img)
    print("* Waiting for any keyboard command...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


print("load image 1...")
img1 = cv2.imread("vosamples/vo01.jpg")
gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)

print("load image 2...")
img2 = cv2.imread("vosamples/vo02.jpg")
gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

print("initiate sift detector...")
sift = cv2.xfeatures2d.SIFT_create()

print("Process sift example...")
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

print("points 1")
print(kp1)
print("points 2")
print(kp2)


print("Initiate brute force matching descriptor")
bf = cv2.BFMatcher();
matches = bf.knnMatch(des1, des2, k=2)

print("Apply ratio test")
good = []
for m, n in matches:
    if m.distance < 0.3* n.distance:
        good.append([m])

img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

imshow(img3)

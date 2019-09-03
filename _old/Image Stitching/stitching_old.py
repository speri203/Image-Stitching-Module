import cv2
import numpy as np
import os

os.chdir('images/5 meters')
img_ = cv2.imread('DJI_0004.jpg', cv2.COLOR_BGR2GRAY)
img = cv2.imread('DJI_0009.jpg', cv2.COLOR_BGR2GRAY)



#Setting up orb key point detector
orb = cv2.ORB_create()

#using orb to compute keypoints and descriptors
kp, des = orb.detectAndCompute(img_, None)
kp2, des2 = orb.detectAndCompute(img, None)
print(len(kp))
print(len(kp2))

#Setting up BFmatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
matches = bf.knnMatch(des, des2, k=2) #Find 2 best matches for each descriptors (This is required for ratio test?)

#Using lowes ratio test as suggested in paper at .7-.8
good = []
for m in matches:
    if m[0].distance < .8 * m[1].distance:
        good.append(m)
matches = np.asarray(good) #matches is essentially a list of matching descriptors
print(len(matches))
#Aligning the images
if(len(matches)) >= 4:
    src = np.float32([kp[m.queryIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)

    #Creating the homography and mask
    H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    print(H)
else:
    print("Could not find 4 good matches to find homography")

dst = cv2.warpPerspective(img_, H, (img.shape[1] + 500, img.shape[0]))
dst[0:img.shape[0], 0:img.shape[1]] = img
cv2.imwrite("Output2.jpg", dst)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
px = gray[100, 100]
print(px)






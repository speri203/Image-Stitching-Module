import cv2
import numpy as np

'''
TODO: Test and see if having the same number of kp in img1, img2 makes a difference when stitching
TODO: Test to see if the intensity of pixel makes a difference when finding keypoints
'''
def sift_keypoint(imgA, imgB):
    sift = cv2.xfeatures2d.SIFT_create(1000)
    kp, desc = sift.detectAndCompute(imgA, None)
    kp2, desc2 = sift.detectAndCompute(imgB, None)

    img = cv2.drawKeypoints(imgA, kp, imgA)
    cv2.imwrite('sift_kp (Light Corner).jpg', img)

    img2 = cv2.drawKeypoints(imgB, kp2, imgB)
    cv2.imwrite('sift_kp2 (Light Corner).jpg', img2)

    print("(SIFT \"Light Corner\") Left img: kp1: {} Right img: kp2: {}".format(len(kp), len(kp2)))

def surf_keypoint(imgA, imgB):
    surf = cv2.xfeatures2d.SURF_create()
    kp, desc = surf.detectAndCompute(imgA, None)
    kp2, desc2 = surf.detectAndCompute(imgB, None)

    img = cv2.drawKeypoints(imgA, kp, imgA)
    cv2.imwrite('surf_kp (Light Corner).jpg', img)

    img2 = cv2.drawKeypoints(imgB, kp2, imgB)
    cv2.imwrite('surf_kp2 (Light Corner).jpg', img2)

    print("(SURF \"Light Corner\") Left img: kp1: {} Right img: kp2: {}".format(len(kp), len(kp2)))

def orb_keypoint(imgA, imgB):
    orb = cv2.ORB_create()
    kp, desc = orb.detectAndCompute(imgA, None)
    kp2, desc2 = orb.detectAndCompute(imgB, None)

    img = cv2.drawKeypoints(imgA, kp, imgA)
    cv2.imwrite('orb_kp  (Light Corner).jpg', img)

    img2 = cv2.drawKeypoints(imgB, kp2, imgB)
    cv2.imwrite('orb_kp2 (Light Corner).jpg', img2)

    print("(ORB \"Light Corner\") Left img: kp1: {} Right img: kp2: {}".format(len(kp), len(kp2)))

def main():
    imgA = cv2.imread('be_bad.jpg', 0)
    imgB = cv2.imread('should.jpg', 0)
    sift_keypoint(imgA, imgB)
    surf_keypoint(imgA, imgB)
    orb_keypoint(imgA, imgB)
    return 0

main()


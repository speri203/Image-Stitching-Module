import stitching_module as stitching
import os
import cv2
from matplotlib import pyplot as plt

imgA = 'columbia_detail.jpg'
imgB = 'columbia_detail_next.jpg'



images = stitching.load_images(imgA, imgB)

chessboard_images = []
filepath = './chessboard_images'
os.chdir(filepath)
for files in os.listdir():
    img = cv2.imread(str(files))
    chessboard_images.append(img)

stitching.camera_calibration(chessboard_images)


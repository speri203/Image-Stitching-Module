'''
Initial check to make sure that the proper python packages are installed on the environment.
Required packages: opencv-contrib-python==3.4.2.14+, tabulate, scikit-image
'''
try:
    import cv2
    import numpy as np
    import matplotlib
    import importlib.util
    import sys
    from matplotlib import pyplot as plt
    from tabulate import tabulate
    import csv
    import os
except ImportError as e:
        print(e.__class__.__name__ + ': ' + e.name)
        exit(1)

'''
Check to make sure script is being run as a module and not as main
'''
if __name__ == '__main__':
    print("This script is supposed to be ran as a module and not as a main program. Please import it and use its functionality that way.")
    exit(1)

def load_images(imgLeft, imgRight):
    '''
    load_images function takes in two parameters for the left and right image.  It loads these images using the
    opencv cv2.imread() function. The images need to be in black and white otherwise errors will be spit at you.
    Conversion to black and white can be done with cv2.COLOR_BGR2GRAY and GRAY2BGR for inverse.
    :param imgLeft: File path of the left image
    :param imgRight: File path for the right image
    :return: list of two images (left=[0] and right=[1])
    '''
    imgA = cv2.imread(imgLeft, cv2.COLOR_BGR2GRAY)
    imgB = cv2.imread(imgRight, cv2.COLOR_BGR2GRAY)
    images = [imgA, imgB]
    return images

def orb_keypoint_detector(images):
    '''
    orb_keypoint_detector takes in images and computes the keypoints and descriptors on the individual images. Since
    we are assuming two images at a time (left, right) we will only have a maximum of 4 data points from this function
    keypoints 1 and descriptors ([0][0], [0][1]) and keypoints 2 and descriptors ([1][0], [1][1]). ORB (Oriented FAST and Rotational Brisk)
    :param images: list of two images (left, right)
    :return: A list of lists containing keypoints and descriptors for each image
    '''
    orb = cv2.ORB_create(nfeatures=1000, edgeThreshold=31) #(nfeatures, edgeThreshold)
    kp, des = orb.detectAndCompute(images[0], None)
    kp2, des2 = orb.detectAndCompute(images[1], None)
    keypoints_descriptors = [[kp, des], [kp2, des2]]
    return keypoints_descriptors

def draw_keypoints(keypoints_descriptors, images):
    '''
    Function takes in key points detected by the ORB algorithm (orb_keypoint_detector(images)) and draws the points on the images.
    This function then displays them side by side.
    :param keypoints_descriptors: Key points and descriptors calculated by the ORB algorithm
    :param images: list of images (left, right)
    :return: None
    '''
    imgA = cv2.drawKeypoints(images[0], keypoints_descriptors[0][0], None, color=(0,255,0))
    imgB = cv2.drawKeypoints(images[1], keypoints_descriptors[1][0], None, color=(255, 0, 0))
    f = plt.figure()
    f.add_subplot(1, 2, 1)
    plt.imshow(imgA)
    f.add_subplot(1, 2, 2)
    plt.imshow(imgB)
    plt.show(block=True)

def match_keypoint(keypoints_descriptors, images):
    '''
    Function takes two set of key points and descriptors and determins a good match based on BruteForce Matcher. From these matches,
    good results are filtered out using the ratio test talked about in Lowes paper. A distance of .7 - .8 should be selected.
    :param keypoints_descriptors: Key points and descriptors calculated by the ORB algorithm
    :param images: list of images (left, right)
    :return: A list of good matches between the two images after Lowes test.
    '''
    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(keypoints_descriptors[0][1], keypoints_descriptors[1][1], k=2) #Find 2 best matches for each descriptors (This is required for ratio test?)

    #applying ratio test
    good = []
    for m in matches:
        if m[0].distance < .8 * m[1].distance:
            good.append(m)
    matches = np.asarray(good) #matches is essentially a list of matching descriptors

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       flags=2)
    img3 = cv2.drawMatchesKnn(images[0],keypoints_descriptors[0][0],images[1],keypoints_descriptors[1][0], good, None, flags=2)
    print(tabulate([['Keypoints A: ', '{:.4f}'], ['Keypoints B: ', '{:.4f}'], ['Matches: ', '{:.4f}'], ['Good Matches: ', '{:.4f}']], tablefmt='orgtbl').format(len(keypoints_descriptors[0][0]), len(keypoints_descriptors[1][0]), len(matches), len(good)))
    plt.imshow(img3), plt.show()
    return matches

def warping_image(keypoints_descriptor, good, images):
    if len(good) > 4:
        kp1 = keypoints_descriptor[0][0]
        kp2 = keypoints_descriptor[1][0]
        src = np.float32([kp1[m.queryIdx].pt for m in good[:, 0]]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good[:, 0]]).reshape(-1, 1, 2)

        h, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

        output = cv2.warpPerspective(images[1], h, (images[1].shape[0] *2, images[1].shape[1]*2))
        plt.imshow(output), plt.show()

def camera_calibration(chessboard_images):
    '''
    This function takes in an array of images containing a chessboad moved in the x, y axis. From these images corners are
    detected in order to create a camera calibration matrix. This matrix is used on the thermal images inorder to undistort
    them which may be caused by the camera lense. This is currently a test to see if stitching fetches better results.
    :param chessboard_images: List of images containing chessboard already loaded up in openCV
    :return: Returns the mtx and dist which are used in the cv2.undistort() function to undistort images
    '''
    _3d_points = []
    _2d_points = []
    x, y = np.meshgrid(range(7), range(6))
    world_points = np.hstack((x.reshape(42, 1), y.reshape(42, 1), np.zeros((42, 1)))).astype(np.float32)
    #Calculating the corners of key points on chessboard
    for images in chessboard_images:
        ret, corners = cv2.findChessboardCorners(images, (7, 6))

        if(ret):
            _2d_points.append(corners)
            _3d_points.append(world_points)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(_3d_points, _2d_points, (images.shape[1],images.shape[0]), None, None)
    return mtx, dist

def bw_to_ironbow(image, csvFile):
    '''
    Function takes in a image (in our case usually black & white) and converts the pixel temperatues to ironbow color palette
    In order for this function to work the csv file of temperatues along with the cooresponding image file needs to be passed in
    :param image: Image file location to be loaded up by openCV
    :param csvFile: CSV file containing the temperatues of all pixels
    :return: Saved image
    '''
    img = cv2.imread(image, cv2.COLOR_BGR2GRAY)
    ironbow_color_values = [[0, 0, 0], [0, 0, 36], [0, 0, 51], [0, 0, 66], [0, 0, 81], [2, 0, 90], [4, 0, 99], [7, 0, 106], [11, 0, 115], [14, 0, 119], [20, 0, 123], [27, 0, 128], [33, 0, 133], [41, 0, 137], [48, 0, 140], [55, 0, 143], [61, 0, 146], [66, 0, 149], [72, 0, 150], [78, 0, 151], [84, 0, 152], [91, 0, 153], [97, 0, 155], [104, 0, 155], [110, 0, 156], [115, 0, 157], [122, 0, 157], [128, 0, 157], [134, 0, 157], [139, 0, 157], [146, 0, 156], [152, 0, 155], [157, 0, 155], [162, 0, 155], [167, 0, 154], [171, 0, 153], [175, 1, 152], [178, 1, 151], [182, 2, 149], [185, 4, 149], [188, 5, 147], [191, 6, 146], [193, 8, 144], [195, 11, 142], [198, 13, 139], [201, 17, 135], [203, 20, 132], [206, 23, 127], [208, 26, 121], [210, 29, 116], [212, 33, 111], [214, 37, 103], [217, 41, 97], [219, 46, 89], [221, 49, 78], [223, 53, 66], [224, 56, 54], [226, 60, 42], [228, 64, 30], [229, 68, 25], [231, 72, 20], [232, 76, 16], [234, 78, 12], [235, 82, 10], [236, 86, 8], [237, 90, 7], [238, 93, 5], [239, 96, 4], [240, 100, 3], [241, 103, 3], [241, 106, 2], [242, 109, 1], [243, 113, 1], [244, 116, 0], [244, 120, 0], [245, 125, 0], [246, 129, 0], [247, 133, 0], [248, 136, 0], [248, 139, 0], [249, 142, 0], [249, 145, 0], [250, 149, 0], [251, 154, 0], [252, 159, 0], [253, 163, 0], [253, 168, 0], [253, 172, 0], [254, 176, 0], [254, 179, 0], [254, 184, 0], [254, 187, 0], [254, 191, 0], [254, 195, 0], [254, 199, 0], [254, 202, 1], [254, 205, 2], [254, 208, 5], [254, 212, 9], [254, 216, 12], [255, 219, 15], [255, 221, 23], [255, 224, 32], [255, 227, 39], [255, 229, 50], [255, 232, 63], [255, 235, 75], [255, 238, 88], [255, 239, 102], [255, 241, 116], [255, 242, 134], [255, 244, 149], [255, 245, 164], [255, 247, 179], [255, 248, 192], [255, 249, 203], [255, 251, 216], [255, 253, 228], [255, 254, 239], [255, 255, 249]]
    min_temp = 1000
    max_temp = 0
    pixel_temperature = []
    with open(csvFile) as csvData:
        reader = csv.reader(csvData, delimiter=',', quotechar='|')  # Seperated by comma (hence csv)
        for i, data in enumerate(reader):
            if (i >= 10):
                #print(data[1:])
                pixel_temperature.append(data[1:]) #Structure will be pixel_temperature[x][y] or pixel_temperature[512-1][640-1]

    for x in range(512):
        for y in range(640):
            if(min_temp > float(pixel_temperature[x][y])):
                min_temp = float(pixel_temperature[x][y])
            if (max_temp < float(pixel_temperature[x][y])):
                max_temp = float(pixel_temperature[x][y])

    average_pixel_temperature_change = (max_temp - min_temp) / len(ironbow_color_values)

    for x in range(512):
        for y in range(640):
            color_index = (float(pixel_temperature[x][y]) - min_temp) / average_pixel_temperature_change
            img[x, y] = ironbow_color_values[int(color_index) - 1]
    #plt.imshow(img), plt.show()
    filename = csvFile.split('.')
    plt.imsave('./ironbow/ ' + filename[0] + '_ironbow.jpg', img)


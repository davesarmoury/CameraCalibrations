## https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html
## https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/chessboard.png

import cv2
import numpy as np
import os
from tqdm import tqdm

# Parameters
IMAGES_DIR = 'images/hero8/'
SQUARE_SIZE = 2.5
WIDTH = 8
HEIGHT = 11
FN = 'hero8'

def calibrate_chessboard(dir_path, square_size, width, height):
    '''Calibrate a camera using chessboard images.'''
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = os.listdir(dir_path)
    print(images)

    # Iterate through all images
    for fname in tqdm(images):
        img = cv2.imread(dir_path + str(fname))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img2 = cv2.resize(img, (1280, 720))
        cv2.imshow('img', img2)
        cv2.waitKey(5)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            cv2.drawChessboardCorners(img, (HEIGHT, WIDTH), corners2, ret)
            img = cv2.resize(img, (1280, 720))
            cv2.imshow('img', img)
            cv2.waitKey(1000)

    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return [ret, mtx, dist, rvecs, tvecs]

def save_coefficients(mtx, dist, path):
    '''Save the camera matrix and the distortion coefficients to given path/file.'''
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write('K', mtx)
    cv_file.write('D', dist)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()

# Calibrate
ret, mtx, dist, rvecs, tvecs = calibrate_chessboard(IMAGES_DIR, SQUARE_SIZE, WIDTH, HEIGHT)

# Save coefficients into a file
save_coefficients(mtx, dist, FN + "_calibration.yaml")

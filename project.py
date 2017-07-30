import cv2
import glob
import numpy as np
from math import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
# Import Video clip creator
import imageio as imageio

"""
Grr, why do I have to do that explicitly? It seems that altohugh ffmpeg.exe
located on several locations it isn't found automatically. An explicit download 
is required. The statement below dowloads and stores under:
C:/Users/paulm/AppData/Local/imageio/ffmpeg  
"""
imageio.plugins.ffmpeg.download()

from moviepy.editor import ImageSequenceClip
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def calibrate_camera(cal_images, nx, ny):
    """Calibrate camera from given chessboard pattern images. 
    """
    objpoints = []  # 3D points
    imgpoints = []  # 2D points

    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1, 2)

    fname = cal_images[0]
    for fname in cal_images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, 
                                                       imgpoints, 
                                                       gray.shape[::-1],
                                                       None, None)
    
    return mtx, dist

def undistort_image(imgage, cam_cal_mtx, cam_cal_dist):
    """Undistort given image according to camera calibration parameters
    """
    return cv2.undistort(image, cam_cal_mtx, cam_cal_dist, None, cam_cal_mtx)

def calibrate_camera_from_data():
    cal_images = glob.glob('camera_cal/calibration*.jpg')
    nx, ny = 9, 6
    return calibrate_camera(cal_images, nx, ny)

def test_after_camera_calibration_images_should_be_undistorted(image,
                                                               cam_mtx,
                                                               cam_dist,
                                                               figsize=(10,5)): 
    undist_image = cv2.undistort(image, cam_mtx, cam_dist, None, cam_mtx)
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.xlabel('Original image')
    plt.xticks([], [])
    plt.yticks([], [])

    plt.subplot(1, 2, 2)
    plt.imshow(undist_image)
    plt.xlabel('undist_imageorted image')
    plt.xticks([], [])
    plt.yticks([], [])
    plt.show()
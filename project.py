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

def calibrate_camera_from_data():
    cal_images = glob.glob('camera_cal/calibration*.jpg')
    nx, ny = 9, 6
    cam_mtx, cam_dst = calibrate_camera(cal_images, nx, ny)
    return cam_mtx, cam_dst, nx, ny

class CameraCalibration:
    cam_mtx = None
    cam_dst = None
    nx = None
    ny = None
    def __init__(self, *args, **kwargs):
        super(CameraCalibration, self).__init__(*args, **kwargs)
        self.cam_mtx, self.cam_dst, self.nx, self.ny = calibrate_camera_from_data()

def undistort(image, cam_cal):
    """Undistort given image according to camera calibration parameters
    """
    return cv2.undistort(image, cam_cal.cam_mtx, cam_cal.cam_dst, None, cam_cal.cam_mtx)

def transform_to_birds_eye_perspective(image, cam_cal, offset = 100):
    """Calculate perspective transform for an image given chessboard corners and camera parameters 
    """
    # 1) Undistort using camera calibration data
    u_image = undistort(image, cam_cal)
    # 2) Convert to grayscale
    g_image = cv2.cvtColor(u_image, cv2.COLOR_BGR2GRAY)
    # 3) Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(g_image, (cam_cal.nx, cam_cal.ny), None)
    p_image, p_mat = None, None
    if ret == True:
            # 4) If corners found: 
            # a) draw corners
            #cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
            # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
                #Note: you could pick any four of the detected corners 
                # as long as those four corners define a rectangle
                #One especially smart way to do this would be to use four well-chosen
                # corners that were automatically detected during the undistortion steps
                #We recommend using the automatic detection of corners in your code
            src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
            # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
            w, h = g_image.shape[1], g_image.shape[0]
            dst = np.float32([[offset, offset], [w-offset, offset], 
                             [w-offset, h-offset], 
                             [offset, h-offset]])
            # d) use cv2.getPerspectiveTransform() to get p_mat, the transform matrix
            p_mat = cv2.getPerspectiveTransform(src, dst)
            # e) use cv2.warpPerspective() to warp your image to a top-down view
            # Warp the image using OpenCV warpPerspective()
            p_image = cv2.warpPerspective(u_image, p_mat, (w, h))
    return p_image, p_mat
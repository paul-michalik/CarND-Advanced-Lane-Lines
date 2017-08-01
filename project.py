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

class CameraCalibration:
    cam_mtx = None
    cam_dst = None
    nx = None
    ny = None
    def __init__(self, *args, **kwargs):
        self.cam_mtx, self.cam_dst, self.nx, self.ny = self.__calibrate_camera_from_data()

    def __calibrate_camera_from_data(self):
        cal_images = glob.glob('camera_cal/calibration*.jpg')
        nx, ny = 9, 6
        cam_mtx, cam_dst = self.__calibrate_camera(cal_images, nx, ny)
        return cam_mtx, cam_dst, nx, ny

    def __calibrate_camera(self, cal_images, nx, ny):
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

    def undistort(self, image):
        """Undistort given image according to camera calibration parameters
        """
        return cv2.undistort(image, self.cam_mtx, self.cam_dst, None, self.cam_mtx)

class BirdsEyeView:
    src = None
    dst = None
    p_mat = None
    def __init__(self):
        ref_image = mpimage.imread('test_images/straight_lines1.jpg')
        self.p_mat, self.src, self.dst = self.__birds_eye_perspective(ref_image)

    def __birds_eye_perspective(self, image):
        """Calculate perspective transform for a road image from the test set 
        """
        # define source and destination points for transform
        w, h = image.shape[1], image.shape[0]
        src = np.float32([(575,464),
                          (707,464),
                          (258,682), 
                          (1049,682)])
        dst = np.float32([(450,0),
                          (w-450,0),
                          (450,h),
                          (w-450,h)])
        # use cv2.getPerspectiveTransform() to get p_mat, the transform matrix
        p_mat = cv2.getPerspectiveTransform(src, dst)
        return p_mat, src, dst

    def transform(self, image):
        """transform an image given the perspective matrix 
        """
        # e) use cv2.warpPerspective() to warp your image to a top-down view
        # Warp the image using OpenCV warpPerspective()
        w, h = image.shape[1], image.shape[0]
        return cv2.warpPerspective(image, self.p_mat, (w, h))
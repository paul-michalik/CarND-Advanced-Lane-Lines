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
    mtx = None
    dst = None
    nx = None
    ny = None
    cal_images = None

    def __init__(self, mtx, dst, nx, ny, cal_images):
        self.mtx, self.dst, self.nx, self.ny, self.cal_images = mtx, dst, nx, ny, cal_images

    def undistort(self, image):
        """Undistort given image according to camera calibration parameters
        """
        return cv2.undistort(image, self.mtx, self.dst, None, self.mtx)

def calibrate_camera(cal_images, nx, ny):
    """Calibrate camera from given chessboard pattern images. returns CameraCalibration object
    """
    objpoints = []  # 3D points
    imgpoints = []  # 2D points

    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1, 2)

    for fname in cal_images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dst, rvecs, tvecs = cv2.calibrateCamera(objpoints, 
                                                      imgpoints, 
                                                      gray.shape[::-1],
                                                      None, None)

    cam_cal = CameraCalibration(mtx=mtx, dst=dst, nx=nx, ny=ny, cal_images=cal_images)
    return cam_cal
        
def calibrate_camera_init():
    nx, ny, cal_images = 9, 6, glob.glob('camera_cal/calibration*.jpg')
    cam_cal = calibrate_camera(cal_images, nx, ny)
    return cam_cal

class BirdsEyeView:
    src = None
    dst = None
    p_mat = None
    ref_image = None

    def __init__(self, src, dst, p_mat, ref_image):
        self.p_mat, self.src, self.dst, ref_image = src, dst, p_mat, ref_image

    def src_vertices_as_region_for_polyFill(self):
        return np.int32([src])

    def transform(self, image):
        """transform an image given the perspective matrix 
        """
        # e) use cv2.warpPerspective() to warp your image to a top-down view
        # Warp the image using OpenCV warpPerspective()
        w, h = image.shape[1], image.shape[0]
        return cv2.warpPerspective(image, self.p_mat, (w, h))

def birds_eye_view(ref_image):
    """Calculate perspective transform for a road image from the test set 
    """
    # define source and destination points for transform
    w, h = ref_image.shape[1], ref_image.shape[0]
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
    return BirdsEyeView(src=src, dst=dst, p_mat=p_mat, ref_image=ref_image) 

def birds_eye_view_init(cam_cal, ref_image_fname = 'test_images/straight_lines1.jpg'):
    ref_image = mpimage.imread(ref_image_fname)
    return birds_eye_view(cam_cal.undistort(ref_image))

def threshold_transform(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.zeros_like(sxbinary)
    color_binary[(sxbinary == 1) | (s_binary == 1)] = 255
    return color_binary

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

"""
Tests
"""

def draw_before_after(image_before, 
                      image_after, 
                      txt_before='Original image',
                      txt_after='Transformed image',
                      cmap=None,
                      figsize=(10,5)):
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.imshow(image_before)
    plt.xlabel(txt_before)
    plt.xticks([], [])
    plt.yticks([], [])

    plt.subplot(1, 2, 2)
    plt.imshow(image_after, cmap=cmap)
    plt.xlabel(txt_after)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.show()

def images_should_be_undistorted(camera_calibration, image_file):
    image = mpimage.imread(image_file)        
    undist_image = camera_calibration.undistort(image)
    draw_before_after(image, undist_image, txt_after='Undistorted image')
    return image, undist_image
        
def images_should_be_transformed_to_birds_eye_perspective(camera_calibration, birds_eye_view, image_file):
    image = mpimage.imread(image_file)
    p_image = birds_eye_view.transform(camera_calibration.undistort(image))
    draw_before_after(image, p_image, txt_after='Birds eye perspective')
    return image, p_image    

def images_after_color_transform_should_acentuate_lanes(image_file):
    image = mpimage.imread(image_file)
    t_image = threshold_transform(image)
    draw_before_after(image, t_image, txt_after='Binarized image', cmap='gray')
    return image, t_image
   
def images_after_transform_show_lanes(camera_calibration, birds_eye_view, image_file):
    image = mpimage.imread(image_file)
    t_image = threshold_transform(\
        birds_eye_view.transform(\
        camera_calibration.undistort(image)))
    draw_before_after(image, t_image, txt_after='Tranformed image', cmap='gray')
    return image, t_image

def images_after_clipping_and_transform_show_lanes(camera_calibration, birds_eye_view, image_file):
    image = mpimage.imread(image_file)

    roi_vertices = birds_eye_view.src_vertices_as_region_for_polyFill()

    t_image = threshold_transform(\
        birds_eye_view.transform(\
        camera_calibration.undistort(\
        region_of_interest(image, roi_vertices))))
    draw_before_after(image, t_image, txt_after='Tranformed image', cmap='gray')
    return image, t_image

if __name__ == '__main__':
    cc = calibrate_camera_init()
    bb = birds_eye_view_init(cc)
    images_should_be_transformed_to_birds_eye_perspective(cc, bb, 'test_images/test1.jpg')
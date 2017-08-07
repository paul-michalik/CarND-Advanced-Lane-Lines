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
    p_inv_mat = None
    ref_image = None

    def __init__(self, src, dst, p_mat, p_inv_mat, ref_image):
        self.src, self.dst, self.p_mat, self.p_inv_mat, self.ref_image = src, dst, p_mat, p_inv_mat, ref_image


    def src_vertices_as_region_for_polyFill(self):
        roi_vertices = np.array([[\
            tuple(self.src[2]), 
            tuple(self.src[3]), 
            tuple(self.src[1]), 
            tuple(self.src[0])]], dtype = np.int32)
        delta_x_low = 30
        delta_x_high = 90
        roi_vertices[0][0][0] = roi_vertices[0][0][0] - delta_x_low
        roi_vertices[0][1][0] = roi_vertices[0][1][0] + delta_x_low
        roi_vertices[0][2][0] = roi_vertices[0][2][0] + delta_x_high
        roi_vertices[0][3][0] = roi_vertices[0][3][0] - delta_x_high
        return roi_vertices

    def dst_vertices_as_region_for_polyFill(self):
        # I hate numpy and the the stupid arrays...
        roi_vertices = np.array([[\
            tuple(self.dst[2]), 
            tuple(self.dst[3]), 
            tuple(self.dst[1]), 
            tuple(self.dst[0])]], dtype = np.int32)
        delta_x_low = 30
        delta_x_high = 90
        roi_vertices[0][0][0] = roi_vertices[0][0][0] - delta_x_low
        roi_vertices[0][1][0] = roi_vertices[0][1][0] + delta_x_low
        roi_vertices[0][2][0] = roi_vertices[0][2][0] + delta_x_high
        roi_vertices[0][3][0] = roi_vertices[0][3][0] - delta_x_high
        return roi_vertices

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
    p_inv_mat = cv2.getPerspectiveTransform(dst, src)
    return BirdsEyeView(src=src, dst=dst, p_mat=p_mat, p_inv_mat=p_inv_mat, ref_image=ref_image) 

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

class ImageProcessing:
    camera_calibration = None
    birds_eye_view = None

    def __init__(self, camera_calibration = None, birds_eye_view = None):
        if camera_calibration == None:
            self.camera_calibration = calibrate_camera_init()
        else:
            self.camera_calibration = camera_calibration

        if birds_eye_view == None:
            self.birds_eye_view = birds_eye_view_init(self.camera_calibration)
        else:
            self.birds_eye_view = birds_eye_view

    def apply(self, image):
        roi_vertices = self.birds_eye_view.dst_vertices_as_region_for_polyFill()
        return region_of_interest(\
            threshold_transform(\
            self.birds_eye_view.transform(\
            self.camera_calibration.undistort(image))), roi_vertices)

"""
Lane detection
"""

class LanePolyfit:
    left_fit = None
    right_fit = None
    left_lane_inds = None
    right_lane_inds = None
    out_img = None

    def __init__(self, 
                 left_fit, right_fit, 
                 left_lane_inds, right_lane_inds, 
                 out_img):
        self.left_fit, self.right_fit = left_fit, right_fit
        self.left_lane_inds, self.right_lane_inds = left_lane_inds, right_lane_inds
        self.out_img = out_img

    def nonzero_xy(self, image):
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        return nonzerox, nonzeroy

    def gen_xy_for_plotting(self, image):
        binary_warped = image
        left_fit, right_fit = self.left_fit, self.right_fit
        left_lane_inds, right_lane_inds = self.left_lane_inds, self.right_lane_inds

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        return ploty, left_fitx, right_fitx

    def rad_of_curvature_in_world_space(self, image):
        warped_image = image

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        ploty, left_fitx, right_fitx = self.gen_xy_for_plotting(warped_image)

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        # Now our radius of curvature is in meters
        return left_curverad, right_curverad

    def reproject(self, warped_image, org_image, p_mat_inv):
        warped = warped_image
        image = org_image
        Minv = p_mat_inv
        ploty, left_fitx, right_fitx = self.gen_xy_for_plotting(warped_image)
        
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        return result

    def draw_init(self, image):
        binary_warped = image
        left_fit, right_fit = self.left_fit, self.right_fit
        left_lane_inds, right_lane_inds = self.left_lane_inds, self.right_lane_inds
        out_img = self.out_img
        nonzerox, nonzeroy = self.nonzero_xy(image)

        # Generate x and y values for plotting
        ploty, left_fitx, right_fitx = self.gen_xy_for_plotting(binary_warped)

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='white')
        plt.plot(right_fitx, ploty, color='white')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()
    
    def draw_next(self, image):
        binary_warped = image
        left_fit, right_fit = self.left_fit, self.right_fit
        left_lane_inds, right_lane_inds = self.left_lane_inds, self.right_lane_inds
        nonzerox, nonzeroy = self.nonzero_xy(binary_warped)

        # Generate x and y values for plotting
        ploty, left_fitx, right_fitx = self.gen_xy_for_plotting(binary_warped)

        # Create an image to draw on and an image to show the selection window
        self.out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)

        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)

def lane_polyfit_sliding_window_init(image, per_cent_of_view=0.5, nwindows=9, margin = 100, minpix=50):
    """Calculate 
    """
    # Assuming you have created a warped binary image called "binary_warped"
    binary_warped = image
    
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]*(1. - per_cent_of_view)):,:], 
                       axis=0)    
    
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    # nwindows=9

    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set the width of the windows +/- margin
    #margin = 100
    # Set minimum number of pixels found to recenter window
    #minpix = 50

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img,
                      (win_xleft_low,win_y_low),
                      (win_xleft_high,win_y_high),
                      (0,255,0), thickness=3) 
        cv2.rectangle(out_img,
                      (win_xright_low,win_y_low),
                      (win_xright_high,win_y_high),
                      (0,255,0), thickness=3) 

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return LanePolyfit(left_fit, right_fit, left_lane_inds, right_lane_inds, out_img)

def lane_polyfit_sliding_window_next(image, lane_polyfit, margin=100):
    binary_warped = image
    left_fit, right_fit = \
        lane_polyfit.left_fit, lane_polyfit.right_fit
    left_lane_inds, right_lane_inds = \
        lane_polyfit.left_lane_inds, lane_polyfit.right_lane_inds

    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    #margin = 100

    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    return LanePolyfit(left_fit, right_fit, left_lane_inds, right_lane_inds, out_img)

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
    #plt.xticks([], [])
    #plt.yticks([], [])

    plt.subplot(1, 2, 2)
    plt.imshow(image_after, cmap=cmap)
    plt.xlabel(txt_after)
    #plt.xticks([], [])
    #plt.yticks([], [])
    plt.show()

def images_should_be_undistorted(camera_calibration, image_file):
    image = mpimage.imread(image_file)        
    undist_image = camera_calibration.undistort(image)
    draw_before_after(image, undist_image, txt_after='Undistorted image')
    #return image, undist_image
        
def images_should_be_transformed_to_birds_eye_perspective(camera_calibration, birds_eye_view, image_file):
    image = mpimage.imread(image_file)
    p_image = birds_eye_view.transform(camera_calibration.undistort(image))
    draw_before_after(image, p_image, txt_after='Birds eye perspective')
    #return image, p_image    

def images_after_color_transform_should_acentuate_lanes(image_file):
    image = mpimage.imread(image_file)
    t_image = threshold_transform(image)
    draw_before_after(image, t_image, txt_after='Binarized image', cmap='gray')
    #return image, t_image
   
def images_after_full_transform_show_lanes(camera_calibration, birds_eye_view, image_file):
    image = mpimage.imread(image_file)
    t_image = threshold_transform(\
        birds_eye_view.transform(\
        camera_calibration.undistort(image)))

    v = birds_eye_view.dst_vertices_as_region_for_polyFill()
    color = (0,255,255)
    cv2.line(t_image, tuple(v[0][0]), tuple(v[0][1]), color=color, thickness = 5)
    cv2.line(t_image, tuple(v[0][1]), tuple(v[0][2]), color=color, thickness = 5)
    cv2.line(t_image, tuple(v[0][2]), tuple(v[0][3]), color=color, thickness = 5)
    cv2.line(t_image, tuple(v[0][3]), tuple(v[0][0]), color=color, thickness = 5)
    draw_before_after(image, t_image, txt_after='Tranformed image, not clipped', cmap='gray')
    #return image, t_image

def images_after_full_transform_and_clipping_show_lanes(camera_calibration, birds_eye_view, image_file):
    image = mpimage.imread(image_file)

    roi_vertices = birds_eye_view.dst_vertices_as_region_for_polyFill()

    t_image = region_of_interest(\
        threshold_transform(\
        birds_eye_view.transform(\
        camera_calibration.undistort(image))), roi_vertices)
    draw_before_after(image, t_image, txt_after='Tranformed image, clipped', cmap='gray')
    #return image, t_image
 

def test_image_transform(camera_calibration, bird_eye_view):
    cc = camera_calibration
    bb = bird_eye_view
    images_should_be_undistorted(cc, 'camera_cal/calibration1.jpg')
    images_should_be_undistorted(cc, 'test_images/test1.jpg')
    images_should_be_transformed_to_birds_eye_perspective(cc, bb, 'test_images/test1.jpg')
    images_should_be_transformed_to_birds_eye_perspective(cc, bb, 'test_images/test2.jpg')
    images_after_color_transform_should_acentuate_lanes('test_images/test1.jpg')
    images_after_color_transform_should_acentuate_lanes('test_images/test2.jpg')

    img_proc = ImageProcessing(cc, bb)
    for fname in glob.glob('test_images/test*.jpg'):
        image = mpimage.imread(fname)
        draw_before_after(image, img_proc.apply(image), cmap='gray')

def test_lane_polyfit_on_image_sequence(camera_calibration, bird_eye_view):
    img_proc = ImageProcessing(cc, bb)
    for fname in glob.glob('test_images/test*.jpg'):
        image = mpimage.imread(fname)
        draw_before_after(image, img_proc.apply(image), cmap='gray')
        lane_pfit = lane_polyfit_sliding_window_init(t_image, 
                                                     per_cent_of_view=0.75, 
                                                     nwindows=4, 
                                                     margin=20, 
                                                     minpix=20)
        lane_pfit.draw_init(t_image)

if __name__ == '__main__':
    cc = calibrate_camera_init()
    bb = birds_eye_view_init(cc)
    
    #test_image_transform(cc, bb)
    
    img_proc = ImageProcessing(cc, bb)
    image = mpimage.imread('test_images/test1.jpg')
    t_image = img_proc.apply(image)
    draw_before_after(image, t_image, cmap='gray')
    lane_pfit = lane_polyfit_sliding_window_init(t_image, per_cent_of_view=0.75, nwindows=5, margin=20, minpix=20)
    lane_pfit.draw_init(t_image)

    test_lane_polyfit_on_image_sequence(cc, bb)


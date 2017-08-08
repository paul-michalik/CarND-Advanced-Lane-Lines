**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undist-chessboard.jpeg "Distorted vs. Undistorted chessboard"
[image2]: ./output_images/undist-road.jpeg "Road Transformed"
[image3]: ./output_images/binarized-2.jpeg "Binarized Example"
[image41]: ./output_images/birds-eye-1.jpeg "Birds eye view"
[image42]: ./output_images/birds-eye-3.jpeg "Birds eye view after binarization"
[image5]: ./output_images/polyfit-1.jpeg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the in lines 24 through 68 of the file called `project.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![Comparison between distorted and undistorited images][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![Distorted vs. undistorted comparison][image2]

I use the functions ``calibrate_camera_init`` and ``calibrate_camera`` to create an immutable object of type ``CameraCalibration``. This object provides a method ``undistort`` which hwen called well "undistorts" the image according to camera calibration. The camera was calibrated using the chessboard patterns provided by Udacity.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 140 through 163 in `project.py`) in function ``threshold_transform``. In addition, after the projective transform of the image, I zeroed everything outside of the expected region in the image using the method from the first lane recognition project. The implementation can be seen in functiobn ``region_of_interest`` at lines 165 through 187 in ``project.py``. Here's an example of my output for this step. 

![Binarized image][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a class called `BirdsEyeView` and functions called `birds_eye_view` and `birds_eye_view_init`, which appears in lines 70 through 138 in the file `project.py`.  The `birds_eye_view_init` function takes as inputs an reference image (`ref_img`), as well as a `CameraCalibration` object (`cam_cal`).  I chose the hardcode the source and destination points in the following manner (`w` and `h` denote the width and hight of the image respectively):

```python
src = np.float32([(575,464),
                  (707,464),
                  (258,682), 
                  (1049,682)])
dst = np.float32([(450,0),
                  (w-450,0),
                  (450,h),
                  (w-450,h)])
```

I verified that my perspective transform was working as expected by drawing the originakl and transformed images next to each other and verified that the lines appear parallel in the transofrmed image image.

![After perspective transform][image41]

The next image shows the images after transformation into bird's eye perspective and binarization:

![After perspective transform and binarization][image42]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I have used the "sliding windows" algorithm to identify the portions of the image which were most likely to contain the lane line pixels. This happens in the functions `lane_polyfit_sliding_window_init` which created an object of type `LanePolyFit`. Again these objects are immutable and store all data required to calculate the output. The parameters for the algorithm are obvious from the signature of the function. Other function `lane_polyfit_sliding_window_next` estimates the polynomials based on coherence of the previous results which are passed to the function via an `LanePolyFit` object.

The class `LanePolyFit` defines following additional methods (plus some debugging facilities):

* `reproject` projects the input image to the world space
* `rad_of_curvature_and_dist_in_world_space` calculates the required parameters
* `estimated_lane_width` estimates the width of the lane

The non-zero pixels inside each windows are identified and used as input data for 2nd degree approximation. The best values for the parameters (number of windows, margin width etc.) were found by experimentation.

![Demostration of the polynomial fitting][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

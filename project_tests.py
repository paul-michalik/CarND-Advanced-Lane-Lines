import unittest
import os
import project
import matplotlib.image as mpimage

class CameraCalibration(unittest.TestCase):
    cam_mtx = None
    cam_dst = None
    nx = None
    ny = None
    def __init__(self, *args, **kwargs):
        super(CameraCalibration, self).__init__(*args, **kwargs)
        self.cam_mtx, self.cam_dst, self.nx, self.ny = project.calibrate_camera_from_data()
        
    def draw_before_after(self, image_before, image_after):
        plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        plt.imshow(image_before)
        plt.xlabel('Original image')
        plt.xticks([], [])
        plt.yticks([], [])

        plt.subplot(1, 2, 2)
        plt.imshow(image_after)
        plt.xlabel('Tranformed image')
        plt.xticks([], [])
        plt.yticks([], [])
        plt.show()

    def images_should_be_undistorted(self, image_file, figsize=(10,5)):
        image = mpimage.imread(image_file)
        undist_image = project.undistort(image, self.cam_mtx, self.cam_dst)
        return image, undist_image
        
    def images_should_be_transformed_to_birds_eye_perspective(self, image_file, figsize=(10,5)):
        image = mpimage.imread(image_file)        
        p_image, _ = project.transform_to_birds_eye_perspective(image, 
                                                                    self.nx, 
                                                                    self.ny,
                                                                    self.cam_mtx,
                                                                    self.cam_dst)
        return image, p_image

        

#if __name__ == '__main__':
#    unittest.main()
import unittest
import project

class TestImageTransform(unittest.TestCase):
    cam_cal = None
    birds_eye_view = None
    def __init__(self, *args, **kwargs):
        super(TestImageTransform, self).__init__(*args, **kwargs)
        self.cam_cal = project.CameraCalibration()
        self.birds_eye_view = project.BirdsEyeView(self.cam_cal)

    def draw_before_after(self,
                          image_before, 
                          image_after, 
                          txt_before='Original image',
                          txt_after='Transformed image',
                          cmap=None,
                          figsize=(10,5)):
        project.plt.figure(figsize=figsize)
        project.plt.subplot(1, 2, 1)
        project.plt.imshow(image_before)
        project.plt.xlabel(txt_before)
        project.plt.xticks([], [])
        project.plt.yticks([], [])

        project.plt.subplot(1, 2, 2)
        project.plt.imshow(image_after, cmap=cmap)
        project.plt.xlabel(txt_after)
        project.plt.xticks([], [])
        project.plt.yticks([], [])
        project.plt.show()

    def images_should_be_undistorted(self, image_file):
        image = project.mpimage.imread(image_file)        
        undist_image = self.cam_cal.undistort(image)
        self.draw_before_after(image, undist_image, txt_after='Undistorted image')
        #return image, undist_image
        
    def images_should_be_transformed_to_birds_eye_perspective(self, image_file):
        image = project.mpimage.imread(image_file)
        p_image = self.birds_eye_view.transform(self.cam_cal.undistort(image))
        self.draw_before_after(image, p_image, txt_after='Birds eye perspective')
        #return image, p_image    

    def images_after_color_transform_should_acentuate_lanes(self, image_file):
        image = project.mpimage.imread(image_file)
        t_image = project.threshold_transform(image)
        self.draw_before_after(image, t_image, txt_after='Binarized image', cmap='gray')
        return image, t_image
   
    def images_after_full_transform_show_lanes(self, image_file):
        image = project.mpimage.imread(image_file)
        t_image = project.threshold_transform(\
            self.birds_eye_view.transform(\
            self.cam_cal.undistort(image)))
        self.draw_before_after(image, t_image, txt_after='Tranformed image', cmap='gray')
        return image, t_image
   

#if __name__ == '__main__':
#    test = TestImageTransform()
#    test.images_should_be_transformed_to_birds_eye_perspective('test_images/straight_lines1.jpg')
#    unittest.main()
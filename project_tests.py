import unittest
import project

def draw_before_after(image_before, image_after, figsize=(10,5)):
    project.plt.figure(figsize=figsize)
    project.plt.subplot(1, 2, 1)
    project.plt.imshow(image_before)
    project.plt.xlabel('Original image')
    project.plt.xticks([], [])
    project.plt.yticks([], [])

    project.plt.subplot(1, 2, 2)
    project.plt.imshow(image_after)
    project.plt.xlabel('Transformed image')
    project.plt.xticks([], [])
    project.plt.yticks([], [])
    project.plt.show()

class TestImageTransform(unittest.TestCase):
    cam_cal = None
    birds_eye = None
    def __init__(self, *args, **kwargs):
        super(TestImageTransform, self).__init__(*args, **kwargs)
        self.cam_cal = project.CameraCalibration()
        self.birds_eye = project.BirdsEyeView()

    def images_should_be_undistorted(self, image_file):
        image = project.mpimage.imread(image_file)        
        undist_image = self.cam_cal.undistort(image)
        draw_before_after(image, undist_image)
        return image, undist_image
        
    def images_should_be_transformed_to_birds_eye_perspective(self, image_file):
        image = project.mpimage.imread(image_file)
        p_image = self.birds_eye.transform(image)
        draw_before_after(image, p_image)
        return image, p_image    

#if __name__ == '__main__':
#    test = TestImageTransform()
#    test.images_should_be_transformed_to_birds_eye_perspective('test_images/straight_lines1.jpg')
#    unittest.main()
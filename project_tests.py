from project import *


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

#if __name__ == '__main__':
#    unittest.main()
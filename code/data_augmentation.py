import numpy as np
import cv2


def flip_center_image(data):
    image, steering_angle = data

    center_image_flipped = np.fliplr(image)
    steering_angle_flipped = - steering_angle
    return center_image_flipped, steering_angle_flipped


def histograms_equalization(data):
    image, steering_angle = data

    center_image_equalized = cv2.equalizeHist(image)
    return center_image_equalized, steering_angle

import numpy as np
import cv2

class Functions:

    @property
    def non_zero_angle_filter(self):
        return lambda x: x.steering_angle != 0

    def flip_center_image(self, steering_angle):
        return lambda x: np.fliplr(x), -steering_angle

    def histograms_equalization(self, steering_angle):
        return lambda x: cv2.equalizeHist(x), steering_angle

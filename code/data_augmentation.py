import numpy as np
import cv2

class Functions:

    @property
    def non_zero_angle_filter(self):
        return lambda x: x.steering_angle != 0

    def flip_h(self, steering_angle):
        return lambda x: np.fliplr(x), -steering_angle

    def flip_v(self, steering_angle):
        return lambda x: np.flipud(x), -steering_angle

    def histograms_equalization(self, steering_angle):
        return lambda x: cv2.equalizeHist(x), steering_angle

    @staticmethod
    def _tune_contrast(image, power):
        maxIntensity = 255.0
        newImage = maxIntensity * (image / maxIntensity) ** power
        return np.array(newImage, dtype=np.uint8)

    def increase_contrast(self, steering_angle):
        return lambda x: self._tune_contrast(x, 3), steering_angle

    def decrease_contrast(self, steering_angle):
        return lambda x: self._tune_contrast(x, 0.3), steering_angle
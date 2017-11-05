import numpy as np
import cv2
import random
from os import path

class Functions:

    def __init__(self):
        self.black_image = None

    @property
    def non_zero_angle_filter(self):
        return lambda x: x.steering_angle != 0

    def flip_h(self, image, steering_angle):
        return np.fliplr(image), -steering_angle

    def add_noise(self, image, steering_angle):
        def get_random_dot_value():
            return [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

        def replace_dot_values(im, rate):
            noisy_image = np.copy(im)
            row_indexes = list(range(0, noisy_image.shape[1]))
            noise_dots_in_row = int(noisy_image.shape[1] * rate)

            for row in noisy_image:
                row_noise_indexes = random.sample(row_indexes, noise_dots_in_row)
                for dot_index in row_noise_indexes:
                    row[dot_index] = get_random_dot_value()

            return noisy_image

        return replace_dot_values(image, 0.2), steering_angle

    @staticmethod
    def _tune_contrast(image, power):
        maxIntensity = 255.0
        newImage = maxIntensity * (image / maxIntensity) ** power
        return np.array(newImage, dtype=np.uint8)

    def increase_contrast(self, image, steering_angle):
        return self._tune_contrast(image, 3), steering_angle

    def decrease_contrast(self, image, steering_angle):
        return self._tune_contrast(image, 0.3), steering_angle

    def decrease_brightness(self, image, steering_angle):
        def get_dark_image(im):
            if self.black_image is None or self.black_image.shape != im.shape:
                self.black_image = np.zeros_like(im)
            return cv2.addWeighted(im, 0.7, self.black_image, 1.0, 0)

        return get_dark_image(image), steering_angle
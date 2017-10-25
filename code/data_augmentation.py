import numpy as np
import cv2
import random

class Functions:

    @property
    def non_zero_angle_filter(self):
        return lambda x: x.steering_angle != 0

    def flip_h(self, steering_angle):
        return lambda x: np.fliplr(x), -steering_angle

    def flip_v(self, steering_angle):
        def transform(image):
            flipped_image = np.flipud(image)
            up = np.zeros([70, image.shape[1], image.shape[2]], dtype=np.uint8)
            cropped = flipped_image[25:(160 - 70), :]
            down = np.zeros([25, image.shape[1], image.shape[2]], dtype=np.uint8)
            r = list()
            r.extend(up)
            r.extend(cropped)
            r.extend(down)
            return np.array(r)

        return lambda x: transform(x), steering_angle

    def add_noise(self, steering_angle):
        def get_random_dot_value():
            return [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

        def replace_dot_values(image, rate):
            noisy_image = np.copy(image)
            row_indexes = list(range(0, noisy_image.shape[1]))
            noise_dots_in_row = int(noisy_image.shape[1] * rate)

            for row in noisy_image:
                row_noise_indexes = random.sample(row_indexes, noise_dots_in_row)
                for dot_index in row_noise_indexes:
                    row[dot_index] = get_random_dot_value()

            return noisy_image

        return lambda x: replace_dot_values(x, 0.2), steering_angle

    @staticmethod
    def _tune_contrast(image, power):
        maxIntensity = 255.0
        newImage = maxIntensity * (image / maxIntensity) ** power
        return np.array(newImage, dtype=np.uint8)

    def increase_contrast(self, steering_angle):
        return lambda x: self._tune_contrast(x, 3), steering_angle

    def decrease_contrast(self, steering_angle):
        return lambda x: self._tune_contrast(x, 0.3), steering_angle
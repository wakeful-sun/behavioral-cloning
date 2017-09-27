from os import path
import csv
from keras.utils import Sequence
from math import ceil
import cv2
from sklearn.utils import shuffle
import numpy as np


class DrivingDataSequence(Sequence):

    def __init__(self, data_provider, batch_size):
        self.data_provider = data_provider
        self.batch_size = batch_size

    def __getitem__(self, index):
        start_index = index * self.batch_size
        stop_index = start_index + self.batch_size
        x, y = self.data_provider.get_range(start_index, stop_index)
        return np.array(x), np.array(y)

    def __len__(self):
        return ceil(self.data_provider.count / self.batch_size)

    def on_epoch_end(self):
        pass


class DataProvider:

    def __init__(self, data_folder_path="../captured_data"):

        data_path = path.join(path.dirname(__file__), data_folder_path)
        with open(path.join(data_path, "driving_log.csv")) as f:
            driving_log = csv.reader(f, delimiter=",")
            self.data = [DataFrame(data_path, line) for line in driving_log]

    @property
    def count(self):
        return len(self.data)

    def get_range(self, start_index, stop_index):
        batch = self.data[start_index:stop_index]
        batch_data = [], []
        for item in batch:
            batch_data[0].append(cv2.imread(item.image_center))
            batch_data[1].append(item.steering_angle)

        return batch_data

    def shuffle(self):
        shuffle(self.data)


class DataFrame:

    def __init__(self, data_folder_path, line):
        self.center = self._fit_image_path(data_folder_path, line[0])
        self.left = self._fit_image_path(data_folder_path, line[1])
        self.right = self._fit_image_path(data_folder_path, line[2])
        self.angle = float(line[3])

    @staticmethod
    def _fit_image_path(data_folder_path, original_path):
        image_in_folder = "/".join(original_path.split("/")[:-2])
        image_path = path.join(data_folder_path, image_in_folder)
        return path.normpath(image_path)

    @property
    def image_center(self):
        return self.center

    @property
    def image_center(self):
        return self.left

    @property
    def image_center(self):
        return self.right

    @property
    def steering_angle(self):
        return self.angle

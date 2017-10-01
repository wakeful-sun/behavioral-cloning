from os import path
import csv
from keras.utils.data_utils import Sequence
from math import ceil
from math import floor
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
        return floor((self.data_provider.count*2)/self.batch_size)

    @property
    def steps_per_epoch(self):
        return self.__len__()

    def on_epoch_end(self):
        pass


class DataContainer:

    def __init__(self, validation_split, data_folder_path="../captured_data"):
        if validation_split < 0 or validation_split >= 1:
            raise Exception("validation_set_len parameter should be in range [0..1]")

        data_path = path.join(path.dirname(__file__), data_folder_path)
        with open(path.join(data_path, "driving_log.csv")) as f:
            driving_log = csv.reader(f, delimiter=",")
            data = [DataFrame(data_path, line) for line in driving_log]

        shuffle(data)
        validation_set_len = floor(len(data) * validation_split)
        self.training_data = DataProvider(data[validation_set_len:])
        self.validation_data = DataProvider(data[:validation_set_len])

    @property
    def training(self):
        return self.training_data

    @property
    def validation(self):
        return self.validation_data


class DataProvider:

    def __init__(self, data):
        self.data = data
        self.augmentation_func = None

    @property
    def count(self):
        return len(self.data)

    def get_range(self, start_index, stop_index):
        batch = self.data[start_index:stop_index]

        if self.augmentation_func:
            return self.augmentation_func(batch)

        batch_data = [], []
        for item in batch:
            batch_data[0].append(cv2.imread(item.image_center))
            batch_data[1].append(item.steering_angle)

        return batch_data

    def shuffle(self, a=None, b=None):
        print("\nshuffle")
        shuffle(self.data)

    def register_data_augmentation(self, augmentation_func, increase_rate=1):
        if augmentation_func and callable(augmentation_func):
            self.augmentation_func = augmentation_func


class DataFrame:

    def __init__(self, data_folder_path, line):
        self.center = self._fit_image_path(data_folder_path, line[0])
        self.left = self._fit_image_path(data_folder_path, line[1])
        self.right = self._fit_image_path(data_folder_path, line[2])
        self.angle = float(line[3])

    @staticmethod
    def _fit_image_path(data_folder_path, original_path):
        image_in_folder = "/".join(original_path.split("\\")[-2:])
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

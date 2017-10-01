from os import path
import csv
import cv2
from sklearn.utils import shuffle
from math import floor
import matplotlib.image as mpimg


class DataContainer:

    def __init__(self, validation_split=0, data_folder_path="../captured_data"):
        if validation_split < 0 or validation_split >= 1:
            raise Exception("validation_set_len parameter should be in range [0..1]")

        data_path = path.join(path.dirname(__file__), data_folder_path)
        with open(path.join(data_path, "driving_log.csv")) as f:
            driving_log = csv.reader(f, delimiter=",")
            data_frame_factory = DataFrameFactory(data_path)
            data = [data_frame_factory.create(line) for line in driving_log]

        validation_set_len = floor(len(data) * validation_split)
        shuffle(data)

        self.training_data = DataProvider(data[validation_set_len:])
        self.validation_data = DataProvider(data[:validation_set_len])

    @property
    def training(self):
        return self.training_data

    @property
    def validation(self):
        return self.validation_data


class DataProvider:

    def __init__(self, data_frames):
        self.data_frames = data_frames

    @property
    def count(self):
        return len(self.data_frames)

    def get_range(self, start_index, stop_index):
        batch = self.data_frames[start_index:stop_index]

        batch_data = [], []
        for item in batch:
            frame_data = item.get_training_data()
            batch_data[0].append(frame_data[0])
            batch_data[1].append(frame_data[1])

        return batch_data

    def shuffle(self, a=None, b=None):
        print("\nshuffle")
        shuffle(self.data_frames)

    def apply_augmentation(self, augmentation_func, apply_rate=1):
        if not callable(augmentation_func):
            raise Exception("augmentation_func expected to be function, but was '{}'".format(type(augmentation_func)))

        def create_frame(original_frame):
            frame = original_frame.create_copy()
            frame.augmentation_functions.append(augmentation_func)
            return frame

        extra_frames_map = map(create_frame, self.data_frames[:self.count*apply_rate])
        self.data_frames = self.data_frames + list(extra_frames_map)
        shuffle(self.data_frames)


class DataFrameFactory:

    def __init__(self, data_folder_path):
        self.data_folder_path = data_folder_path

    def create(self, line):
        center = self._fit_image_path(self.data_folder_path, line[0])
        left = self._fit_image_path(self.data_folder_path, line[1])
        right = self._fit_image_path(self.data_folder_path, line[2])
        angle = float(line[3])
        return DataFrame(center, left, right, angle)

    @staticmethod
    def _fit_image_path(data_folder_path, original_path):
        image_in_folder = "/".join(original_path.split("\\")[-2:])
        image_path = path.join(data_folder_path, image_in_folder)
        return path.normpath(image_path)


class DataFrame:

    def __init__(self, center_image_path, left_image_path, right_image_path, steering_angle):
        self.im_path_center = center_image_path
        self.im_path_left = left_image_path
        self.im_path_right = right_image_path
        self.steering_angle = steering_angle
        self.registered_augmentation_functions = []

    def create_copy(self):
        data_frame_copy = DataFrame(self.im_path_center, self.im_path_left, self.im_path_right, self.steering_angle)
        for func in self.augmentation_functions:
            data_frame_copy.augmentation_functions.append(func)
        return data_frame_copy

    @property
    def augmentation_functions(self):
        return self.registered_augmentation_functions

    def get_training_data(self):
        bgr_image = cv2.imread(self.im_path_center)
        #hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        training_data = bgr_image, self.steering_angle

        for func in self.augmentation_functions:
            if callable(func):
                training_data = func(training_data)

        return training_data

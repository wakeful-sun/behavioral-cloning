from keras.utils.data_utils import Sequence
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
        return self.data_provider.count // self.batch_size

    @property
    def steps_per_epoch(self):
        return self.__len__()

    def on_epoch_end(self):
        pass
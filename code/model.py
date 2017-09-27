from keras.models import Sequential
from keras.layers import Flatten, Dense
from data_provider import DataProvider
from data_provider import DrivingDataSequence


BATCH_SIZE = 200

model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

model.compile("adam", "mse")

data_provider = DataProvider()
data_provider.shuffle()
sequence = DrivingDataSequence(data_provider, BATCH_SIZE)

model.fit_generator(sequence, BATCH_SIZE)
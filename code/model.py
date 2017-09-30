from keras.models import Sequential
from keras.layers import Flatten, Dense
from data_provider import DataContainer
from data_provider import DrivingDataSequence


BATCH_SIZE = 200

model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3), batch_size=BATCH_SIZE))
model.add(Dense(1))

model.compile("adam", "mse")

data_container = DataContainer(0.2)
data_container.training_data.shuffle()
training_sequence = DrivingDataSequence(data_container.training_data, BATCH_SIZE)
validation_sequence = DrivingDataSequence(data_container.validation_data, BATCH_SIZE)

model.fit_generator(training_sequence, training_sequence.steps_per_epoch, epochs=2,validation_steps=validation_sequence)
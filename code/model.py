from nn_model_factory import create_model
from data_provider import DataContainer
from data_provider import DrivingDataSequence
from keras.callbacks import LambdaCallback


BATCH_SIZE = 20

model = create_model()
model.compile("adam", "mse")

data_container = DataContainer(0.2)
data_container.training_data.shuffle()

t_seq = DrivingDataSequence(data_container.training_data, BATCH_SIZE)
v_seq = DrivingDataSequence(data_container.validation_data, BATCH_SIZE)

print("Number of training examples: ", t_seq.steps_per_epoch*BATCH_SIZE)
print("Number of validation examples: ", v_seq.steps_per_epoch*BATCH_SIZE)

model.fit_generator(t_seq, t_seq.steps_per_epoch,
                    epochs=5, callbacks=[LambdaCallback(on_epoch_end=data_container.training_data.shuffle)],
                    validation_data=v_seq, validation_steps=v_seq.steps_per_epoch)

model.save("../model.h5")

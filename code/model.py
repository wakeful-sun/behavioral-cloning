from nn_model_factory import create_model
from data_provider import DataContainer
from data_sequence import DrivingDataSequence
from keras.callbacks import LambdaCallback
from keras.callbacks import TensorBoard
from data_augmentation import flip_center_image
from datetime import datetime
import time


BATCH_SIZE = 20
EPOCHS = 5
output_folder = "../output/{:%d.%m.%y_%H-%M}_B{}_E{}/".format(datetime.now(), BATCH_SIZE, EPOCHS)

model = create_model()
model.compile("adam", "mse")

data_container = DataContainer(0.1)

data_container.training_data.shuffle()
data_container.training_data.apply_augmentation(flip_center_image)
data_container.validation_data.apply_augmentation(flip_center_image)

t_seq = DrivingDataSequence(data_container.training_data, BATCH_SIZE)
v_seq = DrivingDataSequence(data_container.validation_data, BATCH_SIZE)

print("Number of training examples: ", t_seq.steps_per_epoch*BATCH_SIZE)
print("Number of validation examples: ", v_seq.steps_per_epoch*BATCH_SIZE)

start_time = time.time()

callbacks = [
    LambdaCallback(on_epoch_end=data_container.training_data.shuffle),
    TensorBoard(log_dir=output_folder, batch_size=BATCH_SIZE)
]
model.fit_generator(t_seq, t_seq.steps_per_epoch,
                    epochs=EPOCHS, callbacks=callbacks,
                    validation_data=v_seq, validation_steps=v_seq.steps_per_epoch)

model.save(output_folder + "model.h5")

elapsed_time = time.time() - start_time
print("Training time: {:.2f} min".format(elapsed_time/60))

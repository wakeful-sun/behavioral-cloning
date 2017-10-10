from nn_model_factory import create_model
from data_provider import DataContainer
from data_sequence import DrivingDataSequence
from keras.callbacks import LambdaCallback
from keras.callbacks import TensorBoard
from data_augmentation import flip_center_image
from datetime import datetime
import time
from logger import Logger
import argparse


BATCH_SIZE = 20
EPOCHS = 3
DROPOUT = 0.5
model_description = ""
optimizer = "adam"
loss_fn = "mse"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neural network trainer')
    parser.add_argument("--batch", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument("--description", type=str, default=model_description, help="Model description")
    parser.add_argument("--dropout", type=float, default=DROPOUT, help="Model dropout rate in range [0.0:1.0]")
    args = parser.parse_args()

    BATCH_SIZE = args.batch
    EPOCHS = args.epochs
    DROPOUT = args.dropout
    model_description = args.description


output_folder = "../output/{:%d.%m.%y_%H-%M}_B{}_E{}_D{}_O-{}_L-{}/".format(
    datetime.now(), BATCH_SIZE, EPOCHS, DROPOUT, optimizer, loss_fn)

model = create_model(dropout=DROPOUT)
model.compile(optimizer, loss_fn, metrics=['accuracy'])

data_container = DataContainer(0.1)

data_container.training_data.shuffle()
data_container.training_data.apply_augmentation(flip_center_image)
data_container.validation_data.apply_augmentation(flip_center_image)

t_seq = DrivingDataSequence(data_container.training_data, BATCH_SIZE)
v_seq = DrivingDataSequence(data_container.validation_data, BATCH_SIZE)

num_training_msg = "Number of training examples: {}".format(t_seq.steps_per_epoch*BATCH_SIZE)
num_validation_msg = "Number of validation examples: {}".format(v_seq.steps_per_epoch*BATCH_SIZE)
print(num_training_msg)
print(num_validation_msg)

start_time = time.time()

callbacks = [
    LambdaCallback(on_epoch_end=data_container.training_data.shuffle),
    TensorBoard(log_dir=output_folder, batch_size=BATCH_SIZE)
]
history = model.fit_generator(t_seq, t_seq.steps_per_epoch,
                    epochs=EPOCHS, callbacks=callbacks,
                    validation_data=v_seq, validation_steps=v_seq.steps_per_epoch)

model.save(output_folder + "model.h5")

elapsed_time = time.time() - start_time
training_time_msg = "Training time: {:.2f} min".format(elapsed_time/60)
print(training_time_msg)

messages = [
    "Description: " + model_description,
    training_time_msg,
    num_training_msg,
    num_validation_msg,
    "Optimizer: {}, Loss function: {}, Dropout: {}".format(optimizer, loss_fn, DROPOUT)
]

data_logger = Logger(output_folder, model, history.history)
data_logger.save_summary(messages)
data_logger.save_model_json()
data_logger.save_history()

from nn_model_factory import create_model
from data_provider import DataContainer
from data_sequence import DrivingDataSequence
from data_augmentation import Functions
from logger import Logger
from settings import Settings
from keras.callbacks import LambdaCallback, TensorBoard
import time
import argparse
from os import path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neural network trainer')
    parser.add_argument("--batch", type=int, default=20, help="Batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--dropout", type=float, default=0.5, help="Model dropout rate in range [0.0:1.0]")
    parser.add_argument("--description", type=str, default="", help="Script run description")
    args = parser.parse_args()

    settings = Settings(args, optimizer="adam", loss_fn="mse")
    run_description = args.description
else:
    raise Exception("Program is not designed to be used without input parameters")

# --- data preparation ---

validation_set_split_rate = 0.1
data_container = DataContainer(validation_set_split_rate)

data_container.training_data.shuffle()
data_container.validation_data.shuffle()

f = Functions()
data_container.training_data.apply_augmentation(f.flip_h)
# data_container.training_data.apply_augmentation(f.decrease_brightness)
data_container.validation_data.apply_augmentation(f.flip_h)

data_container.print_summary(settings.batch_size)
data_container.training_data.save_top_images(settings.output_folder, "t")
data_container.validation_data.save_top_images(settings.output_folder, "v")

t_seq = DrivingDataSequence(data_container.training_data, settings.batch_size)
v_seq = DrivingDataSequence(data_container.validation_data, settings.batch_size)

# --- training ---

callbacks = [
    LambdaCallback(on_epoch_end=data_container.training_data.shuffle),
    LambdaCallback(on_epoch_end=data_container.validation_data.shuffle),
    TensorBoard(log_dir=settings.output_folder, batch_size=settings.batch_size)
]

model, model_description = create_model(settings.dropout)
model.compile(settings.optimizer, settings.loss_fn)

start_time = time.time()

h = model.fit_generator(t_seq, t_seq.steps_per_epoch,
                        epochs=settings.epochs, callbacks=callbacks,
                        validation_data=v_seq, validation_steps=v_seq.steps_per_epoch)

model.save(settings.output_folder + "model.h5")

# --- logging ---

elapsed_time = time.time() - start_time
print(" Training time : {:.2f} min".format(elapsed_time/60))
print(" Output path   : {}".format(path.abspath(settings.output_folder)))

data_logger = Logger(run_description, model_description, data_container, settings, model, h.history, elapsed_time)
data_logger.save_summary()
data_logger.log_results("../output/history.log")

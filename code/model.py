from nn_model_factory import create_model
from data_provider import DataContainer
from data_sequence import DrivingDataSequence
from data_augmentation import flip_center_image
from logger import Logger
from settings import Settings
from keras.callbacks import LambdaCallback, TensorBoard
import time
import argparse
from os import path
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neural network trainer')
    parser.add_argument("--batch", type=int, default=20, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--dropout", type=float, default=0.5, help="Model dropout rate in range [0.0:1.0]")
    parser.add_argument("--description", type=str, default="", help="Training description")
    args = parser.parse_args()

    s = Settings(args, "adam", "mse")
    info = { "description": args.description }
else:
    raise Exception("Program is not designed to be used without input parameters")


data_container = DataContainer(0.1)

data_container.training_data.shuffle()
data_container.training_data.apply_augmentation(flip_center_image)
data_container.validation_data.apply_augmentation(flip_center_image)

t_seq = DrivingDataSequence(data_container.training_data, s.batch_size)
v_seq = DrivingDataSequence(data_container.validation_data, s.batch_size)

data_summary = data_container.get_summary(s.batch_size)
print("*"*80)
print(" Training items         : {}".format(data_summary["training_items_total"]))
print(" Validation items       : {}".format(data_summary["validation_items_total"]))
print(" Unique steering angles : {}".format(data_summary["unique_steering_angles_count"]))
print(" Output path            : {}".format(path.abspath(s.output_folder)))
print("*"*80)

callbacks = [
    LambdaCallback(on_epoch_end=data_container.training_data.shuffle),
    TensorBoard(log_dir=s.output_folder, batch_size=s.batch_size)
]

model, model_description = create_model(dropout=s.dropout)
model.compile(s.optimizer, s.loss_fn, metrics=['accuracy'])

start_time = time.time()

history = model.fit_generator(t_seq, t_seq.steps_per_epoch,
                              epochs=s.epochs, callbacks=callbacks,
                              validation_data=v_seq, validation_steps=v_seq.steps_per_epoch)

model.save(s.output_folder + "model.h5")

elapsed_time = time.time() - start_time
print("Training time: {:.2f} min".format(elapsed_time/60))


def get_neuron_group(neuron):
    return {
        "name": neuron.name,
        "type": type(neuron).__name__,
        "shape": neuron.shape,
        "neurons_count": np.prod(np.array(neuron.shape))
    }


def get_layer_info(layer):
    return {
        "name": layer.name,
        "type": type(layer).__name__,
        "input_shape": layer.input_shape,
        "output_shape": layer.output_shape,
        "is_trainable": layer.trainable,
        "trainable_neurons": [get_neuron_group(n) for n in layer.weights]
    }


info["model_description"] = model_description
info["settings"] = s.to_dict()
info["data_summary"] = data_summary
info["results"] = {
    "history": history.history,
    "training_time": elapsed_time,
    "metrics": {k: v[-1] for k, v in history.history.items()}
}
info["model_summary"] = {
    "layers": [get_layer_info(l) for l in model.layers],
    "model": model.to_json()
}

data_logger = Logger(info, model)
data_logger.save_summary()
data_logger.save_training_history("../output/history.log")

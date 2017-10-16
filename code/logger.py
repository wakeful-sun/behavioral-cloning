import numpy as np
import json
from os import path
import matplotlib.pyplot as plt

class Logger:

    def __init__(self, run_description, model_description, data_container, settings, model, history, training_time):
        self.run_description = run_description
        self.model_description = model_description
        self.data_summary_dict = data_container.get_summary_dict(settings.batch_size)
        self.settings = settings
        self.model = model
        self.history = history
        self.training_time= training_time

    @staticmethod
    def _save_steering_angles_statistics_as_image(path, data):
        angles, angles_amount = data["angle"], data["size"]

        plt.xlabel("steering angle")
        plt.ylabel("amount of samples")
        plt.plot(angles, angles_amount, "go")
        plt.vlines(angles, 0, angles_amount, "red", "dotted")
        plt.axhline(0, color='black', linewidth=1.0)
        plt.savefig(path)

    def save_summary(self):
        t_accuracy =  self.history["acc"][-1]*100
        v_accuracy =  self.history["val_acc"][-1]*100
        steering_angles_statistics = self.data_summary_dict["steering_angles_statistics"]

        messages = [
            " run description        : {}".format(self.run_description),
            " model description      : {}".format(self.model_description),
            " training time          : {:.2f}".format(self.training_time/60),
            "-"*65,
            " training items         : {}".format(self.data_summary_dict["training_items_total"]),
            " validation items       : {}".format(self.data_summary_dict["validation_items_total"]),
            " unique steering angles : {}".format(len(steering_angles_statistics["angle"])),
            "-"*65,
            " epochs                 : {}".format(self.settings.epochs),
            " batch size             : {}".format(self.settings.batch_size),
            " dropout                : {}".format(self.settings.dropout),
            " optimizer              : {}".format(self.settings.optimizer),
            " loss fn                : {}".format(self.settings.loss_fn),
            " output folder          : {}".format(self.settings.output_folder),
            "-"*65,
            " loss                   : {:.5f}".format(self.history["loss"][-1]),
            " validation loss        : {:.5f}".format(self.history["val_loss"][-1]),
            " accuracy               : {:.5f}%".format(t_accuracy),
            " validation accuracy    : {:.5f}%".format(v_accuracy),
            "-"*65,
            "\n"
        ]

        summary_file_name = "_summary_{:.5f}.txt".format(t_accuracy)
        summary_fig_img_path = path.join(self.settings.output_folder, "_steering_angles.png")

        self._save_steering_angles_statistics_as_image(summary_fig_img_path, steering_angles_statistics)

        with open(path.join(self.settings.output_folder, summary_file_name), "w") as f:
            f.write("\n".join(messages))
            self.model.summary(print_fn=lambda x: f.write(x + "\n"))

    def log_results(self, log_file_path):

        def shape_to_serializable(shape):
            serializable_shape = []
            for i in shape:
                if i is None:
                    serializable_shape.append(None)
                else:
                    serializable_shape.append(int(i))

            return serializable_shape

        def get_tensor_info(tensor):
            tensor_shape = tensor.get_shape()
            return {
                "name": tensor.name,
                "type": type(tensor).__name__,
                "shape": shape_to_serializable(tensor_shape),
                "parameters_count": int(np.prod(np.array(tensor_shape)))
            }

        def get_layer_info(layer):
            return {
                "name": layer.name,
                "type": type(layer).__name__,
                "input_shape": shape_to_serializable(layer.input_shape),
                "output_shape": shape_to_serializable(layer.output_shape),
                "is_trainable": layer.trainable,
                "trainable_tensors": [get_tensor_info(t) for t in layer.weights]
            }

        info = {
            "run_description": self.run_description,
            "model_description": self.model_description,
            "settings": self.settings.to_dict(),
            "data_summary": self.data_summary_dict,
            "results": {
                "history": self.history,
                "training_time": self.training_time,
                "metrics": {k: v[-1] for k, v in self.history.items()}
            },
            "model_summary": {
                "layers": [get_layer_info(l) for l in self.model.layers],
                "model": json.loads(self.model.to_json())
            }
        }

        log_entry = json.dumps(info, separators=(',', ':'), sort_keys=True)

        with open(log_file_path, "a") as log_file:
            log_file.write(log_entry)
            log_file.write("\n")

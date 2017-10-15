import pandas as pd
import json
from os import path

class Logger:

    def __init__(self, run_description, model_description, data_summary_dict, settings, model, history, training_time):
        self.run_description = run_description
        self.model_description = model_description
        self.data_summary_dict = data_summary_dict
        self.settings = settings
        self.model = model
        self.history = history
        self.training_time= training_time

    def save_summary(self):
        output_folder_path = path.abspath(self.settings.output_folder)
        t_accuracy =  self.history["acc"]*100

        statistic_messages = [
            " run description        : {}".format(self.run_description),
            " model description      : {}".format(self.model_description),
            " training time          : {:.2f}".format(self.training_time/60),
            "-"*65,
            " training items         : {}".format(self.data_summary_dict["training_items_total"]),
            " validation items       : {}".format(self.data_summary_dict["validation_items_total"]),
            " unique steering angles : {}".format(self.data_summary_dict["unique_steering_angles_count"]),
            "-"*65,
            " epochs                 : {}".format(self.settings.epochs),
            " batch size             : {}".format(self.settings.batch_size),
            " dropout                : {}".format(self.settings.dropout),
            " optimizer              : {}".format(self.settings.optimizer),
            " loss fn                : {}".format(self.settings.loss_fn),
            " output folder          : {}".format(output_folder_path),
            "-"*65,
            " loss                   : {:.5f}".format(self.history["loss"]),
            " validation loss        : {:.5f}".format(self.history["val_loss"]),
            " accuracy               : {:.5f}%".format(t_accuracy),
            " validation accuracy    : {:.5f}%".format(self.history["val_acc"]*100),
            "-"*65
        ]

        summary_file_path = "{}_summary_{:.5f}.txt".format(output_folder_path, t_accuracy)

        with open(summary_file_path, "w") as f:
            f.write("\n".join(statistic_messages))
            f.write("\n\n")
            self.model.summary(print_fn=lambda x: f.write(x + "\n"))

    def log_results(self, log_file_path):
        pass

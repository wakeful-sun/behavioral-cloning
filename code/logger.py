import pandas as pd

class Logger:

    def __init__(self, output_folder_path, model, history):
        self.output_folder_path = output_folder_path
        self.model = model
        self.history = history

    def save_summary(self, messages):
        t_loss = self.history["loss"][-1]
        v_loss = self.history["val_loss"][-1]
        t_accuracy =  self.history["acc"][-1]*100
        v_accuracy =  self.history["val_acc"][-1]*100

        statistic_messages = [
            "-"*65,
            "loss:                {:.5f}".format(t_loss),
            "validation loss:     {:.5f}".format(v_loss),
            "accuracy:            {:.5f}%".format(t_accuracy),
            "validation accuracy: {:.5f}%".format(v_accuracy),
            "-"*65
        ]

        summary_file_path = "{}_summary_{:.5f}.txt".format(self.output_folder_path, t_accuracy)
        log_messages = messages + statistic_messages

        with open(summary_file_path, "w") as f:
            f.write("\n".join(log_messages))
            f.write("\n\n")
            self.model.summary(print_fn=lambda x: f.write(x + "\n"))


    def save_model_json(self):
        with open(self.output_folder_path + "model.json", "w") as f:
            f.write(self.model.to_json())


    def save_history(self):
        df = pd.DataFrame(self.history)
        df.to_csv(self.output_folder_path + "history.csv")

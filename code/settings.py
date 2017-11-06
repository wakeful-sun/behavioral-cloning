from datetime import datetime


class Settings:

    def __init__(self, args, optimizer="adam", loss_fn="mse"):

        now_date = datetime.now()
        self.l_batch_size = args.batch
        self.l_epochs = args.epochs
        self.l_dropout = args.dropout
        self.l_optimizer = optimizer
        self.l_loss_fn = loss_fn
        self.l_output_folder = "../output/{:%d.%m.%y_%H-%M}_B{}_E{}_D{}_O-{}_L-{}/".format(
            now_date, args.batch, args.epochs, args.dropout, optimizer, loss_fn)

    @property
    def batch_size(self):
        return self.l_batch_size
    @property
    def epochs(self):
        return self.l_epochs
    @property
    def dropout(self):
        return self.l_dropout
    @property
    def optimizer(self):
        return self.l_optimizer
    @property
    def loss_fn(self):
        return self.l_loss_fn
    @property
    def output_folder(self):
        return self.l_output_folder

    def to_dict(self):
        return {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "dropout": self.dropout,
            "optimizer": self.optimizer,
            "loss_fn": self.loss_fn,
            "output_folder": self.output_folder,
        }
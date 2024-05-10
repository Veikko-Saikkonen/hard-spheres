import mlflow
import json
from torch.utils.tensorboard import SummaryWriter


class TrainingLogger:
    def __init__(self, log_dir="./runs", comment=None, mode="both"):
        self.mode = mode
        if self.mode == "tf" or self.mode == "both":
            self.tb_writer = SummaryWriter(log_dir=log_dir, comment=comment)
        if self.mode == "mlflow" or self.mode == "both":
            if comment:
                mlflow.set_experiment(comment)
            mlflow.start_run()

    def add_scalars(self, tag, value, step):
        if self.mode == "tf" or self.mode == "both":
            self.tb_writer.add_scalars(tag, value, step)
        if self.mode == "mlflow" or self.mode == "both":
            if type(value) == dict:
                log_value = {tag + key: value for key, value in value.items()}
            else:
                log_value = value
            mlflow.log_metrics(log_value, step=step)

    def add_image(self, tag, image, step):
        if self.mode == "tf" or self.mode == "both":
            self.tb_writer.add_image(tag, image, step)
        if self.mode == "mlflow" or self.mode == "both":
            mlflow.log_image(image, tag)

    def add_figure(self, tag, image, step):
        if self.mode == "mlflow":
            tag += str(step)

        if tag.split(".")[-1] not in ["png", "jpg", "svg"]:
            tag += ".png"

        if self.mode == "tf" or self.mode == "both":
            self.tb_writer.add_figure(tag, image, step)
        if self.mode == "mlflow" or self.mode == "both":
            mlflow.log_figure(image, tag)

    def add_text(self, tag, text, step=None):
        if self.mode == "tf" or self.mode == "both":
            if type(text) == dict:
                text = json.dumps(text, indent=2)
            self.tb_writer.add_text(tag, text, step)
        if self.mode == "mlflow" or self.mode == "both":
            if type(text) == dict:
                mlflow.log_dict(text, tag)
            else:
                mlflow.log_text(text, tag)

    def end_run(self):
        if self.mode == "mlflow" or self.mode == "both":
            mlflow.end_run()
        if self.mode == "tf" or self.mode == "both":
            self.tb_writer.flush()

    def flush(self):
        if self.mode == "mlflow" or self.mode == "both":
            mlflow.end_run()
        if self.mode == "tf" or self.mode == "both":
            self.tb_writer.flush()

    def close(self):
        if self.mode == "mlflow" or self.mode == "both":
            mlflow.end_run()
        if self.mode == "tf" or self.mode == "both":
            self.tb_writer.close()

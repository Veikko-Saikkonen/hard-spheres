import sklearn.metrics as metrics
import numpy as np
from pandas import Series
from typing import Union
import os
import sys
from datetime import datetime
from torch import exp
import pandas as pd
from torch import nn
import torch
import json
from pathlib import Path
from copy import deepcopy
import logging
import adabound  # https://github.com/Luolc/AdaBound
from datetime import datetime


from models.losses import RMSLELoss
import models


def sigmoid(x, x0=0.0, k=1.0):
    return 1.0 / (1 + exp(-k * (x - x0)))


def save_predictions(predictions: dict, name, save_path, dt_index=None):
    """Saves model predictions to csv

    Args:
        predictions (dict): Expects {"train": {"y": ..., "pred": ...}, "test: ..., "val":...}
        name (str): name of the file
        save_path (_type_): folder of the model
        dt_index (_type_, optional): If datetime is known it is set to be the index of the df. Defaults to None.
    Returns:
        pd.DataFrame: The formatted predictions
    """

    save_path = save_path / f"{name}-predictions.csv"
    if dt_index is None:
        dt_index = pd.RangeIndex(start=0, stop=predictions.shape[0])
    df = pd.DataFrame(data=predictions, index=pd.to_datetime(dt_index))
    df.to_csv(save_path)
    return df


def save_model(model, config, name, model_save_dir="saved_models"):
    filename = datetime.now().strftime("%d-%m-%Y_%H-%M") + "-" + name
    dir_path = Path(model_save_dir) / filename
    os.makedirs(dir_path, exist_ok=True)  # Creates model_save_dir/model_id
    logging.info(f"Saving model {filename} to saved_models")
    torch.save(model.state_dict(), dir_path / "model.pt")
    with open(dir_path / ("config.json"), "w") as fp:
        s = json.dumps(config, sort_keys=False, indent=4) + "\n"
        fp.write(s)
    return dir_path, filename


def regression_results(
    y_true: Union[Series, np.ndarray], y_pred: Union[Series, np.ndarray], verbose=False
):
    if y_true.shape[0] != y_pred.shape[0]:
        raise Warning(
            "Real and predicted y are shaped differently. Shapes : "
            + str(y_true.shape)
            + ", "
            + str(y_pred.shape)
        )

    if verbose:
        print("1/6: Computing explained variance...")
    explained_variance = metrics.explained_variance_score(y_true, y_pred)

    if verbose:
        print("2/6: Computing MAE...")
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)

    if verbose:
        print("3/6: Computing MSE...")
    mse = metrics.mean_squared_error(y_true, y_pred)

    if verbose:
        print("4/6: Computing median absolute error...")
    median_absolute_error = metrics.median_absolute_error(y_true, y_pred)

    if verbose:
        print("5/6: Computing R2 score...")
    r2 = metrics.r2_score(y_true, y_pred)

    if verbose:
        print("6/6: Computing maximum absolute error...")
    max_error = metrics.max_error(y_true, y_pred)

    rmsle = np.sqrt(metrics.mean_squared_log_error(y_true, y_pred))

    return {
        "dataset_size": y_true.shape[0],
        "explained_variance": np.around(explained_variance, 4),
        "r2": np.around(r2, 4),
        "MAE": np.around(mean_absolute_error, 4),
        "median_absolute_error": np.around(median_absolute_error, 4),
        "MSE": np.around(mse, 4),
        "RMSE": np.around(np.sqrt(mse), 4),
        "RMSLE": np.around(rmsle, 4),
        "maximum_absolute_error": np.around(max_error, 4),
    }


def print_progress(count, total, status=""):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = "=" * filled_len + "-" * (bar_len - filled_len)

    sys.stdout.write(
        "\r[%s] %s%s %s/%s... \t>> %s << %s"
        % (
            bar,
            percents,
            "%",
            count,
            total,
            datetime.now().strftime("%H:%M:%S"),
            status,
        )
    )
    sys.stdout.flush()


# The MIT License (MIT)
# Copyright (c) 2016 Vladimir Ignatev
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software
# is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
# OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


def makedir_if_not_exists(path):
    if not os.path.isdir(path):
        print("Creating directory {}".format(path))
        os.mkdir(path)
    return


def build_model_from_config(model_config: dict, feature_order: list) -> torch.nn.Module:
    try:
        model_class = getattr(models, model_config["name"])
        model = model_class(features=feature_order, **model_config)
    except AttributeError as e:
        logging.exception(e)
        raise ValueError(f"Unknown model name {model_config['name']} in config")
    return model


def build_optimizer_fn_from_config(optimizer_config):
    optimizer_config = deepcopy(optimizer_config)
    name = optimizer_config.pop(
        "name"
    )  # Remove name from config as the base constructors will not allow it
    try:
        optimizer_class = getattr(adabound, name)
    except AttributeError:
        optimizer_class = getattr(
            torch.optim,
            name,
            )

    def optimizer_init(*args, **kwargs):
        return optimizer_class(*args, **kwargs, **optimizer_config)

    return optimizer_init


def build_loss_from_config(config):
    if config["loss"] == "MSE" or config["loss"] == "RMSE":
        criterion = nn.MSELoss()
    elif config["loss"] == "RMSLE":
        criterion = RMSLELoss(delta=1.0)
    else:
        raise ValueError(f"Loss {config['loss']} not recognized.")

    return criterion

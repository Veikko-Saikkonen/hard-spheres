import pandas as pd

from glob import glob
from pathlib import Path

from copy import deepcopy

import hypergrad
import adabound
import torch

from src.adopt import adopt

import mlflow


# Log nested items with underscores
def log_nested_dict(d, prefix=""):
    for key, value in d.items():
        if isinstance(value, dict):
            log_nested_dict(value, prefix=f"{prefix}{key}_")
        else:
            mlflow.log_param(f"{prefix}{key}", value)


def build_run_name(run_name=None):
    # Gets two random nouns and an UUID for a human readable string name for a run.
    import uuid
    import random

    # Assuming a predefined list of nouns for simplicity. In a real scenario, this could be replaced with a more dynamic method or larger dataset.
    nouns = [
        "tree",
        "car",
        "mountain",
        "river",
        "cloud",
        "star",
        "forest",
        "beach",
        "sun",
        "moon",
    ]

    # Select two random nouns from the list
    noun1 = random.choice(nouns)
    noun2 = random.choice(nouns)

    # Generate a UUID
    unique_id = uuid.uuid4()

    if run_name is None:
        # Concatenate the nouns and UUID to form a unique, human-readable name
        model_name = f"{noun1}-{noun2}-{unique_id}"
    else:
        model_name = f"{run_name}-{unique_id}"

    return model_name


def build_optimizer_fn_from_config(optimizer_config):
    optimizer_config = deepcopy(optimizer_config)
    name = optimizer_config.pop(
        "name"
    )  # Remove name from config as the base constructors will not allow it
    try:
        optimizer_class = getattr(hypergrad, name)
    except AttributeError:
        try:
            optimizer_class = getattr(adabound, name)
        except AttributeError:
            try:
                optimizer_class = getattr(
                    torch.optim,
                    name,
                )
            except AttributeError:
                try:
                    optimizer_class = getattr(
                        adopt,
                        name,
                    )
                except:
                    raise AttributeError(f"Optimizer {name} not found")

    def optimizer_init(*args, **kwargs):
        return optimizer_class(*args, **kwargs, **optimizer_config)

    return optimizer_init


def read_raw_sample(file, skiprows=2, sep=r"\s+"):
    # Read txt files using pandas
    metadata = pd.read_csv(file, nrows=1, header=None, sep=sep, names=["N", "L", "A"])

    dataframe = pd.read_csv(
        file, skiprows=skiprows, header=None, sep=sep, names=["class", "x", "y", "r"]
    )
    # dataframe.columns = ["class", "x", "y", "r"]
    dataframe = dataframe.apply(pd.to_numeric, errors="raise")
    experiment = "".join(file.split("/")[-2])
    sample = "".join(file.split("/")[-1])
    dataframe["experiment"] = experiment
    dataframe["sample"] = sample

    metadata["experiment"] = experiment
    metadata["sample"] = sample

    dataframe.set_index(["experiment", "sample"], inplace=True)
    metadata.set_index(["experiment", "sample"], inplace=True)

    return dataframe, metadata


def load_raw_data(path="data", phi=[0.72], subpath=""):

    if len(phi) != len(set(phi)):
        raise ValueError("Phi values must be unique")

    path = Path(path)
    search_paths = []
    for p in phi:
        search_paths.append(path / f"phi-{p:.2f}" / subpath)
    files = []
    for sp in search_paths:
        new_files = glob(str(sp / "sample-*"))

        files += new_files

        if len(new_files) == 0:
            print(f"No files found for {sp}")

    print("Number of Files found : ", len(files))

    # Reading data
    data = []
    metadata = []
    for f in files:
        dataframe, meta = read_raw_sample(f, skiprows=2, sep=r"\s+")
        data.append(dataframe)
        metadata.append(meta)

    dataframe = pd.concat(data)
    metadata = pd.concat(metadata)

    return files, dataframe, metadata


def mock_function(a: int, b: str, **kwargs) -> int:
    """_summary_

    Args:
        a (int): _description_
        b (str): _description_

    Returns:
        int: _description_
    """
    return a + int(b)

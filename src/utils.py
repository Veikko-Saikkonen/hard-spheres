import pandas as pd

from glob import glob
from pathlib import Path


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


def load_raw_data(path="data", phi=[0.72]):

    if len(phi) != len(set(phi)):
        raise ValueError("Phi values must be unique")

    path = Path(path)
    search_paths = []
    for p in phi:
        search_paths.append(path / f"phi-{p:.2f}")
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

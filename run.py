# File that runs the experiments for the paper

import os
import sys
import time
from glob import glob
import argparse
import re
from pathlib import Path
import pandas as pd
import yaml
from torchinfo import summary


from src.HSDataset import HSDataset # Import the dataset class
from src.plotting import plot_pointcloud, plot_sample_figures
from src.models.StaticScaler import StaticMinMaxScaler
from src.utils import load_raw_data
from src.models.HardSphereGAN import GAN

def get_data_path(dataset_name):
    if dataset_name == "Sq":
        path = Path("data/raw/crystal/Sq")
    elif dataset_name == "Hex":
        path = Path("data/raw/crystal/Hex")
    elif dataset_name == "Fullscale":
        path = Path("data/raw/samples")
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}, please choose from 'Sq', 'Hex', 'Fullscale'")
    return path 

def get_box_parameters(dataset_name):
    if dataset_name == "Sq":
        # Square lattice
        N = 1600
        X_box = 38.225722823651111
        Y_box = 38.225722823651111 
    
    elif dataset_name == "Hex":
        N = 1600 
        X_box = 41.076212368516387
        Y_box = 35.573043402379753   
    elif dataset_name == "Fullscale":
        N = 2000
        X_box = 44.0
        Y_box = 44.0
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}, please choose from 'Sq', 'Hex', 'Fullscale'")

    return N, X_box, Y_box


def order_data(df):
    """Order the data in the dataset by the type of atom (indicated by r)"""
    return df.sort_values(by=["experiment", "sample", "r"], ignore_index=False)

def load_data(experiment_file):

    path = get_data_path(experiment_file["dataset"]["name"])

    files, df_raw, metadata = load_raw_data(
        path=path, 
        phi=experiment_file["dataset"]["phis"], 
        subpath=experiment_file["dataset"]["subpath"]
        )

    N, X_box, Y_box = get_box_parameters(experiment_file['dataset']["name"])

    _scaler = StaticMinMaxScaler(
        columns = ["x", "y", "r"],
        maximum = [X_box, Y_box, 2*X_box], # NOTE: Tuned for physical feasibility
        minimum = [-X_box, -Y_box, 0] # NOTE: Tuned for physical feasibility
    )

    df_scaled = pd.DataFrame(_scaler.transform(df_raw), columns=df_raw.columns)
    df_scaled.set_index(df_raw.index, inplace=True)
    df_scaled = df_scaled.drop(columns=["class"])
    df_scaled = df_scaled.sort_values(by=["experiment", "sample"])
    df_ordered = order_data(df_scaled)
    print(df_ordered.head(10))
    dataset = HSDataset(df_ordered, **experiment_file["dataset"])

    return dataset
 


def run_experiment(experiment_file, run_name=None, experiment=""):

    if isinstance(experiment_file, str):
        with open(experiment_file, "r") as f:
            experiment_file = yaml.full_load_all(f)
            for key in experiment_file:
                experiment_file = key
    else:
        assert isinstance(experiment_file, dict), f"'experiment_file' must be a dict or str, got: {type(experiment_file)}"

    # Load data into pandas
    dataset = load_data(experiment_file)

    print(f"data loaded, samples: {len(dataset)}")

    print("Creating models...")

    gan = GAN(
        dataset, 
        dataset,# No separate test set
        **experiment_file
    )

    sample_x = dataset[0:32][0].cpu()#.transpose(-1,-2)
    sample_y = dataset[0:32][1].cpu()

    sample_x_mps = sample_x.to("mps")
    sample_y_mps = sample_y.to("mps")

    print("Models created, summary below.\n")

    summary(gan.generator, input_data=sample_x_mps, depth=2)
    summary(gan.discriminator, input_data=sample_y_mps, depth=2)
    _out = gan.generate(sample_x)[0]

    plot_pointcloud(_out, plot_radius=False)

    gan.train_n_epochs(
        epochs=experiment_file["training"]["epochs"],
        batch_size=experiment_file["training"]["batch_size"],
        experiment_name=experiment,
        run_name=run_name,
        requirements_file = Path(experiment_file["requirements_file"]),
        save_model=True
    )



def main():
    """Take in the arguments and run the experiments"""

    parser = argparse.ArgumentParser(description="Run selected experiments")
    parser.add_argument('--experiment', type=str, help='Regex to include experiment')
    parser.add_argument('--include', type=str, default='all', help='Regex to include experiment files')
    parser.add_argument('--repeats', type=int, default=1, help='Number of times to repeat each experiment')
    parser.add_argument('--exclude', type=str, default=None, help='Regex to exclude experiment files')

    args = parser.parse_args()

    experiment = args.experiment

    experiment_files_path = os.path.join("experiments", experiment)
    regex_postfix = "*.yaml"

    experiment_files = glob(os.path.join(experiment_files_path, regex_postfix))

    if args.include != 'all':
        experiment_files = [f for f in experiment_files if re.search(args.include, f)]

    if args.exclude:
        experiment_files = [f for f in experiment_files if not re.search(args.exclude, f)]

    print(f"Found {len(experiment_files)} experiment file(s)")

    if len(experiment_files) == 0:
        print(f"No experiment files found in {experiment_files_path}, exiting")
        exit()
    
    print(f"Running experiment: {experiment}")
    for run_file in experiment_files:
        for _ in range(args.repeats):

            run_name = run_file.split("/")[-1].split(".")[0]
            
            with open(run_file, "r") as f:
            # Create a dict from the yaml string
                run_dict = yaml.full_load_all(f)
                run_dict = run_dict
                for key in run_dict:
                    run_dict = key
                    break # NOTE: This is a hack, need to find proper way to parse yaml
            # Add code to run the experiment here
            run_experiment(run_dict, run_name=run_name, experiment=experiment)




if __name__ == '__main__':
    main()
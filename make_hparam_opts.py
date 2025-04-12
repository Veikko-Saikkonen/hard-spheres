import itertools
import os
import yaml
import argparse
import copy

# A function to create a list of hyperparameter options for a given experiment.

def make_hparam_opts(experiment_file, hparam_opts, output_path):
    """
    Recursively creates different hyperparameter options for a given experiment,
    handling nested dictionaries. For every combination of hyperparameter options,
    the function creates a new experiment file (in YAML format) and saves it to the
    output path.

    Args:
        experiment_file (dict): The original experiment definition (e.g. loaded from YAML).
        hparam_opts (dict): The hyperparameter options for each argument.
            This can include nested dictionaries (see example).
        output_path (str): The directory path where the output YAML files will be saved.

    Example experiment file (YAML as dict):
        requirements_file: ./top-level-requirements.txt
        dataset:
          name: Sq
          phis:
            - 0.84
          descriptor_list:
            - phi
        subpath: disorder-0.2
        downsample: 0.005
        keep_r: false
        synthetic_samples:
          rotational: 0
          shuffling: 0
          spatial_offset_repeats: 0
          spatial_offset_static: 0.
        discriminator:
          channels_coefficient: 1
          class: CCCGDiscriminator
          in_samples: 8
          input_channels: 2
          kernel_size:
            - 1
            - 1
        generator:
          channels_coefficient: 1
          class: CCCGenerator
          clip_output: false
          fix_r: 0.0049
          kernel_size:
            - 1
            - 1
          latent_dim: 128
          out_dimensions: 2
          out_samples: 8
          rand_features: 64
          stride: 1
        metrics:
          packing_fraction: false
          packing_fraction_box_size: 1
          packing_fraction_fix_r: 0.0049
        training:
          batch_size: 32
          d_loss:
            mu: 0.5
            name: CryinGANDiscriminatorLoss
          device: mps
          early_stopping_headstart: 0
          early_stopping_patience: -1
          early_stopping_tolerance: 0.001
          epochs: 200
          g_loss:
            name: HSGeneratorLoss
            coefficients:
              distance_loss: 0
              gan_loss: 1
              grid_density_loss: 0
              physical_feasibility_loss: 0
              radius_loss: 0
              grid_order_loss: 0
              grid_order_k: 4
          generator_headstart: 0
          log_image_frequency: 1
          optimizer_d:
            betas:
              - 0.5
              - 0.999
            lr: 0.0001
            name: Adam
            weight_decay: 0
          optimizer_g:
            betas:
              - 0.5
              - 0.999
            lr: 0.0001
            name: Adam
            weight_decay: 0
          training_ratio_dg: 3

    Example hparam_opts file (YAML as dict):
        training:
          batch_size: [8, 16, 32, 64]
          optimizer_d:
            lr: [0.00001, 0.0001, 0.001, 0.01]
          optimizer_g:
            lr: [0.00001, 0.0001, 0.001, 0.01]
    """
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Flatten the hyperparameter options so that each key path (a list of keys) maps to a list of options.
    option_items = _flatten_hparam_opts(hparam_opts)
    # For each option, we will have a list of values.
    option_values = [vals for key_path, vals in option_items]

    # Compute the Cartesian product of all hyperparameter option lists.
    # If there are no hyperparameter options, product returns one tuple: ()
    all_combinations = list(itertools.product(*option_values))

    print(f"Generating {len(all_combinations)} experiment option file(s)...")
    for idx, combination in enumerate(all_combinations):
        # Make a deep copy of the original experiment configuration.
        experiment_variant = copy.deepcopy(experiment_file)
        # For each hyperparameter (identified by its nested key path), set the chosen value.
        for (key_path, _), value in zip(option_items, combination):
            _set_nested_value(experiment_variant, key_path, value)
        # Construct a filename that optionally encodes some hyperparameter info.
        filename = os.path.join(output_path, f"experiment_{idx + 1}.yaml")
        with open(filename, 'w') as f:
            yaml.dump(experiment_variant, f)
        print(f"Saved: {filename}")


def _flatten_hparam_opts(opts, prefix=None):
    """
    Recursively flatten the hyperparameter options dictionary.
    Each leaf entry (i.e. list of options) is returned as a tuple:
        (key_path, options_list)
    where key_path is a list of keys describing the nested position.

    Args:
        opts (dict): The hyperparameter options dictionary.
        prefix (list): The accumulated keys (used in recursion).

    Returns:
        list of tuples: [ (key_path, options_list), ... ]
    """
    if prefix is None:
        prefix = []
    items = []
    for key, val in opts.items():
        if isinstance(val, dict):
            # Recurse into nested dictionaries.
            items.extend(_flatten_hparam_opts(val, prefix + [key]))
        elif isinstance(val, list):
            items.append((prefix + [key], val))
        else:
            # If not a list or dict, treat it as a single-option list.
            items.append((prefix + [key], [val]))
    return items


def _set_nested_value(d, key_path, value):
    """
    Sets a value in a nested dictionary given a list of keys.
    
    Args:
        d (dict): The dictionary in which to set the value.
        key_path (list): List of keys representing the nested path.
        value: The value to set.
    """
    for key in key_path[:-1]:
        if key not in d or not isinstance(d[key], dict):
            d[key] = {}
        d = d[key]
    d[key_path[-1]] = value

# Example usage:
if __name__ == '__main__':
    # Suppose these dictionaries are loaded from YAML files.
    experiment_config = {
        "training": {
            "batch_size": 32,
            "optimizer_d": {
                "lr": 0.0001,
                "name": "Adam",
                "weight_decay": 0
            },
            "optimizer_g": {
                "lr": 0.0001,
                "name": "Adam",
                "weight_decay": 0
            },
            "epochs": 200
        },
        "dataset": {
            "name": "Sq",
            "phis": [0.84]
        }
    }

    hparam_options = {
        "training": {
            "batch_size": [8, 16, 32, 64],
            "optimizer_d": {
                "lr": [0.00001, 0.0001, 0.001, 0.01]
            },
            "optimizer_g": {
                "lr": [0.00001, 0.0001, 0.001, 0.01]
            }
        }
    }

    # Specify an output directory.
    output_dir = "./experiment_options"
    make_hparam_opts(experiment_config, hparam_options, output_dir)


def main():
    """
    Main function to use the `make_hparam_opts` function. Take inputs from the command line and run the function.
    1. Load the experiment file.
    2. Read the hyperparameter options from a config file.
    3. Create the list of hyperparameter options.
    4. Save the modified experiment files.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate hyperparameter options for an experiment.")
    parser.add_argument('--experiment_file', type=str, help="Path to the original experiment YAML file.")
    parser.add_argument('--hparam_opts_file', type=str, help="Path to the hyperparameter options YAML file.")
    parser.add_argument('--output_path', type=str, help="Path to save the output files.")
    args = parser.parse_args()

    # Load the experiment file
    with open(args.experiment_file, 'r') as file:
        experiment_file = yaml.safe_load(file)

    # Load the hyperparameter options file
    with open(args.hparam_opts_file, 'r') as file:
        hparam_opts = yaml.safe_load(file)

    # Create the list of hyperparameter options
    make_hparam_opts(experiment_file, hparam_opts, args.output_path)

if __name__ == "__main__":
    main()
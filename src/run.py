import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import OrderedDict
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json
from copy import deepcopy
from contextlib import nullcontext

from train_test_fn import train, test
import logging

from TrainingLogger import (
    TrainingLogger,
)  # Logging that is compatible with Tensorboard and MLFlow

from aux import (
    regression_results,
    build_loss_from_config,
    build_optimizer_fn_from_config,
    save_model as save_model_fn,
    save_predictions,
)


def eval_conditions(eval_metrics, prev_metrics, set_name):
    if set_name not in eval_metrics["Loss"].keys():
        raise ValueError("Val set not found in eval_metrics")

    # TODO: Add parameters to a config
    warning_rate_w = 1 / 2  # The metric is very noisy, so we give it less weight
    mae_w = 1
    loss_w = 1

    rel_change_score = (
        warning_rate_w
        * (
            eval_metrics["PreventedWarningRate"][set_name]
            - prev_metrics["PreventedWarningRate"][set_name]
        )
        / max(prev_metrics["PreventedWarningRate"][set_name], 1e-3)
        + mae_w
        * (prev_metrics["MAE"][set_name] - eval_metrics["MAE"][set_name])
        / prev_metrics["MAE"][set_name]
        + loss_w
        * (prev_metrics["Loss"][set_name] - eval_metrics["Loss"][set_name])
        / prev_metrics["Loss"][set_name]
    ) / (warning_rate_w + mae_w + loss_w)

    result = (rel_change_score) > 0
    print("Score: ", rel_change_score)
    return result


def decide_early_stopping(eval_metrics, prev_metrics):
    if prev_metrics is None:
        return True

    conds_test = eval_conditions(
        eval_metrics, prev_metrics, "Test"  #
    )  # TODO: Change to test when validated that all is working, currently test data has issues so better to use this here and test only in the final evaluation

    return conds_test


def run_torch(
    model,
    train_set,
    validation_sets: dict,
    log_comment="",
    log_hparams=False,
    writer=None,
    total_config=None,
    restore_best_weights=True,
    save_model=False,
    log_frequency=1,
    **config,
):
    """
    Run a test
    """
    best_model = None
    # Use GPU if available
    print("Mps available: ", torch.backends.mps.is_available())
    device = ("mps" if torch.backends.mps.is_available() else "cpu",)

    # Move model to device
    model = model.to(device)

    # Move datasets to device
    # train_set = train_set.to(device)
    # validation_sets = {
    #     name: validation_set.to(device)
    #     for name, validation_set in validation_sets.items()
    # }

    # Create the dataloaders
    train_loader = DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=config.get(
            "shuffle_training_data", False
        ),  # Some current loss function features require this to be False
        num_workers=0,
        persistent_workers=False,
        drop_last=False,
    )

    validation_loaders = {
        name: DataLoader(
            loader,
            batch_size=config["test_batch_size"],
            shuffle=False,
            num_workers=0,
            persistent_workers=False,
            drop_last=False,
        )
        for name, loader in validation_sets.items()
    }

    if writer is None:
        # Create a writer to write to Tensorboard
        writer = TrainingLogger(comment=log_comment, mode="both")
        writer.add_text("run_params.json", config)
    if total_config is not None:
        writer.add_text("all_params.json", total_config)

    # Create loss function and optimizer

    criterion = build_loss_from_config(config)
    optimizer = build_optimizer_fn_from_config(config["optimizer"])(model.parameters())

    patience = config.get("early_stopping_patience", torch.inf)
    logging.info(f"Early stopping patience: {patience}")
    prev_metrics = None
    counter = 0
    best_epoch = 0

    logging.info("Starting initial training")
    try:
        for epoch in tqdm(range(config["epochs"])):
            # Train on data
            train_loss_real = train(
                train_loader,
                model,
                optimizer,
                criterion,
                device,
                config["detect_bad_gradients"],
            )
            writer.add_scalars("Loss", {"Train_loss_real": train_loss_real}, epoch)

            if epoch % log_frequency != 0 and epoch != config["epochs"] - 1:
                # Speed up training by not logging all the time
                continue
            # After training set eval mode on
            model.eval()
            # Test on data
            eval_metrics = {
                "Loss": {},
                "MAE": {},
                "PreventedWarningRate": {},
                "PreventedAlertRate": {},
                "LossDiff": {},
                "R2": {},
            }

            # Evaluate on different sets
            for name, loader in validation_loaders.items():
                if name == "Train":
                    figure_kwargs = {
                        "writer": writer,
                        "lims": "infer",
                        "epoch": epoch,
                        "name": name,
                    }
                elif name == "Val":
                    figure_kwargs = {
                        "writer": writer,
                        "lims": "infer",
                        "epoch": epoch,
                        "name": name,
                    }
                else:
                    figure_kwargs = {
                        "writer": writer,
                        "lims": "infer",
                        "epoch": epoch,
                        "name": name,
                    }

                _loss, _mae, _r2, _prevented_alerts, _prevented_warnings = test(
                    loader, model, criterion, device, figure_kwargs=figure_kwargs
                )
                if "diff" in dir(criterion):
                    _diff = deepcopy(criterion.diff)
                else:
                    _diff = 0

                eval_metrics["Loss"][name] = _loss
                eval_metrics["MAE"][name] = _mae
                eval_metrics["PreventedWarningRate"][name] = _prevented_warnings
                eval_metrics["PreventedAlertRate"][name] = _prevented_alerts
                eval_metrics["LossDiff"][name] = _diff
                eval_metrics["R2"][name] = _r2

            # Write the gathered metrics to Tensorboard
            for name, results in eval_metrics.items():
                writer.add_scalars(name, results, epoch)

            writer.add_scalars(
                "LR",
                {
                    "LR": optimizer.param_groups[0]["lr"],
                },
                epoch,
            )

            if log_hparams:
                report_metrics = {
                    "hparam/" + name.lower() + "_" + metric.lower(): value
                    for name, metrics in eval_metrics.items()
                    for metric, value in metrics.items()
                }
                writer.add_hparams(log_hparams, report_metrics, run_name=log_comment)

            if decide_early_stopping(eval_metrics, prev_metrics):
                logging.info("Improvement, resetting counter")
                best_metrics = {
                    name: eval_metrics[name]["Test"] for name in eval_metrics.keys()
                }

                writer.add_scalars(
                    "Best",
                    best_metrics,
                    epoch,
                )
                counter = 0
                best_model = deepcopy(model)
                best_optimizer = deepcopy(optimizer)
                best_epoch = epoch
                prev_metrics = eval_metrics
            else:
                logging.info("No improvement, increasing counter")
                counter += log_frequency  # since we are not logging every epoch
                if counter > patience:
                    logging.info("Initiating early stopping")
                    logging.info(f"Best weights found on epoch {best_epoch}")
                    break

        logging.info("\nTraining Finished.")

        writer.flush()
        writer.close()

        # Finally, use the model to predict the train, validation and test sets
    except KeyboardInterrupt:
        logging.info("Interrupted")

    logging.info("Gathering final predictions")
    last_model = model
    if best_model is not None and restore_best_weights:
        logging.info(f"Restoring best weights from epoch {best_epoch}")
        model = best_model
        optimizer = best_optimizer
    else:
        best_model = last_model

    if save_model:
        best_model_save_path, best_filename = save_model_fn(
            best_model, total_config, "best-model"
        )
        last_model_save_path, last_filename = save_model_fn(
            last_model, total_config, "last-model"
        )
    else:
        best_model_save_path = ""
        last_model_save_path = ""

    if restore_best_weights:
        results_best, predictions_best, model_best = gather_results(
            best_model, validation_loaders, device=device
        )
        results_last, predictions_last, model_last = gather_results(
            last_model, validation_loaders, device=device
        )

        if log_hparams:
            res_dict = results_best.loc[
                ["MAE", "r2", "RMSE", "correct_alert_rate", "correct_warning_rate"]
            ]
            res_dict = res_dict.rename(
                columns={
                    "correct_alert_rate": "prevented_alerts",
                    "correct_warning_rate": "prevented_warnings",
                }
            )
            res_dict = res_dict.to_dict()
            report_metrics = {
                "hparam/best/" + name.lower() + "_" + metric.lower(): value
                for name, metrics in res_dict.items()
                for metric, value in metrics.items()
            }
            writer.add_hparams(log_hparams, report_metrics, run_name=log_comment)

        if save_model:
            logging.info(f"Saving results of {best_model_save_path}")
            results_best.to_csv(best_model_save_path / "results.csv")
            for name, validation_set in validation_sets.items():
                save_predictions(
                    predictions_best[name],
                    name=name,
                    save_path=best_model_save_path,
                    dt_index=validation_set.index,
                )
            # predictions_best.to_csv(best_model_save_path / (best_filename + "-predictions.csv"))
            logging.info(f"Saving results of {last_model_save_path}")
            results_last.to_csv(last_model_save_path / "results.csv")

            for name, validation_set in validation_sets.items():
                save_predictions(
                    predictions_last[name],
                    name=name,
                    save_path=last_model_save_path,
                    dt_index=validation_set.index,
                )

        return {
            "best": (results_best, predictions_best, model_best),
            "last": (results_last, predictions_last, model_last),
            "best_model_save_path": best_model_save_path,
            "last_model_save_path": last_model_save_path,
        }
    else:
        return {"last": gather_results(last_model, validation_loaders, device=device)}


def gather_results_single(
    model,
    loader,
    device,
    dt_index=None,
    modified=False,
    online_learning=None,
):
    model.eval()
    model.to(device)

    if online_learning is None:
        context = torch.no_grad
    else:
        context = nullcontext

    with context():
        y = []
        y_pred = []
        full_mask = []
        for data in loader:
            # Weights not used for gathering results, only for training
            if len(data) == 2:
                inputs, labels = data
            else:
                inputs, labels, _ = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            if modified:
                lims = [0.4, 0.4]  # , 1.0]
                shift_idx = 6  # TODO: Change if change to 60min
                len_prepend = min([shift_idx, labels.shape[0]])
                warning_time = torch.concat(
                    [
                        torch.Tensor([0] * len_prepend).unsqueeze(-1),
                        (labels[:-shift_idx] <= lims[0])
                        & (labels[shift_idx:] > lims[0]),
                    ]
                ).long()
                alert_time = torch.concat(
                    [
                        torch.Tensor([0] * len_prepend).unsqueeze(-1),
                        (labels[:-shift_idx] <= lims[1])
                        & (labels[shift_idx:] > lims[1]),
                    ]
                ).long()
                # If it is a warning or alert, we want to predict it
                mask = (
                    (warning_time | alert_time)
                    .detach()
                    .cpu()
                    .numpy()
                    .cpu()
                    .flatten()
                    .astype(bool)
                )

            if "init_hidden" in dir(model):
                # previous_turbidity = inputs[:, :, model.prev_turbidity_idx]
                # model.init_hidden(inputs.size(0), previous_turbidity)
                model.init_hidden(inputs.size(0))
            pred = model(inputs)
            if modified:
                full_mask.extend(mask)

            if (
                online_learning
            ):  # If we are doing online learning, we need to backpropagate before next epoch
                raise NotImplementedError("Online learning not implemented currently")
                meta_model = online_learning["meta_model"]
                criterion = online_learning["criterion"]
                # Create the optimizer for this batch
                if "init_hidden" in dir(meta_model):
                    meta_model.init_hidden(inputs.size(0))
                online_learning["optimizer"].zero_grad()
                loss_coeff = meta_model(inputs, labels)
                loss = criterion(pred, labels) * loss_coeff
                loss.backward()
                online_learning["optimizer"].step()

            labels = labels.detach().cpu().numpy().flatten()
            pred = pred.detach().cpu().numpy().flatten()

            y.extend(labels)
            y_pred.extend(pred)

        y = np.array(y)
        y_pred = np.array(y_pred)

        if modified:
            full_mask = np.array(full_mask)
            y = y[full_mask]
            y_pred = y_pred[full_mask]
            if dt_index is not None:
                dt_index = dt_index[full_mask]

        reg_res = regression_results(y, y_pred)

        if modified:
            # Dont try to classify with the modified data
            return {"res": reg_res, "predictions": {"y": y, "pred": y_pred}}

        if dt_index is not None:
            y = pd.Series(data=y, index=dt_index)
            y_pred = pd.Series(data=y_pred, index=dt_index)

    return {"res": {**reg_res}, "predictions": {"y": y, "pred": y_pred}}


def gather_results(model, loader_dict: dict, device, online_learning=None):
    """
    Gather the results for train, val and test sets.
    Returns:
        results, predictions, model
    """

    model.eval()

    full_results = {}

    for name, loader in loader_dict.items():
        if "index" in dir(loader.dataset):
            res = gather_results_single(
                model,
                loader,
                dt_index=loader.dataset.index,
                online_learning=online_learning,
                device=device,
            )
        else:
            res = gather_results_single(
                model, loader, online_learning=online_learning, device=device
            )

        full_results[name] = res

    results = pd.DataFrame(
        {name: result["res"] for name, result in full_results.items()}
    )
    predictions = {
        name: pd.DataFrame(
            {"y": result["predictions"]["y"], "pred": result["predictions"]["pred"]}
        )
        for name, result in full_results.items()
    }

    return results, predictions, model


def gather_results_modified(model, loader_dict: dict, device):
    """
    Gather the results for train, val and test sets, but only for the parts of the data where it is critical to predict correctly.
    Returns:
        results, predictions, model
    """

    model.eval()

    full_results = {}

    for name, loader in loader_dict.items():
        if "index" in dir(loader.dataset):
            res = gather_results_single(
                model,
                loader,
                dt_index=loader.dataset.index,
                modified=True,
                device=device,
            )
        else:
            res = gather_results_single(model, loader, modified=True, device=device)

        full_results[name] = res

    results = pd.DataFrame(
        {name: result["res"] for name, result in full_results.items()}
    )
    predictions = {
        name: pd.DataFrame(
            {"y": result["predictions"]["y"], "pred": result["predictions"]["pred"]}
        )
        for name, result in full_results.items()
    }

    return results, predictions, model

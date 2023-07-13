import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics

logger = logging.getLogger(__name__)


def calculate_prediction_errors(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    assert y_true.ndim == y_pred.ndim, f"{y_true.shape=}, {y_pred.shape=}"
    if y_true.ndim <= 2:
        try:
            me = metrics.max_error(y_true, y_pred)
        except ValueError:
            explanation = (
                " The exception can happen if y_true or y_pred has a second dimension of size greater than 1."
                " For example, if batch size is greater 1 and a prediction vector (not point) is the output."
                " Then, y_pred is  categorized as continuous multi-output,"
                " which is not supported. Hence, both dimensions are reshaped"
                " and the maximum error over the entire"
                " batch and the entire prediction vector is calculated."
            )
            logger.warning(
                f"Value Error occurred, y_true and y_pred are reshaped.\n"
                f"{explanation}"
            )
            me = metrics.max_error(y_true.reshape((-1)), y_pred.reshape((-1)))
        return dict(
            mse=metrics.mean_squared_error(y_true, y_pred),
            rmse=metrics.mean_squared_error(y_true, y_pred, squared=False),
            mae=metrics.mean_absolute_error(y_true, y_pred),
            me=me,
        )
    elif y_true.ndim == 3:
        y_true, y_pred = torch.as_tensor(y_true), torch.as_tensor(y_pred)
        mse_function = nn.MSELoss()
        mse = mse_function(y_true, y_pred).item()
        rmse = np.sqrt(mse)
        mae_loss = nn.L1Loss()
        mae = mae_loss(y_true, y_pred).item()
        try:
            me = torch.max(torch.sub(y_true, y_pred).abs()).item()
        except RuntimeError:
            logger.info(
                f"RuntimeError is catched \n " f"{y_true.shape=}\n" f"{y_pred.shape}"
            )
            me = torch.max(torch.sub(y_true, y_pred).abs()).item()
        return dict(
            mse=mse,
            rmse=rmse,
            mae=mae,
            me=me,
        )
    else:
        raise ValueError


def calculate_relative_rmse(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    future_steps: int,
) -> float:
    """For details, see Ernst et al 2013 (DOI: 10.1088/0031-9155/58/11/3911)
    (section 3.4)."""

    # delay y_true based on future_steps
    y_true_delayed = np.concatenate((np.zeros((future_steps,)), y_true[:-future_steps]))
    # cut y_pred, y_true, y_true_delayed
    y_pred = y_pred[future_steps:]
    y_true_delayed = y_true_delayed[future_steps:]
    y_true = y_true[future_steps:]
    if not y_true.shape == y_true_delayed.shape == y_pred.shape:
        raise ValueError(
            f"Shape error.... \n"
            f"{y_true.shape=}\n"
            f"{y_true_delayed.shape=}\n"
            f"{y_pred.shape=}"
        )
    rmse = metrics.mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False)
    scaler = metrics.mean_squared_error(
        y_true=y_true,
        y_pred=y_true_delayed,
        squared=False,
    )
    relative_rmse = rmse / scaler
    return relative_rmse

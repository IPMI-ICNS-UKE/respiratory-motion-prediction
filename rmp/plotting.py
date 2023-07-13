"""
Simple functions for plotting signals during training and final results.
Also, visualization of attention heads and corresponding tikz code.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from rmp import metrics
from rmp.dataloader import ModelPhases
from rmp.my_utils.common_types import PathLike
from rmp.my_utils.decorators import convert

logger = logging.getLogger(__name__)


@convert("result_dir", converter=Path)
def plot_random_signal_in_batch(
    names: Iterable,
    features: np.ndarray,
    targets: np.ndarray,
    outputs: np.ndarray,
    seen_curves: int,
    result_dir: PathLike,
    future_steps: int,
    phase: ModelPhases = ModelPhases.TRAINING,
    max_plots: int | None = 10,
):
    logger.debug(f"{targets.shape=} \n {outputs.shape=} \n ")
    if not (targets.shape == outputs.shape):
        raise ValueError(
            f"Shapes should be equal:\n"
            f"{targets.shape=}\n"
            f"{outputs.shape=} \n {features.shape=}"
        )
    if not targets.ndim == 3:
        raise NotImplementedError(
            "Only works if target has three dimensions, i.e. batch, seq_len, input_features"
        )

    result_dir = result_dir / str(seen_curves)
    result_dir.mkdir(exist_ok=True)
    names = names[:max_plots]
    for index, name in enumerate(names):
        path_to_png = result_dir / f"{phase.value}_{name.replace('csv', 'png')}"
        fig, axis = plt.subplots(
            nrows=2, ncols=1, sharex=True, gridspec_kw={"height_ratios": [2, 1]}
        )
        output = outputs[index, :, :].reshape(-1)
        target = targets[index, :, :].reshape(-1)
        time = np.arange(0, len(target), 1)
        pred_errors = metrics.calculate_prediction_errors(y_true=target, y_pred=output)
        axis[0].set_title(f"{pred_errors}", fontsize=10)
        axis[0].plot(time, target, label="target")
        axis[0].plot(time, output, label="output")
        axis[0].plot(
            np.arange(0, future_steps, 1),
            np.zeros(future_steps),
            label=f"Horizon:{40 * future_steps}ms",
        )
        axis[0].set_ylim([-1, 1])
        axis[0].legend(loc=1)
        axis[1].plot(time, np.subtract(target, output), label="mae")
        axis[1].set_ylim([-1, 1])
        axis[1].legend(loc=1)
        fig.savefig(path_to_png)
        plt.close("all")

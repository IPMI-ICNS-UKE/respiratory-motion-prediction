from __future__ import annotations

import logging

import requests

from rmp.global_config import DATALAKE
from rmp.models import ModelArch

logger = logging.getLogger(__name__)


def is_natural_number(n) -> bool:
    return isinstance(n, int) and n > 0


def input_checks(config: dict) -> dict:  # noqa: C901
    """Performs trivial input checks to catch easy errors."""
    if not isinstance(config["model_arch"], ModelArch):
        raise NotImplementedError(f"{config['model_arch']=} is not supported yet.")
    if not is_natural_number(config["train_batch_size"]):
        raise ValueError(
            f"Batch size have to be natural numbers but {config['train_batch_size']}"
        )
    if not is_natural_number(config["eval_batch_size"]):
        raise ValueError(
            f"Batch size have to be natural numbers but {config['eval_batch_size']}"
        )
    if not (
        1 <= config["future_steps"] <= 100 and is_natural_number(config["future_steps"])
    ):
        raise ValueError(
            f"Phrased prediction task is to output the value which is {config['future_steps'] * 40} ms ahead. "
            f"-> invalid/not supported"
        )
    if not is_natural_number(config["train_min_length_s"]):
        raise ValueError(f"{config['train_min_length_s']=} natural number required")
    if not is_natural_number(config["test_signal_length_s"]):
        raise ValueError(f"{config['test_signal_length_s']=} natural number required")

    if config["model_arch"] in [ModelArch.XGBOOST, ModelArch.LINEAR_OFFLINE]:
        if config["train_signal_length_s"] is not None:
            initial_value = config["train_signal_length_s"]
            config["train_signal_length_s"] = None
            logger.warning(
                f"train_signal_length_s of {initial_value} seconds was set to None. Thus, all available sliding windows"
                f"are used"
            )
    if config["model_arch"] is ModelArch.DLINEAR and config["train_batch_size"] != 1:
        config["train_batch_size"] = 1
        logger.warning(
            f"{ config['train_batch_size']=} was set to 1 since {config['model_arch']=} only supports that."
        )
    if config["train_min_length_s"] < config["training_phase_s"]:
        raise ValueError(
            f"training signal subset ({config['train_min_length_s']=},{config['test_signal_length_s']=}) "
            f"is shorter than training period ({config['training_phase_s']=}). \n "
            f"-> Increase train_min_length_s or test_signal_length_s"
        )
    if config["test_signal_length_s"] < config["training_phase_s"]:
        raise ValueError(
            f"test signal subset ({config['test_signal_length_s']=}) "
            f"is shorter than training period ({config['training_phase_s']=}). \n "
            f"-> Increase test_signal_length_s"
        )
    if (
        config["train_signal_length_s"]
        and config["train_signal_length_s"] < config["training_phase_s"]
    ):
        raise ValueError(
            f"Training signal subset {config['train_signal_length_s']=} "
            f"is shorter than the training period ({config['training_phase_s']=}! \n "
            f"-> Increase train_signal_length_s."
        )
    if (
        config["train_signal_length_s"]
        and config["train_signal_length_s"] > config["train_min_length_s"]
    ):
        raise ValueError(
            f"All signals of the training set have to be longer than the "
            f"signal subset.\n "
            f"Otherwise, sampling is impossile. \n"
            f"{config['train_signal_length_s']=} vs {config['train_min_length_s']=}"
        )
    try:
        response = requests.get(DATALAKE + "/docs")
        response.raise_for_status()
        logger.info(f"API connection to {DATALAKE} is successful.")
    except requests.exceptions.RequestException as e:
        logger.error(f"API connection to {DATALAKE} failed. Error: {e}")
        raise requests.HTTPError(f"API connection to {DATALAKE} failed. Error: {e}")
    return config

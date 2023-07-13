"""Main script to start model training.

Starts and runs the hyperopt pipeline.
"""
from __future__ import annotations

import logging

from rmp.global_config import DATALAKE
from rmp.hyperoptimizer import Hyperoptimizer
from rmp.models import ModelArch
from rmp.my_utils.input_paras_check import input_checks
from rmp.my_utils.logger import init_fancy_logging
from rmp.search_spaces import (
    SEARCH_SPACE_CUSTOM_MODEL,
    SEARCH_SPACE_DLINEAR,
    SEARCH_SPACE_LINEAR,
    SEARCH_SPACE_LSTM,
    SEARCH_SPACE_TRANSFORMER,
    SEARCH_SPACE_XGB,
    SEARCH_SPACE_TRANSFORMER_TSFv2,
)

logger = logging.getLogger(__name__)


def make_config(
    model_arch: ModelArch,
    train_batch_size: int,
    eval_batch_size: int,
    future_steps: int,
    train_signal_length_s: None | int,
    train_min_length_s: int,
    test_signal_length_s: int,
):
    """Creating and checking configuration input before staring hyperparameter
    search.

    :param model_arch: chosen model architecture
    :param train_batch_size: batch size for training
    :param eval_batch_size: batch size for evaluation, i.e. validation and testing
    :param future_steps : determines the prediction horizon, samples per second = 25 HZ; -> 12: 480ms ; 17: 680ms ; 23: 920ms
    :param train_signal_length_s: determines length of random singal subset (allows stacking/batching signal subsets). If None, entire signal is used.
    :param train_min_length_s: all signals shorter than train_min_length_s will be excluded from train set
    :param test_signal_length_s: all signals shorter than test_signal_length_s will be excluded from val/test set # ToDo
    :return:
    """
    flexible_config = dict(
        model_arch=model_arch,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        future_steps=future_steps,
        db_root=DATALAKE,  # Path to downloaded db file -> change in STP.global_config
        train_signal_length_s=train_signal_length_s,
        train_min_length_s=train_min_length_s,
        test_signal_length_s=test_signal_length_s,
    )
    # default config utilized in our study
    fixed_config = dict(
        output_features=1,  # single point prediction
        dropout=0.1,
        weight_decay=0.01,
        lr_scheduler_linear_decay=0.97,
        early_stopper_criteria=dict(patience=4_000, min_delta=0.01),
        max_num_curves=None,  # use all signals from set
        scaling_period_s=(
            0,
            20,
        ),  # min/max scaling is performed on the first 20s of each signal
        white_noise_db=27,  # white noise in dB is added to each signal input
        use_smooth_target=True,  # targets are sampled from smoothed signal
        cache_returned_data=False,
        training_phase_s=20,  # initial learning period at beginning of each signal,
    )

    final_config = {**flexible_config, **fixed_config}
    final_config = input_checks(final_config)
    logger.info(
        f"\n \033[4m Started Hyperopt Report:\033[0m \n"
        f"Chosen model: {final_config['model_arch']}\n"
        f"Applied database: {DATALAKE} \n"
        f"Prediction task and information:\n"
        f"Predict the value which is {final_config['future_steps'] * 40}ms ahead (single point prediction). \n"
        f"Input signal are noisy, signal to noise ratio is {final_config['white_noise_db']} dB. \n"
        f"Targets are sampled from corresponding smooth signal: {final_config['use_smooth_target']}. \n"
        f"MinMax scaling is performed on the first {final_config['scaling_period_s']}s of each signal. \n"
        f" \033[4m End of  Hyperopt Report\033[0m \n"
    )
    return final_config


if __name__ == "__main__":
    init_fancy_logging()
    logger.setLevel(logging.INFO)
    logging.getLogger("rmp").setLevel(logging.INFO)
    MODEL_ARCH = ModelArch.LSTM  # change model type here

    config = make_config(
        model_arch=MODEL_ARCH,
        train_batch_size=1,
        eval_batch_size=1,
        future_steps=12,  # change prediction horizon here;  1 = 40ms, 12 = 480ms
        train_signal_length_s=50,
        train_min_length_s=59,
        test_signal_length_s=59,
    )
    if MODEL_ARCH is ModelArch.LINEAR_OFFLINE:
        search_space = SEARCH_SPACE_LINEAR
    elif MODEL_ARCH is ModelArch.LSTM:
        search_space = SEARCH_SPACE_LSTM
    elif MODEL_ARCH is ModelArch.TRANSFORMER_ENCODER:
        search_space = SEARCH_SPACE_TRANSFORMER
    elif MODEL_ARCH is ModelArch.TRANSFORMER_TSF:
        search_space = SEARCH_SPACE_TRANSFORMER_TSFv2
    elif MODEL_ARCH is ModelArch.DLINEAR:
        search_space = SEARCH_SPACE_DLINEAR
    elif MODEL_ARCH is ModelArch.XGBOOST:
        search_space = SEARCH_SPACE_XGB
    elif MODEL_ARCH is ModelArch.CUSTOM_MODEL:
        search_space = SEARCH_SPACE_CUSTOM_MODEL
    else:
        raise NotImplementedError(
            f"{MODEL_ARCH} not supported in current implementation"
        )

    hyper = Hyperoptimizer(
        search_space=search_space,
        constant_config=config,
    )
    hyper.run_forever()

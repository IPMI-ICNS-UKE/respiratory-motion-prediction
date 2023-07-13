"""For each model, a hyperparameter search space is defined."""

import numpy as np
from hyperopt import hp

SEARCH_SPACE_LINEAR = {
    "input_features": hp.choice("input_features", list(range(435, 501, 10))),
    "alpha": hp.choice(
        "alpha",
        [
            1 * 1e-05,
            5 * 10e-05,
            10e-04,
            5 * 10e-04,
            1 * 10e-03,
            5 * 10e-03,
            1 * 10e-02,
            5 * 10e-02,
            1 * 10e-01,
            0,
            0.5,
            1,
            1.5,
            2,
            10,
        ],
    ),
}
SEARCH_SPACE_LSTM = {
    "max_iterations": hp.choice(
        "max_iterations",
        [20_000],
    ),
    "learning_rate": hp.choice(
        "learning_rate",
        [0.001],
    ),
    "input_features": hp.choice("input_features", [4, 8, 16, 32, 64, 128]),
    "num_layers": hp.choice("num_layers", [1, 2, 3, 4, 8]),
    "hidden_dim": hp.choice("hidden_dim", [8, 32, 64, 128, 256]),
}
SEARCH_SPACE_TRANSFORMER = {
    "max_iterations": hp.choice(
        "max_iterations",
        [20_000],
    ),
    "learning_rate": hp.choice(
        "learning_rate",
        [10e-04],
    ),
    "dropout": hp.choice("dropout", [0.1]),
    "input_features": hp.choice("input_features", [1, 4, 8, 16, 32, 64]),
    "num_layers": hp.choice("num_layers", [1, 4, 8]),
    "n_heads": hp.choice("n_heads", [1, 4, 8, 16]),
    "embedding_dim": hp.choice("embedding_dim", [8, 32, 64, 128]),
}

SEARCH_SPACE_TRANSFORMER_TSFv2 = {
    "max_iterations": hp.choice(
        "max_iterations",
        [15_000],
    ),
    "learning_rate": hp.choice(
        "learning_rate",
        [10e-04],  # 10e-04, 5 * 10e-04, 10e-03],
    ),
    "input_features": hp.choice(
        "input_features",
        list(range(25, 101, 5)),
    ),  # ),
    "num_layers": hp.choice("num_layers", [1, 2, 4, 8]),
    "n_heads": hp.choice("n_heads", [1, 4, 8]),
    "embedding_dim": hp.choice("embedding_dim", [8, 32, 64]),
}

SEARCH_SPACE_DLINEAR = {
    "max_iterations": hp.choice(
        "max_iterations",
        [10_000],
    ),
    "learning_rate": hp.choice(
        "learning_rate",
        [0.001],
    ),
    "input_features": hp.choice(
        "input_features",
        list(range(25, 501, 25)),
    ),
}
SEARCH_SPACE_XGB = {
    "input_features": hp.choice(
        "input_features",
        list(range(150, 501, 10)),
    ),
    "subsample": hp.choice("subsample", [0.95]),
    "max_depth": hp.quniform("max_depth", 3, 18, 1),
    "gamma": hp.uniform("gamma", 0, 4),
    "min_child_weight": hp.quniform("min_child_weight", 2, 10, 1),
    "learning_rate": hp.quniform("learning_rate", 0.01, 0.2, 0.01),
    "n_estimators": hp.choice(
        "n_estimators",
        list(range(20, 501, 10)),
    ),
    "reg_lambda": hp.loguniform("reg_lambda", np.log(1), np.log(100)),
}

SEARCH_SPACE_CUSTOM_MODEL = {
    # for deep learning required hyperparameters:
    "input_features": hp.choice(
        "input_features",
        list(range(150, 501, 10)),
    ),
    "learning_rate": hp.quniform("learning_rate", 0.01, 0.2, 0.01),
    "max_iterations": hp.choice(
        "max_iterations",
        [10_000],
    )
    # add hyperparameters of your custom model here
}

"""Script to evaluate a trained model on the test set."""

import logging

from rmp.eval_model import Eval
from rmp.utils.logger import init_fancy_logging

if __name__ == "__main__":
    init_fancy_logging()
    logging.getLogger("rmp").setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # print all pre-defined functions which can be selected for Eval. ...
    # logger.info('\n'.join(list(filter(lambda x: x.startswith("init"), dir(Eval)))))

    evaluater = Eval.init_dlinear_480()  # select model and pred horizon here
    evaluater.eval_saved_model_using_test_set()

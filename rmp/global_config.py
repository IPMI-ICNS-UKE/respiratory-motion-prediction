import logging
from datetime import datetime
from pathlib import Path

import torch

from rmp.utils.logger import init_fancy_logging

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RESULT_DIR = Path(".")  # change
RESULT_DIR.mkdir(exist_ok=True)
DATALAKE = "https://lukasvm.iam-extern.de/api/resp-db/v1"
NUM_WORKERS = 0

SAVED_MODELS_DIR = Path("../trained_models")
LOGGING_FILE = RESULT_DIR / f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}_debug.log"


init_fancy_logging(
    handlers=[logging.StreamHandler(), logging.FileHandler(LOGGING_FILE)]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.info("DATALAKE: %s", DATALAKE)
logger.info("RESULT_DIR: %s", RESULT_DIR)
logger.info("DEVICE: %s", DEVICE)
logger.info("NUM_WORKERS: %s", NUM_WORKERS)
logger.info("LOGGING FILE: %s", LOGGING_FILE)

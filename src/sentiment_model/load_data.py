"""Loading of splits of the data."""

import hydra
from omegaconf import DictConfig
from pathlib import Path
import os

import pandas as pd
import logging

logger = logging.getLogger(__name__)


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def load_splits(config: DictConfig) -> dict:
    """Loading of training splits, split into a training, validation and test set.

    Args:
        config (DictConfig):
            Configuration object.

    Returns:
        dict:
            A dictionary with a keys `train`, `val` and `test`, containing the
            training, validation and test data, respectively.

    Raises:
        FileNotFoundError:
            If one of the files was not found.
    """
    # Set up the paths to the data
    train_path = Path(config.paths.train)
    val_path = Path(config.paths.val)
    test_path = Path(config.paths.test)

    # Makes raises if any of the files are missing
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"File {train_path} not found.")

    if not os.path.exists(val_path):
        raise FileNotFoundError(f"File {val_path} not found.")

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"File {test_path} not found.")

    # Log loading of dataset
    logger.info(f"Loading data from {train_path}, {val_path} and {test_path}")

    # Read the csv files
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)

    # Log the number of rows in the dataframe
    logger.info(
        f"Loaded {len(train):,}, {len(val):,} and {len(test):,} rows from the "
        "training, validation and test sets, respectively"
    )

    # Return a dictionary containing the training, validation and test data
    return dict(train=train, val=val, test=test)

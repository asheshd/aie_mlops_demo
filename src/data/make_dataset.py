# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from sklearn.datasets import make_classification

from sklearn.model_selection import train_test_split

import numpy as np
import os


def main(output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    X, y = make_classification(n_samples=100000, random_state=51)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=51)

    np.savetxt(os.path.join(output_filepath, "train_features.csv"), X_train)
    np.savetxt(os.path.join(output_filepath, "train_labels.csv"), y_train)

    np.savetxt(os.path.join(output_filepath, "test_features.csv"), X_test)
    np.savetxt(os.path.join(output_filepath, "test_labels.csv"), y_test)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    output_filepath = "data/processed"
    main(output_filepath)

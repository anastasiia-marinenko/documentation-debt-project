import pandas as pd

from config import DATASET_PATH


def load_dataset():
    return pd.read_csv(DATASET_PATH)
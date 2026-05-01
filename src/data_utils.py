"""Data loading helpers."""

import pandas as pd

from .config import RAW_DATA_PATH


def load_raw_data(path=RAW_DATA_PATH):
    df = pd.read_csv(path, sep=";")
    df.columns = [str(c).replace("\ufeff", "").replace("\t", " ").replace("\r", "").strip() for c in df.columns]
    return df

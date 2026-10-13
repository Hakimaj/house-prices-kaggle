"""Loading the Kaggle files and preparing X and y."""
import numpy as np
import pandas as pd

from .config import DATA_DIR, ID_COL, TARGET


def load_train() -> pd.DataFrame:
 return pd.read_csv(DATA_DIR / "train.csv")


def load_test() -> pd.DataFrame:
 return pd.read_csv(DATA_DIR / "test.csv")


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
 """Drop the documented outliers: very large houses that sold for very little.
 The dataset's author recommends removing houses above 4,000 sq ft of living area."""
 mask = (df["GrLivArea"] > 4000) & (df[TARGET] < 300_000)
 return df.loc[~mask].reset_index(drop=True)


def split_xy(df: pd.DataFrame):
    """Features X (without Id) and the LOG-transformed target y."""
    X = df.drop(columns=[TARGET, ID_COL])
    y = np.log1p(df[TARGET])
    return X, y
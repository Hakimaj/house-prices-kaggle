"""Feature engineering (stateless) and the transformers that LEARN from the training data."""
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin


def add_features(df: pd.DataFrame) -> pd.DataFrame:
 """New columns made from existing ones. Call AFTER clean()."""
 f = df.copy()
 df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
 df["TotalBath"] = df["FullBath"] + 0.5 * df["HalfBath"] + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
 df["TotalPorchSF"] = (df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"]
 + df["ScreenPorch"] + df["WoodDeckSF"])
 df["HouseAge"] = (df["YrSold"] - df["YearBuilt"]).clip(lower=0) # a few houses sold before "built"
 df["RemodAge"] = (df["YrSold"] - df["YearRemodAdd"]).clip(lower=0)
 df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)
 df["IsNew"] = (df["YrSold"] == df["YearBuilt"]).astype(int)
 df["HasPool"] = (df["PoolArea"] > 0).astype(int)
 df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
 df["HasBasement"] = (df["TotalBsmtSF"] > 0).astype(int)
 df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
 df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype(int)
 df["QualSF"] = df["OverallQual"] * df["TotalSF"] # quality x size interaction
 return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
 """Everything stateless, in the right order. First step of every model pipeline."""
 from .cleaning import clean
 return add_features(clean(df))

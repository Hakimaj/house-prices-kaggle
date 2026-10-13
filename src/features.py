"""Feature engineering (stateless) and the transformers that LEARN from the training data."""

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin

def add_features(df: pd.DataFrame) -> pd.DataFrame:
 """New columns made from existing ones. Call AFTER clean()."""
 df = df.copy()
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

class LotFrontageImputer(BaseEstimator, TransformerMixin):
 """Fill missing LotFrontage with the median of the house's neighbourhood.
 The medians are learned on the training rows only (fit), then applied anywhere (transform)."""
 def fit(self, X, y=None):
    self.neighborhood_median_ = X.groupby("Neighborhood")["LotFrontage"].median()
    self.global_median_ = X["LotFrontage"].median()
    return self

 def transform(self, X):
    X = X.copy()
    fill = X["Neighborhood"].map(self.neighborhood_median_).fillna(self.global_median_)
    X["LotFrontage"] = X["LotFrontage"].fillna(fill)
    return X


class SkewFixer(BaseEstimator, TransformerMixin):
 """Apply log(1 + x) to the columns that are strongly right-skewed in the TRAINING data."""
 def __init__(self, threshold: float = 0.75):
    self.threshold = threshold
 def fit(self, X, y=None):
    X = np.asarray(X, dtype=float)
    with np.errstate(all="ignore"):
        sk = np.nan_to_num(skew(X, axis=0))
    not_binary = np.array([len(np.unique(X[:, j])) > 2 for j in range(X.shape[1])])
    non_negative = X.min(axis=0) >= 0
    self.skewed_ = np.where((np.abs(sk) > self.threshold) & non_negative & not_binary)[0]
    self.n_features_in_ = X.shape[1]
    return self

 def transform(self, X):
    X = np.array(X, dtype=float, copy=True)
    X[:, self.skewed_] = np.log1p(X[:, self.skewed_])
    return X

 def get_feature_names_out(self, input_features=None):
    return np.asarray(input_features, dtype=object)
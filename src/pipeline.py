"""Preprocessing and model pipelines. Every model gets the SAME preprocessing."""
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler

from .config import RANDOM_STATE
from .features import LotFrontageImputer, SkewFixer, prepare

def build_preprocessor(scale: bool = True) -> ColumnTransformer:
 """Numbers: fill gaps, fix skew, (scale). Categories: fill gaps, one-hot encode."""
 num_steps = [("impute", SimpleImputer(strategy="median")), ("skew", SkewFixer())]
 if scale:
    num_steps.append(("scale", RobustScaler()))
 numeric = Pipeline(num_steps)
 categorical = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10, sparse_output=False)),
 ])
 return ColumnTransformer([
    ("num", numeric, make_column_selector(dtype_include="number")),
    ("cat", categorical, make_column_selector(dtype_exclude="number")),
 ])
 
 
def build_pipeline(model, scale: bool = True) -> Pipeline:
 return Pipeline([
 ("prepare", FunctionTransformer(prepare)), # clean + new features (no learning)
 ("lotfrontage", LotFrontageImputer()), # learned: neighbourhood medians
 ("prep", build_preprocessor(scale)), # learned: medians, skew, scaling, categories
 ("model", model),
 ])
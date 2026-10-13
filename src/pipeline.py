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


def make_models() -> dict:
   """The models we compare and blend. Linear models choose their own regularisation strength by CV."""
   alphas = np.logspace(-4, 1, 30)
   return dict({
      "ridge": build_pipeline(RidgeCV(alphas=np.logspace(-1, 2.5, 30))),
      "lasso": build_pipeline(LassoCV(alphas=alphas[:20], cv=5, max_iter=50_000, random_state=RANDOM_STATE)),
      "elasticnet": build_pipeline(ElasticNetCV(l1_ratio=[0.3, 0.6, 0.9], alphas=alphas[:20], cv=5,
                                                max_iter=50_000, random_state=RANDOM_STATE)),
      "gbr": build_pipeline(GradientBoostingRegressor(
         n_estimators=1000, learning_rate=0.03, max_depth=4, max_features="sqrt",
         min_samples_leaf=15, min_samples_split=10, loss="huber", random_state=RANDOM_STATE), scale=False),
      "forest": build_pipeline(RandomForestRegressor(
      n_estimators=300, max_features=0.33, min_samples_leaf=2,
      n_jobs=-1, random_state=RANDOM_STATE), scale=False),
      })

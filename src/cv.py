"""Scoring, out-of-fold predictions and blending."""
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score

from .config import RANDOM_STATE

CV = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

def rmse(y_true, y_pred) -> float:
 return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))

def cv_rmse(model, X, y, cv=CV) -> tuple:
 """Mean and std of RMSE (on the log target = the Kaggle metric) over the folds."""
 scores = -cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
 return float(scores.mean()), float(scores.std())

def oof_predictions(models: dict, X, y, cv=CV) -> pd.DataFrame:
 """For each model: predictions for every training row made by a model that did NOT see that row."""
 out = {name: cross_val_predict(m, X, y, cv=cv) for name, m in models.items()}
 return pd.DataFrame(out, index=X.index)


def blend_weights(oof: pd.DataFrame, y) -> pd.Series:
 """Non-negative weights that best combine the models' out-of-fold predictions; they sum to 1."""
 w, _ = nnls(oof.values, np.asarray(y))
 return pd.Series(w / w.sum(), index=oof.columns)
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
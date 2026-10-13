import numpy as np
import pandas as pd

from src.cleaning import clean
from src.data import load_train, remove_outliers, split_xy
from src.features import SkewFixer, add_features, prepare

def tiny():
 df = load_train().drop(columns="SalePrice").head(60).copy()
 return df

def test_clean_leaves_no_missing_in_absence_columns():
 out = clean(tiny())
 for col in ["Alley", "PoolQC", "GarageType", "MasVnrType", "MasVnrArea", "TotalBsmtSF"]:
    assert not out[col].isna().any(), col
    
def test_clean_encodes_quality_as_ordered_numbers():
 df = tiny(); df.loc[df.index[0], "ExterQual"] = "Ex"; df.loc[df.index[1], "ExterQual"] = "Fa"
 out = clean(df)
 assert out.loc[df.index[0], "ExterQual"] > out.loc[df.index[1], "ExterQual"]

def test_features_are_added_and_input_not_modified():
 df = clean(tiny()); cols_before = list(df.columns)
 out = add_features(df)
 assert {"TotalSF", "TotalBath", "HouseAge"} <= set(out.columns)
 assert list(df.columns) == cols_before
 assert (out["HouseAge"] >= 0).all()

def test_remove_outliers_drops_big_cheap_houses_only():
 df = load_train(); out = remove_outliers(df)
 assert len(out) < len(df)
 assert not ((out["GrLivArea"] > 4000) & (out["SalePrice"] < 300_000)).any()

def test_target_is_log_transformed():
 X, y = split_xy(load_train())
 assert "SalePrice" not in X.columns and "Id" not in X.columns
 assert 9 < y.mean() < 14

def test_skewfixer_learns_on_fit_and_applies_in_transform():
 rng = np.random.default_rng(0)
 X = np.column_stack([rng.lognormal(0, 1.5, 500), rng.normal(5, 1, 500)])
 sf = SkewFixer().fit(X)
 assert list(sf.skewed_) == [0]
 assert sf.transform(X)[:, 1].tolist() == X[:, 1].tolist()
"""Stateless cleaning: fixes that come from understanding the data dictionary, not from statistics.
Because nothing is learned from the data here, the same function is safe for train and test."""
import pandas as pd

# In these columns a missing value does NOT mean "unknown": it means the house has no such feature.
# (The Kaggle files write it as "NA", which pandas reads as a missing value.)
NONE_CATEGORICAL = [
 "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
 "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
 "PoolQC", "Fence", "MiscFeature", "MasVnrType",
]
# Numeric columns where missing means "no basement / no garage / no veneer": the amount is zero.
ZERO_NUMERIC = [
 "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
 "BsmtFullBath", "BsmtHalfBath", "GarageCars", "GarageArea",
]
# Columns we drop: Utilities has almost no variation; GarageYrBlt repeats YearBuilt for most houses.
DROP_COLUMNS = ["Utilities", "GarageYrBlt"]

QUALITY = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
BSMT_FIN = {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}

# Ordered categories become numbers that keep their order. Unmapped values (true missing) stay NaN.
ORDINAL_MAPS = {
 "ExterQual": QUALITY, "ExterCond": QUALITY, "BsmtQual": QUALITY, "BsmtCond": QUALITY,
 "HeatingQC": QUALITY, "KitchenQual": QUALITY, "FireplaceQu": QUALITY,
 "GarageQual": QUALITY, "GarageCond": QUALITY, "PoolQC": QUALITY,
 "BsmtExposure": {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
 "BsmtFinType1": BSMT_FIN, "BsmtFinType2": BSMT_FIN,
 "GarageFinish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
 "Fence": {"None": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4},
 "LandSlope": {"Sev": 1, "Mod": 2, "Gtl": 3},
 "LotShape": {"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4},
 "PavedDrive": {"N": 0, "P": 1, "Y": 2},
 "CentralAir": {"N": 0, "Y": 1},
 "Functional": {"Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8},
}


def clean(df: pd.DataFrame) -> pd.DataFrame:
 df = df.copy()
 for col in NONE_CATEGORICAL:
    df[col] = df[col].astype("object").fillna("None")
 for col in ZERO_NUMERIC:
    df[col] = df[col].fillna(0)
 for col, mapping in ORDINAL_MAPS.items():
    df[col] = df[col].map(mapping)
 # These look like numbers but are really categories: the code 20 is not "less than" 60.
 df["MSSubClass"] = df["MSSubClass"].astype(str)
 df["MoSold"] = df["MoSold"].astype(str)
 return df.drop(columns=DROP_COLUMNS)
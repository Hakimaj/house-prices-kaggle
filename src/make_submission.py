"""Fit every model on ALL training data, blend them and write the Kaggle submission file.
Run from the project root: python -m src.make_submission"""
import json
import numpy as np
import pandas as pd
from .config import ID_COL, RANDOM_STATE, SUBMISSIONS_DIR
from .cv import blend_weights, cv_rmse, oof_predictions, rmse
from .data import load_test, load_train, remove_outliers, split_xy
from .pipeline import make_models

def check_submission(sub: pd.DataFrame, test: pd.DataFrame) -> None:
 """Cheap sanity checks that catch most silent mistakes before you spend a Kaggle submission."""
 assert list(sub.columns) == [ID_COL, "SalePrice"], "wrong columns"
 assert len(sub) == len(test), f"expected {len(test)} rows, got {len(sub)}"
 assert sub[ID_COL].tolist() == test[ID_COL].tolist(), "Ids differ from test.csv"
 assert sub["SalePrice"].notna().all(), "NaN in predictions"
 assert (sub["SalePrice"] > 0).all(), "non-positive prices"
 
def main() -> None:
 train = remove_outliers(load_train())
 test = load_test()
 X, y = split_xy(train)
 X_test = test.drop(columns=ID_COL)
 models = make_models()
 
 print("Cross-validated RMSE (log scale, lower is better):")
 for name, model in models.items():
    mean, std = cv_rmse(model, X, y)
    print(f" {name:<11} {mean:.4f} +/- {std:.4f}")
    
 oof = oof_predictions(models, X, y)
 weights = blend_weights(oof, y)
 print("Blend weights:", weights.round(3).to_dict())
 print(f"Blend CV RMSE: {rmse(y, oof.values @ weights.values):.4f}")
 
 preds = {}
 for name, model in models.items():
    preds[name] = model.fit(X, y).predict(X_test) # final fit on ALL rows
 blend_log = sum(weights[name] * preds[name] for name in models)
 sub = pd.DataFrame({ID_COL: test[ID_COL], "SalePrice": np.expm1(blend_log)}) # back to dollars
 check_submission(sub, test)
 
 SUBMISSIONS_DIR.mkdir(exist_ok=True)
 path = SUBMISSIONS_DIR / "submission_blend.csv"
 sub.to_csv(path, index=False)
 (SUBMISSIONS_DIR / "blend_weights.json").write_text(json.dumps(weights.round(4).to_dict(), indent=2))
 print(f"Wrote {path} (seed {RANDOM_STATE})")
 
 
if __name__ == "__main__":
    main()
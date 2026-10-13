"""Paths and constants in one place."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
SUBMISSIONS_DIR = ROOT / "submissions"

RANDOM_STATE = 42
TARGET = "SalePrice"
ID_COL = "Id"
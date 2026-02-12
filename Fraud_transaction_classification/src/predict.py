import argparse
import numpy as np
from joblib import load

from src.config import MODEL_PATH, THRESHOLD_PATH, REPORTS_DIR

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=float, required=True)
    parser.add_argument("--amount", type=float, required=True)
    # For this dataset, V1..V28 are required for real predictions.
    # This CLI is a placeholder to show inference packaging.
    args = parser.parse_args()

    model = load(MODEL_PATH)
    threshold = float((REPORTS_DIR / THRESHOLD_PATH.name).read_text().strip())

    print("⚠️ Note: Kaggle dataset uses V1..V28 features; without them, this is only a demo CLI.")
    print("Model loaded OK. Threshold:", threshold)

if __name__ == "__main__":
    main()

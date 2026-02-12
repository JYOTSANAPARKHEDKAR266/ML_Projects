import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from joblib import dump

from src.config import DATA_PATH, TARGET_COL, RANDOM_STATE, PREPROCESSOR_PATH, MODELS_DIR
from src.utils import ensure_dirs

def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)

def build_preprocessor():
    # Scale Time + Amount; V1-V28 are already PCA-like features
    numeric_to_scale = ["Time", "Amount"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("scale_time_amount", StandardScaler(), numeric_to_scale),
        ],
        remainder="passthrough"  # keep V1..V28
    )
    return preprocessor

def split(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

def fit_save_preprocessor(X_train: pd.DataFrame):
    ensure_dirs(MODELS_DIR)
    pre = build_preprocessor()
    pre.fit(X_train)
    dump(pre, PREPROCESSOR_PATH)
    return pre

def main():
    df = load_data()
    X_train, X_test, y_train, y_test = split(df)
    fit_save_preprocessor(X_train)
    print("✅ Preprocessor saved:", PREPROCESSOR_PATH)

if __name__ == "__main__":
    main()

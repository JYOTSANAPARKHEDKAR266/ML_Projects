import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from src.config import MODEL_PATH, PREPROCESSOR_PATH, MODELS_DIR, RANDOM_STATE, TARGET_COL
from src.preprocessing import load_data, split, fit_save_preprocessor
from src.utils import ensure_dirs, save_json

def pr_auc_cv(pipeline, X, y, cv):
    # cross-validated predicted probabilities (no leakage; SMOTE happens inside pipeline per fold)
    probas = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    return average_precision_score(y, probas)

def main():
    ensure_dirs(MODELS_DIR)

    df = load_data()
    X_train, X_test, y_train, y_test = split(df)

    # Fit preprocessor ONLY on training set
    pre = fit_save_preprocessor(X_train)

    # Pipelines
    baseline = Pipeline(steps=[
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    strong = Pipeline(steps=[
        ("pre", pre),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", RandomForestClassifier(
            n_estimators=400,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight=None  # SMOTE handles imbalance
        ))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    baseline_score = pr_auc_cv(baseline, X_train, y_train, cv)
    strong_score = pr_auc_cv(strong, X_train, y_train, cv)

    best_name, best_pipe, best_score = ("baseline_logreg", baseline, baseline_score)
    if strong_score > baseline_score:
        best_name, best_pipe, best_score = ("smote_random_forest", strong, strong_score)

    # Fit best model on full training data
    best_pipe.fit(X_train, y_train)
    dump(best_pipe, MODEL_PATH)

    info = {
        "selected_model": best_name,
        "cv_pr_auc_baseline_logreg": float(baseline_score),
        "cv_pr_auc_smote_random_forest": float(strong_score),
        "best_cv_pr_auc": float(best_score),
    }
    save_json(MODELS_DIR / "training_summary.json", info)

    print("✅ Selected:", best_name)
    print("✅ Saved model:", MODEL_PATH)

if __name__ == "__main__":
    main()

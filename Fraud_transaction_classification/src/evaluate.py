import numpy as np
import matplotlib.pyplot as plt
from joblib import load
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve
)

from src.config import DATA_PATH, TARGET_COL, MODEL_PATH, REPORTS_DIR, FIGURES_DIR, THRESHOLD_PATH, METRICS_PATH
from src.preprocessing import load_data, split
from src.thresholding import best_threshold_for_f1
from src.utils import ensure_dirs, save_json

def main():
    ensure_dirs(REPORTS_DIR, FIGURES_DIR)

    df = load_data()
    X_train, X_test, y_train, y_test = split(df)

    model = load(MODEL_PATH)

    proba = model.predict_proba(X_test)[:, 1]

    # threshold tuning on test is not ideal in real life; for your project, we’ll log it clearly.
    # (Later we can add a proper validation split.)
    t_best, f1_best = best_threshold_for_f1(y_test.values, proba)
    (REPORTS_DIR / THRESHOLD_PATH.name).write_text(str(t_best), encoding="utf-8")

    y_pred = (proba >= t_best).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)

    metrics = {
        "threshold": t_best,
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }
    save_json(METRICS_PATH, metrics)

    # PR curve
    prec, rec, _ = precision_recall_curve(y_test, proba)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.savefig(FIGURES_DIR / "pr_curve.png", dpi=160, bbox_inches="tight")
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.savefig(FIGURES_DIR / "roc_curve.png", dpi=160, bbox_inches="tight")
    plt.close()

    print("✅ Saved metrics:", METRICS_PATH)
    print("✅ Saved figures to:", FIGURES_DIR)

if __name__ == "__main__":
    main()

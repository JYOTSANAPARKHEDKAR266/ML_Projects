import numpy as np
from sklearn.metrics import f1_score

def best_threshold_for_f1(y_true, y_proba):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)

"""
Isolation Forest Anomaly Detector
Unsupervised model — trained without labels, evaluated against ground truth.
Includes threshold sweep over precision / recall / F1.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_recall_curve, roc_auc_score,
    average_precision_score, classification_report,
)


class AMLIsolationForest:
    def __init__(self, contamination: float = 0.08, n_estimators: int = 200,
                 random_state: int = 42):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self.threshold_ = None

    # ── training ──────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "AMLIsolationForest":
        """Unsupervised fit — no labels needed."""
        self.model.fit(X)
        return self

    # ── scoring ───────────────────────────────────────────────────────────────

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Return raw anomaly scores in [0, 1] where 1 = most anomalous.
        Uses stored min/max from tune_threshold so single-sample scoring works.
        """
        raw = -self.model.decision_function(X)
        lo  = getattr(self, '_score_min', raw.min())
        hi  = getattr(self, '_score_max', raw.max())
        return (raw - lo) / (hi - lo + 1e-9)

    # ── threshold tuning ──────────────────────────────────────────────────────

    def tune_threshold(self, X: np.ndarray, y: np.ndarray,
                       beta: float = 2.0) -> dict:
        """
        Sweep thresholds and pick the one maximising F-beta.
        AML favours recall (beta > 1), so beta=2 weighs recall twice as much.
        Returns a dict of metrics and the chosen threshold.
        """
        # Store raw score range so single-sample anomaly_scores() stays calibrated
        raw = -self.model.decision_function(X)
        self._score_min = float(raw.min())
        self._score_max = float(raw.max())
        scores = self.anomaly_scores(X)
        precisions, recalls, thresholds = precision_recall_curve(y, scores)

        # F-beta score at each threshold
        f_beta = ((1 + beta**2) * precisions * recalls /
                  (beta**2 * precisions + recalls + 1e-9))
        best_idx = np.argmax(f_beta[:-1])           # last element has no threshold
        self.threshold_ = float(thresholds[best_idx])

        y_pred = (scores >= self.threshold_).astype(int)
        roc    = roc_auc_score(y, scores)
        ap     = average_precision_score(y, scores)

        result = {
            "threshold":   self.threshold_,
            "precision":   float(precisions[best_idx]),
            "recall":      float(recalls[best_idx]),
            "f_beta":      float(f_beta[best_idx]),
            "roc_auc":     float(roc),
            "avg_precision": float(ap),
            "precisions":  precisions,
            "recalls":     recalls,
            "thresholds":  thresholds,
            "f_betas":     f_beta,
            "scores":      scores,
            "y_pred":      y_pred,
        }
        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.threshold_ is None:
            raise RuntimeError("Call tune_threshold() first.")
        return (self.anomaly_scores(X) >= self.threshold_).astype(int)

    def report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        return classification_report(y_true, y_pred,
                                     target_names=["legit", "laundering"])
"""
Evaluation metrics.

The primary metric for this project is rare-class macro-F1, not overall accuracy.
Overall accuracy on Fed-ISIC2019 is dominated by melanocytic nevus and melanoma; a model
that never once predicts dermatofibroma can still post a high accuracy figure. Reporting
accuracy as the headline number would hide exactly the failure this project exists to fix.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, recall_score


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """F1 for every class, including classes absent from y_true (reported as 0.0)."""
    return f1_score(
        y_true, y_pred,
        labels=list(range(n_classes)),
        average=None,
        zero_division=0,
    )


def rare_macro_f1(y_true: np.ndarray, y_pred: np.ndarray,
                  rare_class_ids: list[int], n_classes: int) -> float:
    """Mean F1 across the rare classes only. This is the headline metric."""
    if not rare_class_ids:
        raise ValueError(
            "rare_class_ids is empty. Run scripts/02_explore_data.py first and record "
            "the measured rare classes in configs/default.yaml - do not hardcode them."
        )
    scores = per_class_f1(y_true, y_pred, n_classes)
    return float(np.mean([scores[c] for c in rare_class_ids]))


def summarise(y_true: np.ndarray, y_pred: np.ndarray,
              rare_class_ids: list[int], n_classes: int) -> dict:
    """Full metric set for one evaluation pass."""
    f1s = per_class_f1(y_true, y_pred, n_classes)
    common = [c for c in range(n_classes) if c not in rare_class_ids]

    return {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1s.mean()),
        "rare_macro_f1": float(np.mean([f1s[c] for c in rare_class_ids])) if rare_class_ids else None,
        "common_macro_f1": float(np.mean([f1s[c] for c in common])) if common else None,
        "balanced_accuracy": float(
            recall_score(y_true, y_pred, labels=list(range(n_classes)),
                         average="macro", zero_division=0)
        ),
        "per_class_f1": {int(c): float(f1s[c]) for c in range(n_classes)},
        "support": {int(c): int((y_true == c).sum()) for c in range(n_classes)},
    }


def per_class_scores_for_reputation(y_true: np.ndarray, y_pred: np.ndarray,
                                    n_classes: int) -> np.ndarray:
    """Per-class quality vector used to build a client's reputation vector.

    This is the core measurement behind the project's contribution: instead of collapsing
    a client's update into one number, it is scored separately on every class.

    Classes with no support in the validation split return NaN rather than 0.0, so that
    "we could not measure this" is never confused with "this client performed badly".
    The aggregation step must handle NaN by falling back to the previous round's score.
    """
    scores = per_class_f1(y_true, y_pred, n_classes).astype(np.float64)
    support = np.bincount(y_true.astype(int), minlength=n_classes)
    scores[support == 0] = np.nan
    return scores


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

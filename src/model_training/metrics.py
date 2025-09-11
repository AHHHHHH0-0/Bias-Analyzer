"""
Metrics computation functions for model training and evaluation.
"""

from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """Compute comprehensive metrics for evaluation."""
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    
    # Basic metrics
    accuracy = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_micro = f1_score(labels, preds, average="micro")
    f1_weighted = f1_score(labels, preds, average="weighted")
    
    # Precision and recall
    precision, recall, _, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "f1_weighted": f1_weighted,
        "precision_macro": precision,
        "recall_macro": recall,
    }

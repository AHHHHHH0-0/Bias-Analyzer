"""
Comprehensive metrics calculation system for model evaluation.
Provides consistent metrics across sentiment and bias classification tasks.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, roc_auc_score, matthews_corrcoef
)
from sklearn.preprocessing import LabelBinarizer
import torch
import torch.nn.functional as F


class ModelMetrics:
    """
    Unified metrics calculation for classification tasks.
    Supports both binary and multi-class classification.
    """
    
    def __init__(self, labels: List[str]):
        """
        Initialize with task-specific labels.
        
        Args:
            labels: List of class labels (e.g., ['negative', 'neutral', 'positive'])
        """
        self.labels = labels
        self.num_classes = len(labels)
        self.label_to_id = {label: i for i, label in enumerate(labels)}
        self.id_to_label = {i: label for i, label in enumerate(labels)}
    
    def calculate_comprehensive_metrics(
        self, 
        y_true: Union[List, np.ndarray], 
        y_pred: Union[List, np.ndarray],
        y_proba: Optional[np.ndarray] = None,
        prefix: str = ""
    ) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels (can be strings or integers)
            y_pred: Predicted labels (can be strings or integers)
            y_proba: Prediction probabilities (optional, for additional metrics)
            prefix: Prefix for metric names (e.g., "train_", "test_")
            
        Returns:
            Dictionary of metric names and values
        """
        # Convert string labels to integers if necessary
        if isinstance(y_true[0], str):
            y_true = [self.label_to_id[label] for label in y_true]
        if isinstance(y_pred[0], str):
            y_pred = [self.label_to_id[label] for label in y_pred]
            
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        metrics = {}
        
        # Basic accuracy
        metrics[f"{prefix}accuracy"] = accuracy_score(y_true, y_pred)
        
        # Precision, Recall, F1-score (macro, micro, weighted)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=range(self.num_classes), zero_division=0
        )
        
        # Per-class metrics
        for i, label in enumerate(self.labels):
            metrics[f"{prefix}precision_{label}"] = precision[i]
            metrics[f"{prefix}recall_{label}"] = recall[i]
            metrics[f"{prefix}f1_{label}"] = f1[i]
            metrics[f"{prefix}support_{label}"] = support[i]
        
        # Averaged metrics
        for avg_type in ['macro', 'micro', 'weighted']:
            p, r, f, _ = precision_recall_fscore_support(
                y_true, y_pred, average=avg_type, zero_division=0
            )
            metrics[f"{prefix}precision_{avg_type}"] = p
            metrics[f"{prefix}recall_{avg_type}"] = r
            metrics[f"{prefix}f1_{avg_type}"] = f
        
        # Matthews Correlation Coefficient
        metrics[f"{prefix}matthews_corrcoef"] = matthews_corrcoef(y_true, y_pred)
        
        # Confusion matrix (flattened)
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                metrics[f"{prefix}cm_{self.labels[i]}_to_{self.labels[j]}"] = cm[i, j]
        
        # ROC-AUC (if probabilities provided and multi-class)
        if y_proba is not None:
            try:
                if self.num_classes == 2:
                    metrics[f"{prefix}roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    # Multi-class ROC-AUC (one-vs-rest)
                    lb = LabelBinarizer()
                    y_true_binary = lb.fit_transform(y_true)
                    if y_true_binary.shape[1] == 1:  # Binary case
                        y_true_binary = np.column_stack([1 - y_true_binary, y_true_binary])
                    
                    auc_scores = []
                    for i in range(self.num_classes):
                        try:
                            auc = roc_auc_score(y_true_binary[:, i], y_proba[:, i])
                            auc_scores.append(auc)
                            metrics[f"{prefix}roc_auc_{self.labels[i]}"] = auc
                        except ValueError:
                            # Handle case where class is not present in y_true
                            metrics[f"{prefix}roc_auc_{self.labels[i]}"] = 0.0
                    
                    if auc_scores:
                        metrics[f"{prefix}roc_auc_macro"] = np.mean(auc_scores)
            except (ValueError, TypeError):
                # Handle cases where ROC-AUC cannot be calculated
                pass
        
        # Class distribution
        unique, counts = np.unique(y_true, return_counts=True)
        total = len(y_true)
        for i, label in enumerate(self.labels):
            if i in unique:
                idx = np.where(unique == i)[0][0]
                metrics[f"{prefix}distribution_{label}"] = counts[idx] / total
            else:
                metrics[f"{prefix}distribution_{label}"] = 0.0
        
        return metrics
    
    def calculate_prediction_confidence(self, y_proba: np.ndarray, prefix: str = "") -> Dict[str, float]:
        """
        Calculate prediction confidence metrics.
        
        Args:
            y_proba: Prediction probabilities [n_samples, n_classes]
            prefix: Prefix for metric names
            
        Returns:
            Dictionary of confidence metrics
        """
        metrics = {}
        
        # Maximum probability (confidence)
        max_probs = np.max(y_proba, axis=1)
        metrics[f"{prefix}confidence_mean"] = np.mean(max_probs)
        metrics[f"{prefix}confidence_std"] = np.std(max_probs)
        metrics[f"{prefix}confidence_min"] = np.min(max_probs)
        metrics[f"{prefix}confidence_max"] = np.max(max_probs)
        
        # Entropy (uncertainty)
        epsilon = 1e-8  # Prevent log(0)
        entropy = -np.sum(y_proba * np.log(y_proba + epsilon), axis=1)
        metrics[f"{prefix}entropy_mean"] = np.mean(entropy)
        metrics[f"{prefix}entropy_std"] = np.std(entropy)
        
        # Prediction margin (difference between top 2 predictions)
        sorted_probs = np.sort(y_proba, axis=1)
        margins = sorted_probs[:, -1] - sorted_probs[:, -2]
        metrics[f"{prefix}margin_mean"] = np.mean(margins)
        metrics[f"{prefix}margin_std"] = np.std(margins)
        
        return metrics
    
    def get_confusion_matrix_df(self, y_true: Union[List, np.ndarray], y_pred: Union[List, np.ndarray]) -> pd.DataFrame:
        """
        Get confusion matrix as a pandas DataFrame with label names.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Confusion matrix DataFrame
        """
        # Convert string labels to integers if necessary
        if isinstance(y_true[0], str):
            y_true = [self.label_to_id[label] for label in y_true]
        if isinstance(y_pred[0], str):
            y_pred = [self.label_to_id[label] for label in y_pred]
            
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
        return pd.DataFrame(cm, index=self.labels, columns=self.labels)

def get_task_metrics_calculator(task: str) -> ModelMetrics:
    """
    Factory function to get appropriate metrics calculator for a task.
    
    Args:
        task: Task name ('sentiment' or 'bias')
        
    Returns:
        ModelMetrics instance configured for the task
    """
    task_labels = {
        "sentiment": ["negative", "neutral", "positive"],
        "bias": ["left", "center", "right"]
    }
    
    if task not in task_labels:
        raise ValueError(f"Unknown task: {task}. Supported tasks: {list(task_labels.keys())}")
    
    return ModelMetrics(task_labels[task])


def calculate_model_size_metrics(model) -> Dict[str, Union[int, float]]:
    """
    Calculate model size and parameter metrics.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary of model size metrics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
    }


def calculate_inference_metrics(model, dataloader, device: str = "cpu") -> Dict[str, float]:
    """
    Calculate inference speed and memory metrics.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to run inference on
        
    Returns:
        Dictionary of inference metrics
    """
    import time
    import psutil
    import os
    
    model.eval()
    model.to(device)
    
    # Warm up
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 3:  # Warm up with 3 batches
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            _ = model(**batch)
    
    # Measure inference time
    start_time = time.time()
    total_samples = 0
    
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            _ = model(**batch)
            total_samples += batch['input_ids'].size(0)
    
    end_time = time.time()
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    
    total_time = end_time - start_time
    
    return {
        "inference_time_total": total_time,
        "inference_time_per_sample": total_time / total_samples,
        "samples_per_second": total_samples / total_time,
        "memory_usage_mb": memory_after - memory_before,
        "peak_memory_mb": memory_after
    }

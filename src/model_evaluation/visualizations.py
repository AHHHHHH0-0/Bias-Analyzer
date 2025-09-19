"""
Simple visualization system for 12-model comparison analysis.
Generates focused, clean charts for model evaluation results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict
from pathlib import Path
import os


def create_all_visualizations(
    results_df: pd.DataFrame,
    confusion_matrices: Dict,
    output_dir: str = "results/visualizations"
) -> None:
    """
    Create all visualizations for comprehensive 12-model comparison analysis.
    
    Args:
        results_df: Results DataFrame with all model evaluations
        confusion_matrices: Confusion matrices for each evaluation
        output_dir: Output directory for visualizations
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create confusion matrices subfolder
    cm_dir = output_path / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating visualizations...")
    
    # 1. Create 12 confusion matrices (one file each)
    print("Creating confusion matrices...")
    create_confusion_matrices(confusion_matrices, str(cm_dir))
    
    # 2. Create performance comparison bar chart
    print("Creating performance comparison chart...")
    create_performance_comparison(results_df, str(output_path))
    
    # 3. Create size and inference speed comparison chart
    print("Creating size and inference speed comparison chart...")
    create_size_and_speed_comparison(results_df, str(output_path))
    
    print(f"✓ All visualizations saved to: {output_dir}\n")


def create_confusion_matrices(confusion_matrices: Dict, output_dir: str) -> None:
    """
    Create individual confusion matrix files for each model.
    Dynamically generates one confusion matrix per model based on the provided data.
        
    Args:
        confusion_matrices: Dictionary with confusion matrices (key format: "arch_layer_mode_task_phase")
        output_dir: Output directory for confusion matrix files
    """
    print(f"Creating {len(confusion_matrices)} confusion matrices...")
    
    for key, cm in confusion_matrices.items():
        # Extract model identifier from key 
        # Key format: "roberta_1layer_ft_part_bias_trained_model" -> "roberta_1layer_ft_part"
        parts = key.split('_')
        if len(parts) >= 3:
            # Take first 3 parts as the model identifier (arch_layer_mode)
            model_id = '_'.join(parts[:3])
        else:
            # Fallback to the key itself if format is unexpected
            model_id = key
        
        # Create figure
        plt.figure(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            cbar_kws={'shrink': 0.8}
        )
        
        plt.title(f'Confusion Matrix: {model_id}', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        
        # Save the figure with the full model identifier to avoid collisions
        filename = f"{model_id}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()


def create_performance_comparison(results_df: pd.DataFrame, output_dir: str) -> None:
    """
    Create single plot with grouped bars for accuracy, f1, precision, recall across all 12 models.
        
        Args:
        results_df: Results DataFrame with model evaluations
        output_dir: Output directory for the chart
    """
    # Define metrics to plot
    metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
    metric_labels = ['Accuracy', 'F1-Macro', 'Precision-Macro', 'Recall-Macro']
    
    # Filter available metrics
    available_metrics = [m for m in metrics if m in results_df.columns]
    available_labels = [metric_labels[i] for i, m in enumerate(metrics) if m in results_df.columns]
    
    if not available_metrics:
        print("No performance metrics found in results")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Set up bar positions
    n_models = len(results_df)
    n_metrics = len(available_metrics)
    x_pos = np.arange(n_models)
    width = 0.2
            
    # Define colors for each metric
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Create grouped bars
    for i, (metric, label) in enumerate(zip(available_metrics, available_labels)):
        bars = ax.bar(
            x_pos + i * width, 
            results_df[metric], 
            width, 
            label=label,
            color=colors[i],
            alpha=0.8
        )
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2., 
                height + 0.01,
                f'{height:.3f}', 
                ha='center', 
                va='bottom', 
                fontsize=8,
                fontweight='bold'
            )
    
    # Customize the plot
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Performance Comparison Across All Models', fontsize=16, fontweight='bold')
    ax.set_xticks(x_pos + width * (n_metrics - 1) / 2)
    ax.set_xticklabels(results_df.index, rotation=45, ha='right')
    ax.legend(loc='upper left')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
        
    plt.tight_layout()
        
    # Save the figure
    filename = "performance_comparison.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()


def create_size_and_speed_comparison(results_df: pd.DataFrame, output_dir: str) -> None:
    """
    Create combined chart with model size and inference speed comparison.
        
        Args:
        results_df: Results DataFrame with model evaluations
        output_dir: Output directory for the chart
    """
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Model Size Comparison
    if 'total_parameters' in results_df.columns:
        # Convert to millions of parameters
        params_millions = results_df['total_parameters'] / 1e6
        
        bars1 = ax1.bar(
            range(len(results_df)), 
            params_millions,
            color='skyblue',
            alpha=0.8
        )
        
        # Add value labels on bars
        for bar, value in zip(bars1, params_millions):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width()/2., 
                height + 0.5,
                f'{value:.1f}M', 
                ha='center', 
                va='bottom', 
                fontsize=8,
                fontweight='bold'
            )
        
        ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Parameters (Millions)', fontsize=12, fontweight='bold')
        ax1.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(results_df)))
        ax1.set_xticklabels(results_df.index, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
    else:
        ax1.text(0.5, 0.5, 'Model size data not available', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
    
    # Inference Speed Comparison
    if 'inference_time_per_sample' in results_df.columns:
        # Convert to milliseconds
        time_ms = results_df['inference_time_per_sample'] * 1000
        
        bars2 = ax2.bar(
            range(len(results_df)), 
            time_ms,
            color='lightcoral',
            alpha=0.8
        )
        
        # Add value labels on bars
        for bar, value in zip(bars2, time_ms):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width()/2., 
                height + 0.1,
                f'{value:.2f}ms', 
                ha='center', 
                va='bottom', 
                fontsize=8,
                fontweight='bold'
            )
        
        ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Inference Time per Sample (ms)', fontsize=12, fontweight='bold')
        ax2.set_title('Inference Speed Comparison', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(results_df)))
        ax2.set_xticklabels(results_df.index, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
    else:
        ax2.text(0.5, 0.5, 'Inference speed data not available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Inference Speed Comparison', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the figure
    filename = "Size and Inference Speed Comparison.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

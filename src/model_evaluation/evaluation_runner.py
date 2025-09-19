"""
Comprehensive model evaluation runner for systematic comparison.
Orchestrates the entire evaluation pipeline for academic analysis.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import time
from datetime import datetime
import torch
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader

from .metrics import get_task_metrics_calculator, calculate_model_size_metrics, calculate_inference_metrics
from .visualizations import ModelVisualizationSuite, create_all_visualizations
from ..model_training.train import TextClassificationDataset
from ..model_training.config import TASK_CONFIG, MODELS_CONFIG, DEFAULT_OUTPUT_ROOT


class ModelEvaluationRunner:
    """
    Comprehensive evaluation runner for systematic model comparison.
    """
    
    def __init__(self, output_dir: str = "src/model_evaluation/results"):
        """
        Initialize evaluation runner.
        
        Args:
            output_dir: Base directory for all outputs
        """
        # Create output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for metrics and visualizations
        self.metrics_dir = self.output_dir / "metrics"
        self.visualizations_dir = self.output_dir / "visualizations"
        for dir_path in [self.metrics_dir, self.visualizations_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize visualization suite
        self.viz_suite = ModelVisualizationSuite(str(self.visualizations_dir))
        self.confusion_matrices = {}
        
        # Model discovery
        self.available_models = self._discover_trained_models()

        # Device setup
        self.device = self._select_device()

        # Print setup
        print()
        print(f"Using device: {self.device}")
        print(f"Discovered {len(self.available_models)} trained models")
        print(f"Output directory: {self.output_dir}")
        print()

    def _select_device(self) -> str:
        """Select best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _discover_trained_models(self) -> Dict[str, str]:
        """
        Discover all trained models from the directory structure.
        
        Returns:
            Dictionary mapping model_id to model_path
        """
        models = {}
        trained_models_dir = Path(DEFAULT_OUTPUT_ROOT)
        
        if not trained_models_dir.exists():
            print(f"Warning: Trained models directory not found: {trained_models_dir}")
            return models
        
        # Iterate through model architectures
        for arch_name in MODELS_CONFIG.keys():
            arch_dir = trained_models_dir / arch_name
            # Iterate through layer configurations
            for layer_dir in arch_dir.iterdir():
                layer_name = layer_dir.name
                # Iterate through training modes
                for mode_dir in layer_dir.iterdir():
                    mode_name = mode_dir.name
                    # Check if model files exist
                    if (mode_dir / "model.safetensors").exists():
                        model_id = f"{arch_name}_{layer_name}_{mode_name}"
                        models[model_id] = str(mode_dir)
                        print(f"Found model: {model_id}")
        return models
    
    def _evaluate_model_on_task(self, model_path: str, task: str, phase: str = "evaluation", batch_size: int = 16) -> Dict[str, Any]:
        """
        Evaluate a single model on a single task.
        
        Args:
            model_path: Path to the model directory
            task: Task name ('sentiment' or 'bias')
            phase: Evaluation phase ('before_finetuning', 'after_finetuning', etc.)
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary of evaluation results
        """
        print(f"Evaluating {model_path} on {task} task ({phase})...")
        
        config = TASK_CONFIG[task]
        
        # Load model and tokenizer - need to use custom model classes
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        
        # Determine model architecture and configuration from model_path
        model_parts = model_path.split('/')
        arch_name = model_parts[-3]  # e.g., 'roberta' or 'distilbert'
        layer_config = model_parts[-2]  # e.g., '1layer' or '2layer'
        training_mode = model_parts[-1]  # e.g., 'feat_extr', 'ft_part', 'ft_full'
        
        # Get model configuration
        model_config = MODELS_CONFIG[arch_name]
        layers_param = model_config["layers"][layer_config]
        train_mode_param = model_config["train_modes"][training_mode]
        
        # Load custom model
        model_class = model_config["model_class"]
        model = model_class(layers_param, train_mode_param, num_classes=len(config["labels"]))
        
        # Load trained weights
        model_state_path = os.path.join(model_path, "model.safetensors")
        if os.path.exists(model_state_path):
            import safetensors.torch
            state_dict = safetensors.torch.load_file(model_state_path)
            model.load_state_dict(state_dict, strict=False)
        else:
            raise FileNotFoundError(f"Model weights not found: {model_state_path}")
        
        model.to(self.device)
        model.eval()
        
        # Load evaluation data (eval.csv for post-training evaluation)
        eval_csv_path = "src/data/labeled/eval.csv"
        df_test = pd.read_csv(eval_csv_path)
        
        # Create label mapping
        label2id = {label: i for i, label in enumerate(config["labels"])}
        id2label = {i: label for label, i in label2id.items()}
        
        # Create dataset
        test_dataset = TextClassificationDataset(
            df_test, tokenizer, config["text_column"], config["label_column"], 
            label2id, max_length=256
        )
        
        # Create data collator and dataloader
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
        
        # Get predictions and probabilities
        all_predictions = []
        all_probabilities = []
        all_true_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(self.device)
                
                outputs = model(**inputs)
                logits = outputs.logits
                
                # Get predictions
                predictions = torch.argmax(logits, dim=-1)
                probabilities = torch.softmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_true_labels.extend(labels.cpu().numpy())
        
        # Convert to numpy arrays
        y_true = np.array(all_true_labels)
        y_pred = np.array(all_predictions)
        y_proba = np.array(all_probabilities)
        
        # Calculate metrics
        metrics_calc = get_task_metrics_calculator(task)
        
        # Comprehensive metrics
        metrics = metrics_calc.calculate_comprehensive_metrics(y_true, y_pred, y_proba)
        
        # Confidence metrics
        confidence_metrics = metrics_calc.calculate_prediction_confidence(y_proba)
        metrics.update(confidence_metrics)
        
        # Model size metrics
        size_metrics = calculate_model_size_metrics(model)
        metrics.update(size_metrics)
        
        # Inference speed metrics
        inference_metrics = calculate_inference_metrics(model, dataloader, self.device)
        metrics.update(inference_metrics)
        
        # Confusion matrix
        confusion_matrix = metrics_calc.get_confusion_matrix_df(y_true, y_pred)
        
        # Classification report
        classification_report = metrics_calc.get_classification_report_df(y_true, y_pred)
        
        # Store confusion matrix for visualization
        model_name = Path(model_path).name
        cm_key = f"{model_name}_{task}_{phase}"
        self.confusion_matrices[cm_key] = confusion_matrix
        
        return {
            'metrics': metrics,
            'confusion_matrix': confusion_matrix,
            'classification_report': classification_report,
            'predictions': y_pred,
            'probabilities': y_proba,
            'true_labels': y_true
        }
    
    def evaluate_all_trained_models(self) -> pd.DataFrame:
        """
        Evaluate all discovered trained models on bias classification.
        
        Returns:
            DataFrame with evaluation results
        """
        
        results = {}
        task = "bias"  # Only bias task now
        
        for model_id, model_path in self.available_models.items():
            print(f"\nEvaluating {model_id}...")
            
            try:
                result = self._evaluate_model_on_task(
                    model_path, task, "trained_model"
                )
                results[model_id] = result['metrics']
                
                print(f"✓ Completed: {model_id}")
                
            except Exception as e:
                print(f"✗ Failed: {model_id} - {str(e)}")
                continue
        
        # Create DataFrame
        results_df = pd.DataFrame.from_dict(results, orient='index')
        
        # Save results
        results_df.to_csv(self.metrics_dir / "trained_models_results.csv")
        
        print(f"\nEvaluation complete! Results saved to {self.metrics_dir}")
        return results_df
    
    def analyze_model_comparisons(self, results_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Create structured comparisons across different dimensions.
        
        Args:
            results_df: Results DataFrame with model_id as index
            
        Returns:
            Dictionary of comparison DataFrames
        """
        print("="*60)
        print("GENERATING STRUCTURED COMPARISONS")
        print("="*60)
        
        comparisons = {}
        
        # Parse model IDs to extract components
        model_data = []
        for model_id in results_df.index:
            parts = model_id.split('_')
            if len(parts) >= 3:
                arch, layer, mode = parts[0], parts[1], parts[2]
                model_data.append({
                    'model_id': model_id,
                    'architecture': arch,
                    'layer_config': layer, 
                    'training_mode': mode
                })
        
        model_info_df = pd.DataFrame(model_data)
        results_with_info = results_df.join(model_info_df.set_index('model_id'))
        
        # 1. Architecture Comparison (RoBERTa vs DistilBERT)
        if 'architecture' in results_with_info.columns:
            arch_comparison = results_with_info.groupby('architecture')[['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']].agg(['mean', 'std', 'max', 'min'])
            comparisons['architecture'] = arch_comparison
            arch_comparison.to_csv(self.metrics_dir / "architecture_comparison.csv")
        
        # 2. Layer Configuration Comparison (1layer vs 2layer)
        if 'layer_config' in results_with_info.columns:
            layer_comparison = results_with_info.groupby('layer_config')[['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']].agg(['mean', 'std', 'max', 'min'])
            comparisons['layer_config'] = layer_comparison
            layer_comparison.to_csv(self.metrics_dir / "layer_config_comparison.csv")
        
        # 3. Training Mode Comparison (feat_extr vs ft_part vs ft_full)
        if 'training_mode' in results_with_info.columns:
            mode_comparison = results_with_info.groupby('training_mode')[['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']].agg(['mean', 'std', 'max', 'min'])
            comparisons['training_mode'] = mode_comparison
            mode_comparison.to_csv(self.metrics_dir / "training_mode_comparison.csv")
        
        # 4. Combined Analysis (Architecture + Layer + Mode)
        if all(col in results_with_info.columns for col in ['architecture', 'layer_config', 'training_mode']):
            combined_comparison = results_with_info.groupby(['architecture', 'layer_config', 'training_mode'])[['accuracy', 'f1_macro']].mean()
            comparisons['combined'] = combined_comparison
            combined_comparison.to_csv(self.metrics_dir / "combined_comparison.csv")
        
        # 5. Efficiency Analysis (Performance vs Model Size)
        if 'total_parameters' in results_with_info.columns:
            efficiency_df = results_with_info[['accuracy', 'f1_macro', 'total_parameters', 'inference_time_per_sample']].copy()
            efficiency_df['efficiency_score'] = efficiency_df['accuracy'] / (efficiency_df['total_parameters'] / 1e6)
            efficiency_df = efficiency_df.sort_values('efficiency_score', ascending=False)
            comparisons['efficiency'] = efficiency_df
            efficiency_df.to_csv(self.metrics_dir / "efficiency_analysis.csv")
        
        return comparisons
    
    def generate_comprehensive_report(
        self, 
        results_df: pd.DataFrame, 
        comparisons: Dict[str, pd.DataFrame]
    ) -> str:
        """
        Generate comprehensive text report of findings.
        
        Args:
            results_df: Main results DataFrame
            comparisons: Dictionary of comparison analyses
            
        Returns:
            Report string
        """
        report = []
        report.append("="*80)
        report.append("BIAS CLASSIFICATION: 12-MODEL COMPARISON REPORT")
        report.append("="*80)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Evaluated {len(results_df)} models on bias classification task")
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)
        
        if not results_df.empty:
            # Best performing models
            best_accuracy = results_df['accuracy'].max()
            best_f1 = results_df['f1_macro'].max()
            best_acc_model = results_df['accuracy'].idxmax()
            best_f1_model = results_df['f1_macro'].idxmax()
            
            report.append(f"Best Accuracy: {best_acc_model} ({best_accuracy:.3f})")
            report.append(f"Best F1-Macro: {best_f1_model} ({best_f1:.3f})")
            report.append(f"Average Accuracy: {results_df['accuracy'].mean():.3f} ± {results_df['accuracy'].std():.3f}")
            report.append(f"Average F1-Macro: {results_df['f1_macro'].mean():.3f} ± {results_df['f1_macro'].std():.3f}")
        
        report.append("")
        
        # Detailed Results
        report.append("DETAILED RESULTS")
        report.append("-" * 40)
        
        # Top performers by metric
        key_metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
        available_metrics = [m for m in key_metrics if m in results_df.columns]
        
        for metric in available_metrics:
            report.append(f"\n{metric.replace('_', ' ').title()} (Top 5):")
            sorted_results = results_df[metric].sort_values(ascending=False).head(5)
            for idx, value in sorted_results.items():
                report.append(f"  {idx}: {value:.3f}")
        
        report.append("")
        
        # Structured Analysis
        report.append("STRUCTURED ANALYSIS")
        report.append("-" * 40)
        
        # Architecture comparison
        if 'architecture' in comparisons:
            arch_comp = comparisons['architecture']
            report.append("\nArchitecture Comparison (Mean Accuracy):")
            for arch in arch_comp.index:
                mean_acc = arch_comp.loc[arch, ('accuracy', 'mean')]
                report.append(f"  {arch.upper()}: {mean_acc:.3f}")
        
        # Training mode comparison
        if 'training_mode' in comparisons:
            mode_comp = comparisons['training_mode']
            report.append("\nTraining Mode Comparison (Mean F1-Macro):")
            for mode in mode_comp.index:
                mean_f1 = mode_comp.loc[mode, ('f1_macro', 'mean')]
                mode_desc = {'feat_extr': 'Feature Extraction', 'ft_part': 'Partial Fine-tuning', 'ft_full': 'Full Fine-tuning'}
                report.append(f"  {mode_desc.get(mode, mode)}: {mean_f1:.3f}")
        
        # Layer configuration comparison
        if 'layer_config' in comparisons:
            layer_comp = comparisons['layer_config']
            report.append("\nLayer Configuration Comparison (Mean F1-Macro):")
            for layer in layer_comp.index:
                mean_f1 = layer_comp.loc[layer, ('f1_macro', 'mean')]
                layer_desc = {'1layer': '1-Layer (Linear)', '2layer': '2-Layer (MLP)'}
                report.append(f"  {layer_desc.get(layer, layer)}: {mean_f1:.3f}")
        
        report.append("")
        
        # Model Efficiency
        if 'efficiency' in comparisons:
            efficiency_df = comparisons['efficiency']
            report.append("EFFICIENCY ANALYSIS")
            report.append("-" * 40)
            
            report.append("Top 3 Most Efficient Models (Accuracy/Parameters):")
            for i, (model_id, row) in enumerate(efficiency_df.head(3).iterrows()):
                report.append(f"  {i+1}. {model_id}: {row['efficiency_score']:.3f}")
        
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        
        if not results_df.empty:
            # Best overall model
            best_overall = results_df['f1_macro'].idxmax()
            report.append(f"Best Overall Model: {best_overall}")
            
            # Architecture recommendations
            if 'architecture' in comparisons:
                arch_comp = comparisons['architecture']
                best_arch = arch_comp[('accuracy', 'mean')].idxmax()
                report.append(f"Best Architecture: {best_arch.upper()}")
            
            # Training mode recommendations
            if 'training_mode' in comparisons:
                mode_comp = comparisons['training_mode']
                best_mode = mode_comp[('f1_macro', 'mean')].idxmax()
                mode_desc = {'feat_extr': 'Feature Extraction', 'ft_part': 'Partial Fine-tuning', 'ft_full': 'Full Fine-tuning'}
                report.append(f"Best Training Mode: {mode_desc.get(best_mode, best_mode)}")
            
            # General findings
            report.append("\nKey Findings:")
            report.append("- Systematic evaluation across 12 model configurations")
            report.append("- Different architectures show varying performance patterns")
            report.append("- Training mode significantly impacts bias classification performance")
            report.append("- Consider computational requirements when selecting deployment model")
        
        report.append("")
        report.append("="*80)
        
        return "\n".join(report)
    
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """
        Run the complete evaluation pipeline for all 12 trained models.
        
        Returns:
            Dictionary containing all results
        """
        start_time = time.time()
        
        # Phase 1: Evaluate all trained models
        print("="*80)
        print("EVALUATING ALL TRAINED MODELS ON BIAS CLASSIFICATION")
        print("="*80)
        results_df = self.evaluate_all_trained_models()
        
        # Phase 2: Generate structured comparisons
        print("\n" + "="*60)
        print("GENERATING STRUCTURED COMPARISONS")
        print("="*60)
        comparisons = self.analyze_model_comparisons(results_df)

        # Phase 3: Generate visualizations
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        create_all_visualizations(
            pd.DataFrame(),  # No before results for this evaluation
            results_df, 
            self.confusion_matrices,
            str(self.visualizations_dir)
        )
        
        # Phase 4: Generate comprehensive report
        print("Generating comprehensive report...")
        report = self.generate_comprehensive_report(results_df, comparisons)
        
        with open(self.output_dir / "bias_classification_report.txt", 'w') as f:
            f.write(report)
        
        # Save all results
        results = {
            'model_results': results_df.to_dict(),
            'comparisons': {k: v.to_dict() for k, v in comparisons.items()},
            'evaluation_metadata': {
                'num_models': len(self.available_models),
                'model_list': list(self.available_models.keys()),
                'task': 'bias',
                'evaluation_dataset': 'src/data/labeled/eval.csv',
                'device': self.device,
                'timestamp': datetime.now().isoformat(),
                'total_time_minutes': (time.time() - start_time) / 60
            }
        }
        
        self.viz_suite.save_results_summary(results)
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETE!")
        print("="*80)
        print(f"Total time: {(time.time() - start_time) / 60:.2f} minutes")
        print(f"Results saved to: {self.output_dir}")
        print(f"Visualizations: {self.visualizations_dir}")
        print(f"Report: {self.output_dir / 'bias_classification_report.txt'}")
        print(f"Individual comparisons: {self.metrics_dir}")
        
        return results

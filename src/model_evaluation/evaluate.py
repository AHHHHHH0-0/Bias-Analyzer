#!/usr/bin/env python3
"""
Unified evaluation script for 12-model bias classification comparison.
Uses eval.csv and generates comprehensive comparisons, visualizations, and reports.
"""

import sys
from pathlib import Path

# Project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.model_evaluation.evaluation_runner import ModelEvaluationRunner


def main():
    """
    Run unified evaluation of all 12 trained models on bias classification:
    """
    
    print("="*80)
    print("UNIFIED EVALUATION SCRIPT: 12 Models Bias Classification Comparison")
    print("="*80)
    
    # Initialize evaluation runner
    runner = ModelEvaluationRunner()
    
    # If no models found
    if not runner.available_models:
        print("❌ ERROR: No Models found!")
        return
    
    # Run comprehensive evaluation
    try:
        results = runner.run_comprehensive_evaluation()
        
        print("="*80)
        print("EVALUATION COMPLETE!")
        print("="*80)
        
        metadata = results.get('evaluation_metadata', {})
        print(f"Models evaluated: {metadata.get('num_models', 0)}")
        print(f"Evaluation time: {metadata.get('total_time_minutes', 0):.2f} minutes")
        print(f"Results directory: {runner.output_dir}")
        
        print("\nGenerated files:")
        print(f"  📊 Metrics: {runner.metrics_dir}/")
        print(f"  🎨 Visualizations: {runner.visualizations_dir}/")
        
        # Display best models
        print("\nKey Findings:")
        
        model_results = results['model_results']
        accuracies = {}
        f1_scores = {}
        
        for model_name, metrics in model_results.items():
            if isinstance(metrics, dict):
                accuracies[model_name] = metrics.get('accuracy', 0)
                f1_scores[model_name] = metrics.get('f1_macro', 0)
        
        if accuracies:
            best_acc_model = max(accuracies, key=accuracies.get)
            best_f1_model = max(f1_scores, key=f1_scores.get)
            
            print(f"  🏆 Best accuracy: {best_acc_model} ({accuracies[best_acc_model]:.3f})")
            print(f"  🏆 Best F1-macro: {best_f1_model} ({f1_scores[best_f1_model]:.3f})")
        
        print(f"\n✅ Evaluation completed successfully!\n")
    
    except Exception as e:
        print(f"\n❌ Evaluation failed: {str(e)}")


if __name__ == "__main__":
    main()



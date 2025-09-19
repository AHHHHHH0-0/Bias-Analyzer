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
    1. Evaluate each of 12 models on eval.csv dataset
    2. Generate structured comparisons across:
       - Architecture (RoBERTa vs DistilBERT)
       - Layer configuration (1layer vs 2layer)
       - Training mode (feat_extr vs ft_part vs ft_full)
    3. Create comprehensive visualizations
    4. Generate detailed evaluation report
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
        
        if results:
            print("\n" + "="*80)
            print("EVALUATION SUMMARY")
            print("="*80)
            
            metadata = results.get('evaluation_metadata', {})
            print(f"Models evaluated: {metadata.get('num_models', 0)}")
            print(f"Evaluation time: {metadata.get('total_time_minutes', 0):.2f} minutes")
            print(f"Results directory: {runner.output_dir}")
            
            print("\nGenerated files:")
            print(f"  📊 Main results: {runner.metrics_dir}/trained_models_results.csv")
            print(f"  📈 Comparisons: {runner.metrics_dir}/*_comparison.csv")
            print(f"  📋 Report: {runner.output_dir}/bias_classification_report.txt")
            print(f"  🎨 Visualizations: {runner.visualizations_dir}/")
            
            # Display key findings
            if 'model_results' in results:
                model_results = results['model_results']
                if model_results:
                    print("\nKey Findings:")
                    
                    # Find best models
                    accuracies = {k: v.get('accuracy', 0) for k, v in model_results.items()}
                    f1_scores = {k: v.get('f1_macro', 0) for k, v in model_results.items()}
                    
                    if accuracies:
                        best_acc_model = max(accuracies, key=accuracies.get)
                        best_f1_model = max(f1_scores, key=f1_scores.get)
                        
                        print(f"  🏆 Best accuracy: {best_acc_model} ({accuracies[best_acc_model]:.3f})")
                        print(f"  🏆 Best F1-macro: {best_f1_model} ({f1_scores[best_f1_model]:.3f})")
            
            print(f"\n✅ Evaluation completed successfully!")
            print(f"📁 All results saved to: {runner.output_dir}")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {str(e)}")


if __name__ == "__main__":
    main()



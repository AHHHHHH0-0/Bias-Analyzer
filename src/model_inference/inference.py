#!/usr/bin/env python3
"""
Political Bias Classification Inference Pipeline

This script runs inference on articles using trained bias classification models.
Configure the model and input settings in config.py before running.
"""

import os
import sys
import time

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.model_inference.config import MODEL_CONFIG, DATA_CONFIG, INFERENCE_CONFIG, OUTPUT_CONFIG
from src.model_inference.model_loader import ModelLoader
from src.model_inference.text_processor import InferenceTextProcessor
from src.model_inference.prediction_engine import PredictionEngine
from src.model_inference.output_formatter import OutputFormatter


def run_inference():
    """Main inference pipeline."""
    print("=" * 60)
    print("Political Bias Classification Inference")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # 1. Load model
        print(f"\nLoading model...")
        model_loader = ModelLoader(MODEL_CONFIG)
        model, tokenizer, device = model_loader.load_model()
        label_mapping = model_loader.get_label_mapping()
        
        # 2. Load input data
        print(f"\nLoading input data...")
        text_processor = InferenceTextProcessor(tokenizer, device, INFERENCE_CONFIG["max_length"])
        df = text_processor.load_data(DATA_CONFIG["input_file"])
        
        # 3. Run predictions
        print(f"\nRunning predictions...")
        prediction_engine = PredictionEngine(model, label_mapping, device)
        results = prediction_engine.predict_dataset(text_processor, df, INFERENCE_CONFIG["batch_size"])
        
        # 4. Generate results
        print("\n" + "="*80)
        print("✅ POLITICAL BIAS CLASSIFICATION RESULTS")
        print("="*80)
        summary = prediction_engine.get_prediction_summary(results)
        model_info = {
            "model_name": MODEL_CONFIG["model"],
            "layers": MODEL_CONFIG["layers"],
            "training": MODEL_CONFIG["training"],
            "device": str(device),
            "parameters": model_loader.metadata["model_info"]["num_parameters"],
            "training_config": model_loader.get_training_config()
        }
        total_time = time.time() - start_time

        output_formatter = OutputFormatter(OUTPUT_CONFIG)
        output_formatter.process_output(results, summary, model_info, total_time)

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")

def main():
    """Entry point for the script."""
    print("-" * 30)
    print("CONFIGURATION")
    print("-" * 30)
    print(f"Model: {MODEL_CONFIG['model']}")
    print(f"Layers: {MODEL_CONFIG['layers']}")
    print(f"Training: {MODEL_CONFIG['training']}")
    print(f"Input file: {DATA_CONFIG['input_file']}")
    print(f"Batch size: {INFERENCE_CONFIG['batch_size']}")
    print(f"Max length: {INFERENCE_CONFIG['max_length']}")
    print(f"Output file: {OUTPUT_CONFIG['output_file']}")
    print()
    
    # Run inference
    run_inference()

    print()
    print("="*80)
    print()
    print("✅ Done!")
    print()

if __name__ == "__main__":
    main()


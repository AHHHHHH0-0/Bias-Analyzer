import json
import os
from typing import Dict, Tuple, List
import time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

# Import configurations and dataset
from .config import TASK_CONFIG, MODELS_CONFIG, DEFAULT_OUTPUT_ROOT
from .dataset import TextClassificationDataset
from .metrics import compute_metrics


def train_single_model(
    model_name: str, 
    layer_type: str, 
    train_mode: str, 
    task: str = "bias",
    max_length: int = 256, 
    batch_size: int = 16, 
    epochs: int = 5, 
    lr: float = 2e-5
) -> None:
    """Train a single model configuration."""
    
    print(f"\n=== Training {model_name} - {layer_type} - {train_mode} ===")
    
    config = TASK_CONFIG[task]
    model_config = MODELS_CONFIG[model_name]
    
    # Load training dataset and split into train/validation
    df_full = pd.read_csv(config["train_csv"])
    
    # Split the training data into train/validation with shuffling
    # Use stratified split to maintain class balance
    df_train, df_test = train_test_split(
        df_full,
        test_size=0.2,  # 20% for validation
        random_state=42,  # For reproducibility
        shuffle=True,  # Explicitly shuffle the data
        stratify=df_full[config["label_column"]]  # Maintain class proportions
    )
    
    print(f"Dataset split completed:")
    print(f"  Training: {len(df_train)} examples")
    print(f"  Testing: {len(df_test)} examples")
    
    # Create label mapping
    label2id = {label: i for i, label in enumerate(config["labels"])}
    id2label = {i: label for label, i in label2id.items()}
    
    # Load tokenizer from preloaded encoders
    tokenizer = AutoTokenizer.from_pretrained(model_config["tokenizer_path"], use_fast=True)
    
    # Instantiate our custom model
    layers_param = model_config["layers"][layer_type]  # "linear" or "mlp"
    train_mode_param = model_config["train_modes"][train_mode]  # "head", "part", or "full"
    
    model = model_config["model_class"](
        layers=layers_param,
        train_mode=train_mode_param,
        num_classes=len(config["labels"])
    )
    
    # Create datasets
    train_ds = TextClassificationDataset(df_train, tokenizer, config["text_column"], config["label_column"], label2id, max_length)
    test_ds = TextClassificationDataset(df_test, tokenizer, config["text_column"], config["label_column"], label2id, max_length)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Output directory following the directory structure
    out_dir = os.path.join(DEFAULT_OUTPUT_ROOT, model_name, layer_type, train_mode)
    os.makedirs(out_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=lr,
        weight_decay=0.01,
        eval_strategy="epoch",  # Updated from evaluation_strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to=[],
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        logging_first_step=True,
        save_total_limit=2,  # Keep only best 2 checkpoints
        dataloader_pin_memory=False,  # Avoid memory issues
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,  # Use validation set for evaluation during training
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    print(f"Starting training for {model_name} - {layer_type} - {train_mode}...")
    trainer.train()
    
    # Evaluate on validation set one final time
    print("Running final evaluation on validation set...")
    final_metrics = trainer.evaluate(eval_dataset=test_ds)
    print(f"Final validation metrics: {final_metrics}")
    
    # Save final metrics
    final_metrics_file = os.path.join(out_dir, "final_metrics.json")
    with open(final_metrics_file, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    # Save model and tokenizer
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    
    # Save comprehensive metadata
    metadata = {
        "label2id": label2id, 
        "id2label": id2label, 
        "task": task, 
        "model_name": model_name,
        "layer_type": layer_type,
        "train_mode": train_mode,
        "model_config": {
            "layers_param": layers_param,
            "train_mode_param": train_mode_param,
            "num_classes": len(config["labels"])
        },
        "training_config": {
            "max_length": max_length,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": lr,
            "weight_decay": 0.01
        },
        "dataset_info": {
            "train_samples": len(df_train),
            "val_samples": len(df_test),
            "num_labels": len(config["labels"]),
            "labels": config["labels"]
        },
        "model_info": {
            "tokenizer_path": model_config["tokenizer_path"],
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "num_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
    }
    
    with open(os.path.join(out_dir, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved trained model to: {out_dir}")
    print(f"Final metrics: Accuracy={final_metrics.get('eval_accuracy', 0):.3f}, F1-Macro={final_metrics.get('eval_f1_macro', 0):.3f}")


def main() -> None:
    """Main training loop: train all 12 model combinations."""
    
    print("Starting comprehensive model training...")
    print("Training 12 model combinations: 2 models × 2 layer types × 3 training modes")
    
    # Generate all combinations
    total_jobs = 0
    combinations = []
    
    for model_name in MODELS_CONFIG.keys():
        for layer_type in MODELS_CONFIG[model_name]["layers"].keys():
            for train_mode in MODELS_CONFIG[model_name]["train_modes"].keys():
                combinations.append((model_name, layer_type, train_mode))
                total_jobs += 1
    
    print(f"Total jobs to run: {total_jobs}")
    print("\nModel combinations:")
    for i, (model_name, layer_type, train_mode) in enumerate(combinations, 1):
        print(f"  {i:2d}. {model_name:10s} - {layer_type:6s} - {train_mode:8s}")
    
    # Train all combinations
    current_job = 0
    successful_jobs = 0
    failed_jobs = []
    
    for model_name, layer_type, train_mode in combinations:
        current_job += 1
        print(f"\n{'='*80}")
        print(f"Job {current_job}/{total_jobs}: {model_name} - {layer_type} - {train_mode}")
        print(f"{'='*80}")
        
        try:
            train_single_model(model_name, layer_type, train_mode)
            successful_jobs += 1
            print(f"✅ SUCCESS: {model_name} - {layer_type} - {train_mode}")
        except Exception as e:
            print(f"❌ ERROR training {model_name} - {layer_type} - {train_mode}: {e}")
            failed_jobs.append((model_name, layer_type, train_mode, str(e)))
            continue
    
    # Final summary
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"Total jobs: {total_jobs}")
    print(f"Successful: {successful_jobs}")
    print(f"Failed: {len(failed_jobs)}")
    
    if failed_jobs:
        print(f"\nFailed jobs:")
        for model_name, layer_type, train_mode, error in failed_jobs:
            print(f"  - {model_name} - {layer_type} - {train_mode}: {error}")
    
    print(f"\nResults saved in: {DEFAULT_OUTPUT_ROOT}/")


if __name__ == "__main__":
    main()
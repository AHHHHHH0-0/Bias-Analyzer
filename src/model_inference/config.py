# SELECT MODEL, ARCHITECTURE, AND TRAINING
MODEL_CONFIG = {
    "model": "roberta",
    "layers": "2layer",
    "training": "ft_full"
}

# SELECT INPUT DATA
DATA_CONFIG = {
    "input_file": "src/data/processed/processed_articles.csv",
    "text_column": "full_text"
}

# Output configuration
OUTPUT_CONFIG = {
    "output_file": "src/model_inference/results/predictions.json",
    "show_in_terminal": True,
    "terminal_display_limit": 3
}

# Inference configuration
INFERENCE_CONFIG = {
    "max_length": 256,
    "batch_size": 16,
}

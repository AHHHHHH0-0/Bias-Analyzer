import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Support Apple Silicon if available
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_pretrained_model(model_name, save_path):
    """
    Load a tokenizer and sequence classification model and save to custom folder.
    """
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Save to custom folder name
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    print(f"\nSaved to {save_path}\n")
    
    return tokenizer, model


def main():
    # Configure which models to load with clean folder names
    models_config = {
        "cardiffnlp/twitter-roberta-base-sentiment-latest": "roberta",
        "distilbert-base-uncased": "distilbert"
    }

    device = _select_device()
    torch.set_grad_enabled(False)

    for model_id, folder_name in models_config.items():
        save_path = f"src/models/encoders/{folder_name}"
        load_pretrained_model(model_id, save_path)

    print(f"Loaded {len(models_config)} model(s) on device: {device}")


if __name__ == "__main__":
    main()

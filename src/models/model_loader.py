import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Support Apple Silicon if available
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_pretrained_model(model_name, cache_dir):
    """
    Load a tokenizer and sequence classification model for the given model name.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir)

    model.eval()

    return tokenizer, model


if __name__ == "__main__":
    # Configure which models to load. Edit this list as needed.
    model_names = [
        "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "distilbert-base-uncased",
    ]

    device = _select_device()
    torch.set_grad_enabled(False)

    loaded = {}
    for name in model_names:
        # Use the checkpoint's own classification head if available (e.g., CardiffNLP sentiment)
        tokenizer, model = load_pretrained_model(name, cache_dir="src/models/base")
        loaded[name] = (tokenizer, model)

    print(f"Loaded {len(loaded)} model(s) on device: {device}")

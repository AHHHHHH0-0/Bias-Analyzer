import os
import json
import torch
from transformers import AutoTokenizer
from safetensors.torch import load_file
from ..models.base.distilbert_bias_classifier import DistilbertBiasClassifier
from ..models.base.roberta_bias_classifier import RobertaBiasClassifier


class ModelLoader:
    """
    Loads trained bias classification models for inference.
    """
    
    def __init__(self, model_config):
        """
        Initialize the model loader.
        
        Args:
            model_config (dict): Configuration containing model, layers, and training mode
        """
        # Model config
        self.model_name = model_config["model"]
        self.layers = model_config["layers"]
        self.training = model_config["training"]
        self.device = self._select_device()
        # Build model path
        self.model_path = self._build_model_path()
        # Load metadata
        self.metadata = self._load_metadata()
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        
    def _select_device(self):
        """Select the best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    def _build_model_path(self):
        """Build the path to the trained model."""
        # Build path
        model_path = os.path.join("src", "models", "trained", self.model_name, self.layers, self.training)
        # Check path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")
        # Return path
        return model_path
    
    def _load_metadata(self):
        """Load training metadata for the model."""
        # Build path
        metadata_path = os.path.join(self.model_path, "training_metadata.json")
        # Check path
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found at: {metadata_path}")
        # Load metadata
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def load_model(self):
        """Load the trained model and tokenizer."""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Initialize model 
        model_config = self.metadata["model_config"]
        if self.model_name == "distilbert":
            self.model = DistilbertBiasClassifier(
                layers=model_config["layers_param"],
                train_mode=model_config["train_mode_param"],
                num_classes=model_config["num_classes"]
            )
        elif self.model_name == "roberta":
            self.model = RobertaBiasClassifier(
                layers=model_config["layers_param"],
                train_mode=model_config["train_mode_param"],
                num_classes=model_config["num_classes"]
            )
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        # Load trained weights
        model_weights_path = os.path.join(self.model_path, "model.safetensors")
        if os.path.exists(model_weights_path):
            state_dict = load_file(model_weights_path)
            self.model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"Model weights not found at: {model_weights_path}")
        
        # Move to device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded: {self.model_path}")
        print(f"  - Parameters: {self.metadata['model_info']['num_parameters']:,}")
        print(f"  - Labels: {self.metadata['dataset_info']['labels']}")
        
        return self.model, self.tokenizer, self.device
    
    def get_label_mapping(self):
        """Get the label mapping from the metadata."""
        return self.metadata["id2label"]
    
    def get_training_config(self):
        """Get the training configuration used for this model."""
        return self.metadata["training_config"]

from ..models.base.roberta_bias_classifier import RobertaBiasClassifier
from ..models.base.distilbert_bias_classifier import DistilbertBiasClassifier


DEFAULT_OUTPUT_ROOT = "src/models/trained"

# Fixed task configurations
TASK_CONFIG = {
    "bias": {
        "labels": ["left", "center", "right"],
        "train_csv": "src/data/labeled/train.csv", 
        "test_csv": "src/data/labeled/test.csv",
        "text_column": "text",
        "label_column": "label"
    }
}

# Model configurations mapping directory structure to model parameters
MODELS_CONFIG = {
    "roberta": {
        "model_class": RobertaBiasClassifier,
        "tokenizer_path": "src/models/encoders/roberta",
        "layers": {"1layer": "linear", "2layer": "mlp"},
        "train_modes": {"feat_extr": "head", "ft_part": "part", "ft_full": "full"}
    },
    "distilbert": {
        "model_class": DistilbertBiasClassifier,
        "tokenizer_path": "src/models/encoders/distilbert", 
        "layers": {"1layer": "linear", "2layer": "mlp"},
        "train_modes": {"feat_extr": "head", "ft_part": "part", "ft_full": "full"}
    }
}

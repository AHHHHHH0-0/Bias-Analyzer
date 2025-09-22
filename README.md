# News Article Bias Analyzer

The goal of this project is to explore different RoBERTa and DistilBERT architectures for NLP tasks, and use trained models for analyzing political bias in news articles.

## Project Overview

This project implements a sophisticated news analysis system that can:

- **Data ETL**: Gather news articles from via NewsAPI, preprocess and vectorize
- **Model Training**: Train 12 different model configurations using RoBERTa and DistilBERT with various layer architectures and fine-tuning strategies
- **Political Bias Classification**: Classify articles as left-leaning, center, or right-leaning
- **Comprehensive Model Evaluation**: Compare 12 different model configurations systematically
- **Real-time Inference**: Process new articles with trained models

## Model Architectures and Training Configurations

**Models**

- `RoBERTa`: Robust transformer model optimized for understanding
- `DistilBERT`: Lightweight, efficient transformer model

**Layer Configurations:**

- `1layer`: Linear classifier head
- `2layer`: Multi-layer perceptron (MLP) classifier head

**Training Modes:**

- `feat_extr`: Feature extraction (frozen backbone, train head only)
- `ft_part`: Partial fine-tuning (unfreeze top layers)
- `ft_full`: Full fine-tuning (train entire model)

### Available Combinations

| Model        | Layers   | Training    | Description                                      |
| ------------ | -------- | ----------- | ------------------------------------------------ |
| `roberta`    | `1layer` | `feat_extr` | Fast, frozen RoBERTa with linear head            |
| `roberta`    | `1layer` | `ft_part`   | Partially fine-tuned RoBERTa with linear head    |
| `roberta`    | `1layer` | `ft_full`   | Fully fine-tuned RoBERTa with linear head        |
| `roberta`    | `2layer` | `feat_extr` | Fast, frozen RoBERTa with MLP head               |
| `roberta`    | `2layer` | `ft_part`   | Partially fine-tuned RoBERTa with MLP head       |
| `roberta`    | `2layer` | `ft_full`   | Fully fine-tuned RoBERTa with MLP head           |
| `distilbert` | `1layer` | `feat_extr` | Fast, frozen DistilBERT with linear head         |
| `distilbert` | `1layer` | `ft_part`   | Partially fine-tuned DistilBERT with linear head |
| `distilbert` | `1layer` | `ft_full`   | Fully fine-tuned DistilBERT with linear head     |
| `distilbert` | `2layer` | `feat_extr` | Fast, frozen DistilBERT with MLP head            |
| `distilbert` | `2layer` | `ft_part`   | Partially fine-tuned DistilBERT with MLP head    |
| `distilbert` | `2layer` | `ft_full`   | Fully fine-tuned DistilBERT with MLP head        |

## Quick Start

### 1. Data Collection

Create a `.env` file in the project root with your NewsAPI key:

```bash
NEWS_API_KEY=your_news_api_key_here
```

```bash
python src/data/data.py
```

### 2. Model Training

Train all 12 model configurations:

```bash
python src/model_training/train.py
```

### 3. Model Evaluation

```bash
python src/model_evaluation/evaluate.py
```

### 4. Run Inference

Configure your model in `src/model_inference/config.py`:

```python
MODEL_CONFIG = {
    "model": "roberta",        # Options: "roberta", "distilbert"
    "layers": "2layer",        # Options: "1layer", "2layer"
    "training": "ft_full"      # Options: "feat_extr", "ft_part", "ft_full"
}
```

Run:

```bash
python src/model_inference/inference.py
```

## Project Structure

```
News-Article-Sentiment-Bias-Analyzer/
├── src/
│   ├── data/                         # Data collection and preprocessing
│   │   ├── labeled/                  # Training/evaluation datasets
│   │   ├── processed/                # Preprocessed articles for inference
│   │   ├── raw/                      # Raw collected articles
│   │   ├── utils/
│   │   │   |── config.py             # API configurations
│   │   │   ├── labeled_datasets.py   # Sample labeled data generation
│   │   │   ├── news_api.py           # NewsAPI client
│   │   │   └── text_preprocessor.py  # Text cleaning utilities
│   │   └── data.py                   # Main data collection script
│   │
│   ├── models/                       # Model definitions and storage
│   │   ├── base/
│   │   │   ├── roberta_bias_classifier.py
│   │   │   └── distilbert_bias_classifier.py
│   │   ├── encoders/                 # Pre-downloaded model encoders
│   │   ├── trained/                  # Trained model checkpoints
│   │   └── model_loader.py           # Model downloading utilities
│   │
│   ├── model_training/               # Training pipeline
│   │   ├── train.py                  # Main training script
│   │   ├── dataset.py                # PyTorch dataset classes
│   │   ├── metrics.py                # Training metrics
│   │   └── config.py                 # Training configurations
│   │
│   ├── model_evaluation/             # Evaluation and analysis
│   │   ├── evaluate.py               # Main evaluation script
│   │   ├── evaluation_runner.py      # Comprehensive evaluation pipeline
│   │   ├── metrics.py                # Evaluation metrics
│   │   ├── visualizations.py         # Result visualizations
│   │   └── results/                  # Evaluation outputs
│   │       ├── metrics/              # Performance metrics CSVs
│   │       └── visualizations/       # Charts and confusion matrices
│   │
│   └── model_inference/              # Inference pipeline
│       ├── inference.py              # Main inference script
│       ├── config.py                 # Inference configurations
│       ├── model_loader.py           # Model loading utilities
│       ├── prediction_engine.py      # Prediction logic
│       ├── text_processor.py         # Text preprocessing for inference
│       ├── output_formatter.py       # Result formatting
│       └── results/                  # Inference outputs
│
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Model Comparisons

The project implements a systematic comparison of:

1. **RoBERTa vs DistillBERT**
2. **Linear probe vs 2-layer MLP** classifier heads
3. **Partial vs full fine-tuning** unfreezing top layers

### Evaluation Metrics

- **Accuracy**: Overall classification correctness
- **F1-Score**: Harmonic mean of precision and recall (macro-averaged)
- **Precision/Recall**: Per-class and averaged performance
- **Confusion Matrix**: Detailed misclassification analysis
- **Model Size**: Parameter count and memory usage
- **Inference Speed**: Processing time per article

## Future Plans

### 1. Sentiment Analysis Classification

- **Objective**: Train models to classify emotional tone in news articles
- **Implementation**: Extend current training pipeline to support sentiment tasks
- **Evaluation**: Compare sentiment models using same 12-configuration framework

### 2. Experiment with Different Model Architectures

- **Layer Freezing Strategies**: Experiment with freezing different transformer layers (early, middle, late layers)
- **Attention Mechanisms**: Add custom attention layers on top of transformer outputs
- **Regularization Techniques**: Dropout variations, weight decay, and layer-wise learning rates

### 3. Different Pooling Strategies

- **Attention Pooling**: Learned attention weights to focus on important tokens
- **Hierarchical Pooling**: Multi-level pooling for longer documents
- **Comparative Analysis**: Systematic evaluation of pooling impact on bias classification performance

## References

- **RoBERTa**: [Liu et al., 2019](https://arxiv.org/abs/1907.11692)
- **DistilBERT**: [Sanh et al., 2019](https://arxiv.org/abs/1910.01108)

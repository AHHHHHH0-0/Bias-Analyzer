"""
Custom dataset class for text classification tasks.
"""
from typing import Dict
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TextClassificationDataset(Dataset):
    """
    Custom PyTorch Dataset for text classification.
    
    Converts raw text and labels into tokenized tensors for training.
    Handles tokenization on-the-fly for memory efficiency.
    """
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        tokenizer: AutoTokenizer, 
        text_column: str, 
        label_column: str, 
        label2id: Dict[str, int], 
        max_length: int
    ) -> None:
        """
        Initialize the dataset.
        
        Args:
            df: DataFrame containing text and labels
            tokenizer: HuggingFace tokenizer (RoBERTa, DistilBERT, etc.)
            text_column: Name of column containing text data
            label_column: Name of column containing labels
            label2id: Mapping from label strings to integer IDs
            max_length: Maximum sequence length for tokenization
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.label_column = label_column
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single tokenized example.
        
        Args:
            idx: Index of the example to retrieve
            
        Returns:
            Dictionary containing:
                - input_ids: Tokenized text as tensor
                - attention_mask: Attention mask tensor
                - labels: Label as tensor
        """
        row = self.df.iloc[idx]
        text = str(row[self.text_column])
        label_id = self.label2id[str(row[self.label_column])]
        
        # Tokenize text
        enc = self.tokenizer(
            text, 
            truncation=True, 
            padding=False, 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        
        # Convert to single example (remove batch dimension)
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label_id, dtype=torch.long)
        
        return item

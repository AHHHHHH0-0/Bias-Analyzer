import torch
import torch.nn as nn
from transformers import RobertaModel


class RobertaBiasClassifier(nn.Module):
    """
    Custom RoBERTa model for political bias classification (left/center/right).
    """
    
    def __init__(self, layers, train_mode, num_classes=3):
        super(RobertaBiasClassifier, self).__init__()
        
        # Load base RoBERTa model
        self.roberta = RobertaModel.from_pretrained("roberta-base")

        # Define configs
        self.hidden_size = self.roberta.config.hidden_size
        self.hidden_dim = 256
        self.dropout_rate = 0.1
        
        # Create classifier based on layers parameter
        if layers == "linear":
            self.classifier = nn.Linear(self.hidden_size, num_classes)
        elif layers == "mlp":
            self.classifier = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim, num_classes)
            )
        else:
            raise ValueError("layers must be 'linear' or 'mlp'")
        
        # Apply training strategy
        self._set_params(train_mode)
    

    def _set_params(self, train_mode):
        """Set trainable parameters based on training mode."""

        if train_mode == "head":
            # Freeze all embeddings and layers
            for param in self.roberta.parameters():
                param.requires_grad = False

        elif train_mode == "part":
            # Freeze embeddings and early layers
            for param in self.roberta.embeddings.parameters():
                param.requires_grad = False
            for layer in self.roberta.encoder.layer[:8]:
                for param in layer.parameters():
                    param.requires_grad = False

        elif train_mode == "full":
            # Unfreeze everything
            for param in self.roberta.parameters():
                param.requires_grad = True

        else:
            raise ValueError("train_mode must be 'head', 'part', or 'full'")
    

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass through the model.
        """
        # Get RoBERTa outputs
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use [CLS] token for classification
        pooled_output = outputs.last_hidden_state[:, 0]  # (batch_size, hidden_size)
        
        # Get classification logits
        logits = self.classifier(pooled_output)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return logits, loss

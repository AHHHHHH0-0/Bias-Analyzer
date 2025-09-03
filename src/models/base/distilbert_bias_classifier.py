import torch
import torch.nn as nn
from transformers import DistilBertModel
from transformers.modeling_outputs import SequenceClassifierOutput


class DistilbertBiasClassifier(nn.Module):
    """
    Custom DistilBERT model for political bias classification (left/center/right).
    """
    
    def __init__(self, layers, train_mode, num_classes=3):
        super(DistilbertBiasClassifier, self).__init__()
        
        # Load base DistilBERT model from preloaded encoders
        self.distilbert = DistilBertModel.from_pretrained("src/models/encoders/distilbert")

        # Define configs
        self.hidden_size = self.distilbert.config.hidden_size
        self.hidden_dim = 256
        self.dropout_rate = 0.2
        
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
            for param in self.distilbert.parameters():
                param.requires_grad = False

        elif train_mode == "part":
            # Freeze embeddings and early layers
            for param in self.distilbert.embeddings.parameters():
                param.requires_grad = False
            for layer in self.distilbert.transformer.layer[:3]:
                for param in layer.parameters():
                    param.requires_grad = False

        elif train_mode == "full":
            # Unfreeze everything
            for param in self.distilbert.parameters():
                param.requires_grad = True

        else:
            raise ValueError("train_mode must be 'head', 'part', or 'full'")
    

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass through the model.
        """
        # Get DistilBERT outputs
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use mean pooling for classification
        pooled_output = outputs.last_hidden_state.mean(dim=1)  # (batch_size, hidden_size)
        
        # Get classification logits
        logits = self.classifier(pooled_output)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None
        )

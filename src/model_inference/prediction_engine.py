import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class PredictionEngine:
    """
    Handles model inference and prediction generation.
    """
    
    def __init__(self, model, label_mapping, device):
        """
        Initialize the prediction engine.
        
        Args:
            model: Trained PyTorch model
            label_mapping (dict): Mapping from label IDs to label names
            device: PyTorch device
        """
        self.model = model
        self.label_mapping = label_mapping
        self.device = device
        
        # Ensure model is in evaluation mode
        self.model.eval()
    
    def predict_batch(self, batch_data):
        """
        Make predictions on a batch of data.
        
        Args:
            batch_data (dict): Tokenized batch data
            
        Returns:
            tuple: (predictions, confidences, probabilities)
        """
        with torch.no_grad():
            # Forward pass
            outputs = self.model(
                input_ids=batch_data["input_ids"],
                attention_mask=batch_data["attention_mask"]
            )
            
            # Get logits and convert to probabilities
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)
            
            # Get predictions and confidence scores
            confidences, predictions = torch.max(probabilities, dim=-1)
            
            # Convert to numpy for easier handling
            predictions = predictions.cpu().numpy()
            confidences = confidences.cpu().numpy()
            probabilities = probabilities.cpu().numpy()
            
            return predictions, confidences, probabilities
    
    def predict_dataset(self, text_processor, df, text_column="full_text", batch_size=16):
        """
        Make predictions on an entire dataset.
        
        Args:
            text_processor: InferenceTextProcessor instance
            df: Input dataframe
            text_column (str): Column containing text
            batch_size (int): Batch size for processing
            
        Returns:
            list: List of prediction results
        """
        all_results = []
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        print(f"Processing {len(df)} articles")
        print(f"Processing in {total_batches} batches of size {batch_size}")
        
        # Process data in batches
        with tqdm(total=len(df), desc="Predicting") as pbar:
            for batch_data, batch_indices, batch_texts in text_processor.prepare_data_for_inference(
                df, text_column, batch_size
            ):
                # Make predictions
                predictions, confidences, probabilities = self.predict_batch(batch_data)
                
                # Process results for this batch
                for idx, pred_id, conf, probs, text in zip(
                    batch_indices, predictions, confidences, probabilities, batch_texts
                ):
                    # Get original article data
                    article_data = df.iloc[idx].to_dict()
                    
                    # Create prediction result
                    result = {
                        "article_index": idx,
                        "article_data": article_data,
                        "prediction": {
                            "label": self.label_mapping[str(pred_id)],
                            "confidence": float(conf),
                            "probabilities": {
                                self.label_mapping[str(i)]: float(prob) 
                                for i, prob in enumerate(probs)
                            }
                        },
                        "text_length": len(text) if text else 0
                    }
                    
                    all_results.append(result)
                
                # Update progress bar
                pbar.update(len(batch_indices))
        
        print(f"✓ Completed predictions for {len(all_results)} articles")
        return all_results
    
    def get_prediction_summary(self, results):
        """
        Generate a summary of predictions.
        
        Args:
            results (list): List of prediction results
            
        Returns:
            dict: Summary statistics
        """
        if not results:
            return {}
        
        # Extract predictions and confidences
        predictions = [r["prediction"]["label"] for r in results]
        confidences = [r["prediction"]["confidence"] for r in results]
        
        # Count predictions by label
        label_counts = {}
        for label in predictions:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Calculate confidence statistics
        conf_stats = {
            "mean": float(np.mean(confidences)),
            "std": float(np.std(confidences)),
            "min": float(np.min(confidences)),
            "max": float(np.max(confidences)),
            "median": float(np.median(confidences))
        }
        
        summary = {
            "total_articles": len(results),
            "label_distribution": label_counts,
            "confidence_stats": conf_stats,
            "low_confidence_count": sum(1 for c in confidences if c < 0.5),
            "high_confidence_count": sum(1 for c in confidences if c > 0.8)
        }
        
        return summary


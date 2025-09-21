import pandas as pd


class InferenceTextProcessor:
    """
    Processes text data for model inference.
    """
    def __init__(self, tokenizer, device, max_length=256):
        """
        Initialize the text processor.
        
        Args:
            tokenizer: Hugging Face tokenizer
            device: PyTorch device
            max_length (int): Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
    
    def load_data(self, file_path):
        """
        Load and validate input data.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pandas.DataFrame: Loaded data
        """
        try:
            df = pd.read_csv(file_path)
            
            print(f"✓ Loaded {len(df)} articles from {file_path}")
            return df
            
        except Exception as e:
            raise RuntimeError(f"Failed to load data from {file_path}: {str(e)}")
    
    def prepare_texts(self, texts):
        """
        Prepare texts for tokenization (handle None/NaN values only).
        
        Args:
            texts (list): List of already preprocessed text strings
            
        Returns:
            list: Clean text strings ready for tokenization
        """
        return [str(text) if not pd.isna(text) and text is not None else "" for text in texts]
    
    def tokenize_batch(self, texts):
        """
        Tokenize a batch of texts for model input.
        
        Args:
            texts (list): List of already preprocessed text strings
            
        Returns:
            dict: Tokenized inputs ready for model
        """
        # Prepare texts (handle None/NaN only)
        prepared_texts = self.prepare_texts(texts)
        
        # Tokenize
        encoded = self.tokenizer(
            prepared_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move to device
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        
        return encoded
    
    def prepare_data_for_inference(self, df, text_column="full_text", batch_size=16):
        """
        Prepare data for inference by creating batches.
        
        Args:
            df (pandas.DataFrame): Input dataframe
            text_column (str): Column containing text
            batch_size (int): Batch size for processing
            
        Yields:
            tuple: (batch_data, batch_indices, batch_texts)
        """
        texts = df[text_column].tolist()
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_indices = list(range(i, min(i + batch_size, len(texts))))
            
            # Tokenize batch
            batch_data = self.tokenize_batch(batch_texts)
            
            yield batch_data, batch_indices, batch_texts


# src/data/preprocessor.py
import re
import unicodedata
import pandas as pd
from bs4 import BeautifulSoup


class TextPreprocessor:
    """
    Preprocesses text data by removing HTML tags, normalizing whitespace, and normalizing Unicode characters.
    """

    def __init__(self):
        pass
        

    def clean_html(self, text):
        """Remove HTML tags from text"""
        try:
            return BeautifulSoup(text, "html.parser").get_text()
        except Exception as e:
            print(f"Error removing HTML: {e}")
            return text
    

    def normalize_whitespace(self, text):
        """Normalize whitespace in text"""
        try:
            return re.sub(r'\s+', ' ', text).strip()
        except Exception as e:
            print(f"Error normalizing whitespace: {e}")
            return text
    

    def normalize_unicode(self, text):
        """Normalize Unicode characters"""
        try:
            return unicodedata.normalize('NFKC', text)
        except Exception as e:
            print(f"Error normalizing Unicode: {e}")
            return text
    

    def preprocess_text(self, text):
        """Apply all preprocessing steps to text"""
        if pd.isna(text) or text is None:
            return ""
        
        text = self.clean_html(text)
        text = self.normalize_unicode(text)
        text = self.normalize_whitespace(text)
        return text
    

    def preprocess_dataframe(self, df, text_columns=['title', 'description', 'content']):
        """
        Preprocess specified text columns in a DataFrame
        
        Args:
            df (pandas.DataFrame): DataFrame containing articles
            text_columns (list): List of column names to preprocess
            
        Returns:
            pandas.DataFrame: Processed DataFrame
        """
        # Create a copy to avoid modifying the original
        processed_df = df.copy()
        
        # Apply preprocessing to specified columns
        for column in text_columns:
            if column in processed_df.columns:
                processed_df[column] = processed_df[column].apply(self.preprocess_text)
        
        # Create a new column combining title and content for analysis
        if 'title' in processed_df.columns and 'content' in processed_df.columns:
            processed_df['full_text'] = processed_df['title'] + " " + processed_df['content']
        
        return processed_df
    
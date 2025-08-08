import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(src_dir)

from config import NEWS_API_KEY, BASE_URL, DAYS_BACK, LANGUAGE, PAGE_SIZE


class NewsAPIClient:
    def __init__(self, api_key=NEWS_API_KEY, base_url=BASE_URL):
        self.api_key = api_key
        self.base_url = BASE_URL
        
    def get_articles_by_keyword(self, keyword, days_back=DAYS_BACK, language=LANGUAGE, page_size=PAGE_SIZE):
        """
        Fetch articles through the API based on keyword search
        
        Args:
            keyword (str): Keyword to search for
            days_back (int): How many days back to search
            language (str): Language code, e.g., 'en' for English
            page_size (int): Number of results per page (max 100)
            
        Returns:
            pandas.DataFrame: Dataframe containing article information
        """
        # Calculate date range
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        # Build API endpoint
        endpoint = f"{self.base_url}/everything"
        
        # Set parameters
        params = {
            'q': keyword,
            'from': start_date,
            'to': end_date,
            'language': language,
            'sortBy': 'publishedAt',
            'pageSize': page_size,
            'apiKey': self.api_key
        }
        
        # Make the request
        response = requests.get(endpoint, params=params)
        
        # Check if request was successful
        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])
            
            # Convert to DataFrame
            if articles:
                df = pd.DataFrame(articles)
                return df
            else:
                print(f"No articles found for keyword: {keyword}")
                return pd.DataFrame()
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return pd.DataFrame()
    

    def get_articles_by_source(self, sources, days_back=DAYS_BACK, page_size=PAGE_SIZE):
        """
        Fetch articles through the API based on specific sources
        
        Args:
            sources (str or list): Source ID or comma-separated list of source IDs
            days_back (int): How many days back to search
            page_size (int): Number of results per page (max 100)
            
        Returns:
            pandas.DataFrame: Dataframe containing article information
        """
        # Convert list of sources to comma-separated string if needed
        if isinstance(sources, list):
            sources = ','.join(sources)
        
        # Calculate date range
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        # Build API endpoint
        endpoint = f"{self.base_url}/everything"
        
        # Set parameters
        params = {
            'sources': sources,
            'from': start_date,
            'to': end_date,
            'sortBy': 'publishedAt',
            'pageSize': page_size,
            'apiKey': self.api_key
        }
        
        # Make the request
        response = requests.get(endpoint, params=params)
        
        # Check if request was successful
        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])
            
            # Convert to DataFrame
            if articles:
                df = pd.DataFrame(articles)
                return df
            else:
                print(f"No articles found for sources: {sources}")
                return pd.DataFrame()
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return pd.DataFrame()
    

    def save_articles_to_csv(self, df, filename):
        """
        Save articles DataFrame to CSV file
        
        Args:
            df (pandas.DataFrame): DataFrame containing articles
            filename (str): Output filename
        """
        if not df.empty:
            df.to_csv(f"src/data/raw/{filename}", index=False)
            print(f"Saved {len(df)} articles to {filename}")
        else:
            print("No articles to save")
            
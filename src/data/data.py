import pandas as pd
import os

from utils.news_api import NewsAPIClient
from utils.text_preprocessor import TextPreprocessor
from utils.labeled_datasets import save_labeled_datasets


def collect_dataset(topics, sources, days_back):
    """
    Collect a diverse dataset of news articles
    
    Args:
        topics (list): List of topics to search for
        sources (list): List of news sources
        days_back (int): How many days back to search
        
    Returns:
        pandas.DataFrame: Combined dataset
    """
    # Make directories
    os.makedirs("src/data/raw", exist_ok=True)
    os.makedirs("src/data/processed", exist_ok=True)
    
    # Initialize NewsAPI client
    client = NewsAPIClient()
    
    all_articles = []
    
    # Collect articles by topic
    for topic in topics:
        print(f"Fetching articles about {topic}...")
        df = client.get_articles_by_keyword(topic, days_back=days_back)
        if not df.empty:
            df['search_topic'] = topic
            df.to_csv(f"src/data/raw/{topic}.csv", index=False)
            all_articles.append(df)
    
    # Collect articles by source
    print(f"Fetching articles from sources {sources}...")
    df_sources = client.get_articles_by_source(sources, days_back=days_back)
    if not df_sources.empty:
        df_sources['search_topic'] = 'source_specific'
        df_sources.to_csv(f"src/data/raw/sources.csv", index=False)
        all_articles.append(df_sources)
    
    # Combine articles
    if all_articles:
        combined_df = pd.concat(all_articles, ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['url'])
        return combined_df
    else:
        print("No articles collected")
        return pd.DataFrame()


def main():
    # Define topics and sources
    topics = [
        "climate change",
        "artificial intelligence",
        "healthcare",
        "economy",
        "politics"
    ]
    sources = [
        "bbc-news",
        "cnn",
        "fox-news",
        "the-washington-post",
        "reuters",
        "the-wall-street-journal"
    ]

    days_back = 7

    # Collect dataset
    raw_df = collect_dataset(topics, sources, days_back)
    
    if not raw_df.empty:
        # Save raw dataset
        raw_df.to_csv("src/data/raw/combined_articles.csv", index=False)
        print(f"Saved {len(raw_df)} raw articles")
        
        # Preprocess dataset
        preprocessor = TextPreprocessor()
        processed_df = preprocessor.preprocess_dataframe(raw_df)
        
        # Save processed dataset
        processed_df.to_csv("src/data/processed/processed_articles.csv", index=False)
        print(f"Saved {len(processed_df)} processed articles")


if __name__ == "__main__":
    main()
    save_labeled_datasets()

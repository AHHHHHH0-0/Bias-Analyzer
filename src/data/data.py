from utils.news_api import NewsAPIClient
import os


def save_articles_to_csv(df, filename):
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

def main(keywords, sources):
    # Create raw data directory
    os.makedirs("src/data/raw", exist_ok=True)
    
    # Initialize NewsAPI client
    client = NewsAPIClient()
    
    # Test fetching articles by keyword
    df_ai = client.get_articles_by_keyword(keywords, days_back=7)
    if not df_ai.empty:
        save_articles_to_csv(df_ai, "ai_articles.csv")
    
    # Test fetching articles by source
    df_sources = client.get_articles_by_source(sources, days_back=7)
    if not df_sources.empty:
        save_articles_to_csv(df_sources, "source_articles.csv")

if __name__ == "__main__":
    keywords = ["artificial intelligence"]
    sources = ["bbc-news", "cnn", "fox-news", "the-verge"]
    main(keywords, sources)
    
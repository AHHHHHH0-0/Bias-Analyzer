from src.data.news_api import NewsAPIClient
import os

def main():
    # Create raw data directory
    os.makedirs("src/data/raw", exist_ok=True)
    
    # Initialize NewsAPI client
    client = NewsAPIClient()
    
    # Test fetching articles by keyword
    df_ai = client.get_articles_by_keyword("artificial intelligence", days_back=7)
    if not df_ai.empty:
        client.save_articles_to_csv(df_ai, "ai_articles.csv")
    
    # Test fetching articles by source
    sources = ["bbc-news", "cnn", "fox-news", "the-verge"]
    df_sources = client.get_articles_by_source(sources, days_back=7)
    if not df_sources.empty:
        client.save_articles_to_csv(df_sources, "source_articles.csv")

if __name__ == "__main__":
    main()
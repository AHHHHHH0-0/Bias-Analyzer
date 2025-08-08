import os
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# API keys and configurations
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
BASE_URL = "https://newsapi.org/v2"

# API fetch parameters
DAYS_BACK = 7
LANGUAGE = 'en'
PAGE_SIZE = 100

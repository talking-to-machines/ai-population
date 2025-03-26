import os
from dotenv import load_dotenv

load_dotenv()

APIFY_API = os.getenv("APIFY_API")
APIFY_ACTOR_ID = os.getenv("APIFY_ACTOR_ID")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GPT_MODEL = "gpt-4o-mini"  # gpt-4o, gpt-4o-mini
RESULTS_PER_PAGE = 1000
TOP_N_PROFILES = 100
PROJECT = "market-signals-finfluencer"

SEARCH_TERMS_FILE = "market_signals_finfluencer_search_terms.txt"
PROFILES_FILE = "market_signals_finfluencer_profiles_finfluencers.txt"
PROFILESEARCH_VIDEO_METADATA_FILE = "profilesearch_video_metadata_identification.csv"
PROFILESEARCH_PROFILE_METADATA_FILE = (
    "profilesearch_profile_metadata_identification.csv"
)
KEYWORDSEARCH_VIDEO_METADATA_FILE = "keywordsearch_video_metadata.csv"
KEYWORDSEARCH_PROFILE_METADATA_FILE = "keywordsearch_profile_metadata.csv"
POST_IDENTIFICATION_FILE = "profile_metadata_post_identification.csv"
PANEL_PROFILE_METADATA_FILE = "profile_metadata_panel.csv"
POST_REFLECTION_FILE = "profile_metadata_post_reflection.csv"
POST_STOCK_EXTRACTION_FILE = "profile_metadata_post_extraction.csv"
POST_INTERVIEW_FILE = "profile_metadata_post_interview.csv"
FORMATTED_POST_INTERVIEW_FILE = "profile_metadata_post_interview_formatted.csv"
STOCK_RECOMMENDATION_FILE = "stock_recommendations_{interview_date}.csv"
RUSSELL_4000_STOCK_TICKER_FILE = "russell4000_stock_tickers_shorten.csv"

import os
from dotenv import load_dotenv

load_dotenv()

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

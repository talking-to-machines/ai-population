import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(dotenv_path="ai_population/config/.env")

base_dir = os.path.dirname(os.path.abspath(__file__))

# Common configurations for market signals project
PIPELINE_EXECUTION_DATE = datetime.today().date().strftime("%d-%m-%Y")
MIN_FOLLOWER_COUNT = 5000
MIN_VIDEO_COUNT = 10
NUM_POST_PER_KEYWORD = 50
NUM_RESULTS_PER_PROFILE = 25
PROFILE_SEARCH_START_DATE = "06-08-2025"
PROFILE_SEARCH_END_DATE = "06-08-2025"
RUSSELL_4000_STOCK_TICKER_FILE = "russell4000_stock_tickers_shorten.csv"

# Tiktok-specific configurations
PROJECT_NAME_TIKTOK = "market-signals-tiktok"
KEYWORD_SEARCH_FILE_TIKTOK = f"tiktok_keyword_search_{PIPELINE_EXECUTION_DATE}.csv"
PROFILE_METADATA_SEARCH_FILE_TIKTOK = (
    f"tiktok_profile_metadata_{PIPELINE_EXECUTION_DATE}.csv"
)
FINFLUENCER_POOL_FILE_TIKTOK = "tiktok_verified_finfluencer_profiles.csv"
ONBOARDING_RESULTS_FILE_TIKTOK = (
    f"tiktok_onboarding_results_{PIPELINE_EXECUTION_DATE}.csv"
)
FINFLUENCER_PROFILE_METADATA_SEARCH_FILE_TIKTOK = (
    f"tiktok_finfluencer_profile_metadata_{PIPELINE_EXECUTION_DATE}.csv"
)
FINFLUENCER_PROFILE_SEARCH_FILE_TIKTOK = (
    f"tiktok_finfluencer_profile_search_{PIPELINE_EXECUTION_DATE}.csv"
)
FINFLUENCER_EXPERT_REFLECTION_FILE_TIKTOK = (
    f"tiktok_finfluencer_expert_reflection_{PIPELINE_EXECUTION_DATE}.csv"
)
FINFLUENCER_STOCK_MENTIONS_FILE_TIKTOK = (
    f"tiktok_finfluencer_stock_mentions_{PIPELINE_EXECUTION_DATE}.csv"
)
FINFLUENCER_POST_INTERVIEW_FILE_TIKTOK = (
    f"tiktok_finfluencer_post_interview_{PIPELINE_EXECUTION_DATE}.csv"
)
FINFLUENCER_STOCK_RECOMMENDATION_FILE_TIKTOK = (
    f"tiktok_finfluencer_stock_recommendation_{PIPELINE_EXECUTION_DATE}.csv"
)
SEARCH_TERMS_TIKTOK = [
    "stocks",
    "stock market",
    "stock picks",
    "sp 500",
    "ticker symbol",
    "stockstowatch",
    "invest",
    "investing",
    "wealth investing",
    "trading",
    "traders",
    "option traders",
    "daytrading",
    "follow trades",
    "market",
    "company",
    "nvidia",
    "economy",
    "business",
    "wall street",
    "cash flow",
    "federal reserve",
    "interest rates",
    "finance",
    "financial advice",
    "united states",
    "tariffs",
    "retirement",
    "wealth",
    "trump",
    "elon musk",
    "news",
    "ai",
]

# X-specific configurations
PROJECT_NAME_X = "market-signals-x"
KEYWORD_SEARCH_FILE_x = f"x_keyword_search_{PIPELINE_EXECUTION_DATE}.csv"
PROFILE_METADATA_SEARCH_FILE_x = f"x_profile_metadata_{PIPELINE_EXECUTION_DATE}.csv"
FINFLUENCER_POOL_FILE_X = "x_verified_finfluencer_profiles.csv"
ONBOARDING_RESULTS_FILE_X = f"x_onboarding_results_{PIPELINE_EXECUTION_DATE}.csv"
FINFLUENCER_PROFILE_METADATA_SEARCH_FILE_X = (
    f"x_finfluencer_profile_metadata_{PIPELINE_EXECUTION_DATE}.csv"
)
FINFLUENCER_PROFILE_SEARCH_FILE_X = (
    f"x_finfluencer_profile_search_{PIPELINE_EXECUTION_DATE}.csv"
)
FINFLUENCER_EXPERT_REFLECTION_FILE_X = (
    f"x_finfluencer_expert_reflection_{PIPELINE_EXECUTION_DATE}.csv"
)
FINFLUENCER_STOCK_MENTIONS_FILE_X = (
    f"x_finfluencer_stock_mentions_{PIPELINE_EXECUTION_DATE}.csv"
)
FINFLUENCER_POST_INTERVIEW_FILE_X = (
    f"x_finfluencer_post_interview_{PIPELINE_EXECUTION_DATE}.csv"
)
FINFLUENCER_STOCK_RECOMMENDATION_FILE_X = (
    f"x_finfluencer_stock_recommendation_{PIPELINE_EXECUTION_DATE}.csv"
)
SEARCH_TERMS_X = [
    "",
]

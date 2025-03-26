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
PROJECT = "canada-elections"

SEARCH_TERMS_FILE = "canada_elections_search_terms.txt"
PROFILES_FILE = "canada_elections_profiles.txt"
KEYWORDSEARCH_VIDEO_METADATA_FILE = ("",)
KEYWORDSEARCH_PROFILE_METADATA_FILE = ("",)
POOL_TYPE = "LATEST"  # LATEST or STORAGE
POLLED_USER_POOL_FILE = "canada_elections_polled_user_pool.csv"
POLLING_INCLUSION_PERIOD = 7  # 7 days exclusion
SURVEY_FILE = "canada_elections_survey.csv"
SURVEY_PROGRESS_LOG = "survey_progress_log.pkl"

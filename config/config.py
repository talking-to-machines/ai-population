import os
from dotenv import load_dotenv

load_dotenv()

APIFY_API = os.getenv("APIFY_API")
APIFY_ACTOR_ID = os.getenv("APIFY_ACTOR_ID")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

RESULTS_PER_PAGE = 1000
TOP_N_PROFILES = 100

### Configuration for Market Signals Finfluencer (START) ###
PROJECT = "market-signals-finfluencer"

SEARCH_TERMS_FILE = "market_signals_finfluencer_search_terms.txt"
PROFILES_FILE = "market_signals_finfluencer_profiles_finfluencers.txt"

PROFILESEARCH_VIDEO_METADATA_FILE = "profilesearch_video_metadata_identification.csv"
PROFILESEARCH_PROFILE_METADATA_FILE = (
    "profilesearch_profile_metadata_identification.csv"
)

KEYWORDSEARCH_VIDEO_METADATA_FILE = "keywordsearch_video_metadata.csv"
KEYWORDSEARCH_PROFILE_METADATA_FILE = "keywordsearch_profile_metadata.csv"

POST_INTERVIEW_FILE = "profile_metadata_postinterview_identification.csv"
### Configuration for Market Signals Finfluencer (END) ###

### Configuration for Market Signals Political Influencer (START) ###
# PROJECT = "market-signals-politicalinfluencer"
# SEARCH_TERMS_FILE = "market_signals_politicalinfluencer_search_terms.txt"
# PROFILES_FILE = "market_signals_politicalinfluencer_profiles.txt"
### Configuration for Market Signals Political Influencer (END) ###

### Configuration for Market Signals Political Influencer (START) ###
# PROJECT = "canada-elections"
# SEARCH_TERMS_FILE = "canada_elections_search_terms.txt"
# PROFILES_FILE = "canada_elections_profiles.txt"
# KEYWORDSEARCH_VIDEO_METADATA_FILE = "",
# KEYWORDSEARCH_PROFILE_METADATA_FILE = "",
# POOL_TYPE = "LATEST"  # LATEST or STORAGE
# POLLED_USER_POOL_FILE = "canada_elections_polled_user_pool.csv"
# POLLING_INCLUSION_PERIOD = 7  # 7 days exclusion
# SURVEY_FILE = "canada_elections_survey.csv"
# SURVEY_PROGRESS_LOG = "survey_progress_log.pkl"
### Configuration for Market Signals Political Influencer (END) ###

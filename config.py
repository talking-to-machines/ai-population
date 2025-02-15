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
PROFILES_FILE = "market_signals_finfluencer_profiles.txt"
### Configuration for Market Signals Finfluencer (END) ###

### Configuration for Market Signals Political Influencer (START) ###
# PROJECT = "market-signals-politicalinfluencer"
# SEARCH_TERMS_FILE = "market_signals_politicalinfluencer_search_terms.txt"
# PROFILES_FILE = "market_signals_politicalinfluencer_profiles.txt"
### Configuration for Market Signals Political Influencer (END) ###

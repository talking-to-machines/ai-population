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

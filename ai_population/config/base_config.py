import os
from dotenv import load_dotenv


load_dotenv(dotenv_path="ai_population/config/.env")

BRIGHTDATA_API = os.getenv("BRIGHTDATA_API")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GPT_MODEL = "gpt-4o"  # gpt-4o, gpt-4o-mini
TOP_N_PROFILES = 100
WAIT_TIME_BETWEEN_RETRIEVAL_REQUESTS = 300  # in seconds

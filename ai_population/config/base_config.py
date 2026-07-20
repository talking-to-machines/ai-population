import os
from dotenv import load_dotenv


load_dotenv(dotenv_path="ai_population/config/.env")

BRIGHTDATA_API = os.getenv("BRIGHTDATA_API")
X_API_USERNAME = os.getenv("X_API_USERNAME")
X_API_PASSWORD = os.getenv("X_API_PASSWORD")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")
GPT_MODEL = "gpt-5.1-2025-11-13"  # gpt-4.1-2025-04-14, gpt-5-mini-2025-08-07, gpt-5-nano-2025-08-07, gpt-5.1-2025-11-13, grok-4.3, claude-haiku-4-5
XAI_BASE_URL = "https://api.x.ai/v1"
ANTHROPIC_MAX_OUTPUT_TOKENS = 4096
TOP_N_PROFILES = 100
WAIT_TIME_BETWEEN_RETRIEVAL_REQUESTS = 300  # in seconds
MAX_RETRIES = 5
NUM_PARALLEL_PROCESSES = 20

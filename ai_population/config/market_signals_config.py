import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(dotenv_path="ai_population/config/.env")

base_dir = os.path.dirname(os.path.abspath(__file__))

# Common configurations for market signals project
PIPELINE_EXECUTION_DATE = datetime.today().date().strftime("%d-%m-%Y")
MIN_FOLLOWER_COUNT = 5000
MIN_VIDEO_COUNT = 10
MIN_POSTS_COUNT = 10
NUM_POST_PER_KEYWORD = 20
NUM_RESULTS_PER_PROFILE = 20
PROFILE_SEARCH_START_DATE = "06-23-2025"  # MM-DD-YYYY format
PROFILE_SEARCH_END_DATE = "06-24-2025"  # MM-DD-YYYY format
RUSSELL_4000_STOCK_TICKER_FILE = "russell4000_stock_tickers_shorten.csv"
ONBOARDING_INTERVIEW_REGEX_PATTERNS = [
    r"^Is this a finfluencer.*\-\s*explanation$",
    r"^Is this a finfluencer.*\-\s*symbol$",
    r"^Is this a finfluencer.*\-\s*category$",
    r"^Is this a finfluencer.*\-\s*speculation$",
    r"^Indicate on a scale of 0 to 100, how influential this influencer is.*\-\s*explanation$",
    r"^Indicate on a scale of 0 to 100, how influential this influencer is.*\-\s*speculation$",
    r"^Indicate on a scale of 0 to 100, how influential this influencer is.*\-\s*value$",
    r"^Indicate on a scale of 0 to 100, how credible or authoritative this influencer is.*\-\s*explanation$",
    r"^Indicate on a scale of 0 to 100, how credible or authoritative this influencer is.*\-\s*speculation$",
    r"^Indicate on a scale of 0 to 100, how credible or authoritative this influencer is.*\-\s*value$",
    r"^Which of these areas of finance are the primary focus of the influencer’s posts.*\-\s*explanation$",
    r"^Which of these areas of finance are the primary focus of the influencer’s posts.*\-\s*symbol$",
    r"^Which of these areas of finance are the primary focus of the influencer’s posts.*\-\s*category$",
    r"^Which of these areas of finance are the primary focus of the influencer’s posts.*\-\s*speculation$",
    r"^Indicate on a scale of 0 to 100, how would you rate the quality of this influencer's individual stock predictions.*\-\s*explanation$",
    r"^Indicate on a scale of 0 to 100, how would you rate the quality of this influencer's individual stock predictions.*\-\s*speculation$",
    r"^Indicate on a scale of 0 to 100, how would you rate the quality of this influencer's individual stock predictions.*\-\s*value$",
    r"^Indicate on a scale of 0 to 100, how would you rate the quality of this influencer's evaluation of market sentiment.*\-\s*explanation$",
    r"^Indicate on a scale of 0 to 100, how would you rate the quality of this influencer's evaluation of market sentiment.*\-\s*speculation$",
    r"^Indicate on a scale of 0 to 100, how would you rate the quality of this influencer's evaluation of market sentiment.*\-\s*value$",
    r"^Indicate on a scale of 0 to 100, how would you rate the quality of this influencer's broader evaluation of the economy.*\-\s*explanation$",
    r"^Indicate on a scale of 0 to 100, how would you rate the quality of this influencer's broader evaluation of the economy.*\-\s*speculation$",
    r"^Indicate on a scale of 0 to 100, how would you rate the quality of this influencer's broader evaluation of the economy.*\-\s*value$",
    r"^Who is the finfluencer’s target audience.*\-\s*explanation$",
    r"^Who is the finfluencer’s target audience.*\-\s*symbol$",
    r"^Who is the finfluencer’s target audience.*\-\s*category$",
    r"^Who is the finfluencer’s target audience.*\-\s*speculation$",
]
FINFLUENCER_INTERVIEW_REGEX_PATTERNS = [
    r"^Do you agree or disagree with the following statement: The U.S. economy is likely to enter a recession in the next 12 months.*\-\s*explanation$",
    r"^Do you agree or disagree with the following statement: The U.S. economy is likely to enter a recession in the next 12 months.*\-\s*symbol$",
    r"^Do you agree or disagree with the following statement: The U.S. economy is likely to enter a recession in the next 12 months.*\-\s*category$",
    r"^Do you agree or disagree with the following statement: The U.S. economy is likely to enter a recession in the next 12 months.*\-\s*speculation$",
    r"^How would you describe the current market sentiment among investors.*\-\s*explanation$",
    r"^How would you describe the current market sentiment among investors.*\-\s*symbol$",
    r"^How would you describe the current market sentiment among investors.*\-\s*category$",
    r"^How would you describe the current market sentiment among investors.*\-\s*speculation$",
    r"^Regarding the future direction of the stock market.*\-\s*explanation$",
    r"^Regarding the future direction of the stock market.*\-\s*symbol$",
    r"^Regarding the future direction of the stock market.*\-\s*category$",
    r"^Regarding the future direction of the stock market.*\-\s*speculation$",
    r"^In the next 1–3 months, do you expect U.S. stock market indices to rise, stay about the same, or fall.*\-\s*explanation$",
    r"^In the next 1–3 months, do you expect U.S. stock market indices to rise, stay about the same, or fall.*\-\s*symbol$",
    r"^In the next 1–3 months, do you expect U.S. stock market indices to rise, stay about the same, or fall.*\-\s*category$",
    r"^In the next 1–3 months, do you expect U.S. stock market indices to rise, stay about the same, or fall.*\-\s*speculation$",
    r"^In the next 1–3 months, do you expect U.S. bond prices (or interest rates) to rise, remain unchanged, or fall.*\-\s*explanation$",
    r"^In the next 1–3 months, do you expect U.S. bond prices (or interest rates) to rise, remain unchanged, or fall.*\-\s*symbol$",
    r"^In the next 1–3 months, do you expect U.S. bond prices (or interest rates) to rise, remain unchanged, or fall.*\-\s*category$",
    r"^In the next 1–3 months, do you expect U.S. bond prices (or interest rates) to rise, remain unchanged, or fall.*\-\s*speculation$",
    r"^Considering current market conditions, which sectors do you believe are poised to do well in the next 3–6 months.*\-\s*speculation$",
    r"^Considering current market conditions, which sectors do you believe are poised to do well in the next 3–6 months.*\-\s*response$",
    r"^Considering current market conditions, which sectors do you believe are poised to do poorly in the next 3–6 months.*\-\s*speculation$",
    r"^Considering current market conditions, which sectors do you believe are poised to do poorly in the next 3–6 months.*\-\s*response$",
    r"^Did you mention any stocks or stock tickers in the Russell 4000 list.*\-\s*explanation$",
    r"^Did you mention any stocks or stock tickers in the Russell 4000 list.*\-\s*symbol$",
    r"^Did you mention any stocks or stock tickers in the Russell 4000 list.*\-\s*category$",
    r"^Did you mention any stocks or stock tickers in the Russell 4000 list.*\-\s*speculation$",
    r"^Is there anything else about the economy or markets that you’d like to comment on that we didn’t cover.*\-\s*speculation$",
    r"^Is there anything else about the economy or markets that you’d like to comment on that we didn’t cover.*\-\s*response$",
]
STOCK_RECOMMENDATION_OUTPUT_COLUMNS = [
    "stock_name",
    "stock_ticker",
    "mention_date",
    "post",
    "account_id",
    "followers",
    "url",
    "influence",
    "credibility",
    "mentioned_by_finfluencer",
    "recommendation",
    "explanation",
    "confidence",
    "virality",
]

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
EXPERT_REFLECTION_FILE_TIKTOK = (
    f"tiktok_expert_reflection_{PIPELINE_EXECUTION_DATE}.csv"
)
FINFLUENCER_PROFILE_METADATA_SEARCH_FILE_TIKTOK = (
    f"tiktok_finfluencer_profile_metadata_{PIPELINE_EXECUTION_DATE}.csv"
)
FINFLUENCER_PROFILE_SEARCH_FILE_TIKTOK = (
    f"tiktok_finfluencer_profile_search_{PIPELINE_EXECUTION_DATE}.csv"
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
    "top stock",
    "underrated stocks",
    "stockstowatch",
    "stockstobuy",
    "invest",
    "invest follow",
    "investing stocks",
    "investingtips",
    "investing101",
    "investingbeginner",
    "follow trades",
    "daytrading",
    "option traders",
    "tariffs",
    "company",
    "business",
    "inflation",
    "interest rates",
    "ticker symbol",
    "wall street",
    "cash flow",
    "millennial money",
    "money finance",
    "finance investing",
    "financial advice",
    "financetips",
    "financetok",
    "financialfreedom",
    "financialliteracy",
    "united states",
    "donald trump",
    "news",
    "ai",
]


# X-specific configurations
PROJECT_NAME_X = "market-signals-x"
KEYWORD_SEARCH_FILE_X = f"x_keyword_search_{PIPELINE_EXECUTION_DATE}.csv"
PROFILE_METADATA_SEARCH_FILE_X = f"x_profile_metadata_{PIPELINE_EXECUTION_DATE}.csv"
FINFLUENCER_POOL_FILE_X = "x_verified_finfluencer_profiles.csv"
ONBOARDING_RESULTS_FILE_X = f"x_onboarding_results_{PIPELINE_EXECUTION_DATE}.csv"
EXPERT_REFLECTION_FILE_X = f"x_expert_reflection_{PIPELINE_EXECUTION_DATE}.csv"
FINFLUENCER_PROFILE_METADATA_SEARCH_FILE_X = (
    f"x_finfluencer_profile_metadata_{PIPELINE_EXECUTION_DATE}.csv"
)
FINFLUENCER_PROFILE_SEARCH_FILE_X = (
    f"x_finfluencer_profile_search_{PIPELINE_EXECUTION_DATE}.csv"
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
    "stocks",
    "stock market",
    "investing",
    "finance stock market",
    "dividends",
    "market cap",
    "stock watchlist",
    "spx spy",
    "es spx",
    "spy qqq",
    "dia djia",
    "trading",
    "entry price",
    "profit per share",
    "elliot wave trading",
    "fastest momentum",
    "momentum system",
    "trading zone",
    "short float",
    "jerome powell",
    "rate cuts",
    "bitcoin",
    "traderinsights",
    "smallaccounttrading",
    "abnormal returns",
    "tradinglounge",
    "stockstotrade",
    "optiontrading",
    "marketsurge",
]

import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="ai_population/config/.env")

base_dir = os.path.dirname(os.path.abspath(__file__))

# Common configurations for market signals project
PIPELINE_EXECUTION_DATE = "14-08-2025"  # datetime.today().date().strftime("%d-%m-%Y")
MIN_FOLLOWER_COUNT = 5000
MIN_VIDEO_COUNT = 10
MIN_POSTS_COUNT = 10
NUM_POSTS_PER_KEYWORD = 20
NUM_POSTS_PER_PROFILE = 120  # original value 20
PROFILE_SEARCH_START_DATE = "08-14-2025"  # MM-DD-YYYY format
PROFILE_SEARCH_END_DATE = "08-15-2025"  # MM-DD-YYYY format
RUSSELL_4000_STOCK_TICKER_FILE = "russell4000_stock_tickers_shorten.csv"
ONBOARDING_INTERVIEW_REGEX_PATTERNS = [
    r"^Indicate on a scale of 0 to 100, how likely this creator is a finfluencer.*\-\s*explanation$",
    r"^Indicate on a scale of 0 to 100, how likely this creator is a finfluencer.*\-\s*speculation$",
    r"^Indicate on a scale of 0 to 100, how likely this creator is a finfluencer.*\-\s*value$",
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
    r"^What is your best estimate of the probability that the U.S. economy will enter a recession in the next 12 months.*\-\s*explanation$",
    r"^What is your best estimate of the probability that the U.S. economy will enter a recession in the next 12 months.*\-\s*symbol$",
    r"^What is your best estimate of the probability that the U.S. economy will enter a recession in the next 12 months.*\-\s*category$",
    r"^What is your best estimate of the probability that the U.S. economy will enter a recession in the next 12 months.*\-\s*speculation$",
    r"^Considering business conditions in the country as a whole, do you think that during the next 12 months we’ll have good times financially, or bad times, or what.*\-\s*explanation$",
    r"^Considering business conditions in the country as a whole, do you think that during the next 12 months we’ll have good times financially, or bad times, or what.*\-\s*symbol$",
    r"^Considering business conditions in the country as a whole, do you think that during the next 12 months we’ll have good times financially, or bad times, or what.*\-\s*category$",
    r"^Considering business conditions in the country as a whole, do you think that during the next 12 months we’ll have good times financially, or bad times, or what.*\-\s*speculation$",
    r"^In the next six months, do you expect business conditions to be better, worse, or the same.*\-\s*explanation$",
    r"^In the next six months, do you expect business conditions to be better, worse, or the same.*\-\s*symbol$",
    r"^In the next six months, do you expect business conditions to be better, worse, or the same.*\-\s*category$",
    r"^In the next six months, do you expect business conditions to be better, worse, or the same.*\-\s*speculation$",
    r"^In your view, is the overall investor sentiment currently bearish, neutral, or bullish.*\-\s*explanation$",
    r"^In your view, is the overall investor sentiment currently bearish, neutral, or bullish.*\-\s*symbol$",
    r"^In your view, is the overall investor sentiment currently bearish, neutral, or bullish.*\-\s*category$",
    r"^In your view, is the overall investor sentiment currently bearish, neutral, or bullish.*\-\s*speculation$",
    r"^I feel that the direction of the stock market over the next six months will be: Up (Bullish), No Change (Neutral), or Down (Bearish).*\-\s*explanation$",
    r"^I feel that the direction of the stock market over the next six months will be: Up (Bullish), No Change (Neutral), or Down (Bearish).*\-\s*symbol$",
    r"^I feel that the direction of the stock market over the next six months will be: Up (Bullish), No Change (Neutral), or Down (Bearish).*\-\s*category$",
    r"^I feel that the direction of the stock market over the next six months will be: Up (Bullish), No Change (Neutral), or Down (Bearish).*\-\s*speculation$",
    r"^In the next 6 months, do you expect U.S. stock market indices to rise, stay about the same, or fall.*\-\s*explanation$",
    r"^In the next 6 months, do you expect U.S. stock market indices to rise, stay about the same, or fall.*\-\s*symbol$",
    r"^In the next 6 months, do you expect U.S. stock market indices to rise, stay about the same, or fall.*\-\s*category$",
    r"^In the next 6 months, do you expect U.S. stock market indices to rise, stay about the same, or fall.*\-\s*speculation$",
    r"^In the next 12 months, do you expect interest rates on U.S. bonds to rise, stay about the same, or fall.*\-\s*explanation$",
    r"^In the next 12 months, do you expect interest rates on U.S. bonds to rise, stay about the same, or fall.*\-\s*symbol$",
    r"^In the next 12 months, do you expect interest rates on U.S. bonds to rise, stay about the same, or fall.*\-\s*category$",
    r"^In the next 12 months, do you expect interest rates on U.S. bonds to rise, stay about the same, or fall.*\-\s*speculation$",
    r"^Considering current market conditions, select up to 2-3 sectors you believe are poised for better than average performance over the next 6 months, and briefly explain why.*\-\s*speculation$",
    r"^Considering current market conditions, select up to 2-3 sectors you believe are poised for better than average performance over the next 6 months, and briefly explain why.*\-\s*response$",
    r"^Considering current market conditions, select up to 2-3 sectors you believe are poised for poorer than average performance over the next 6 months, and briefly explain why.*\-\s*speculation$",
    r"^Considering current market conditions, select up to 2-3 sectors you believe are poised for poorer than average performance over the next 6 months, and briefly explain why.*\-\s*response$",
    r"^Did you mention any stocks or stock tickers in the Russell 4000 list.*\-\s*explanation$",
    r"^Did you mention any stocks or stock tickers in the Russell 4000 list.*\-\s*symbol$",
    r"^Did you mention any stocks or stock tickers in the Russell 4000 list.*\-\s*category$",
    r"^Did you mention any stocks or stock tickers in the Russell 4000 list.*\-\s*speculation$",
    r"^Is there anything else about the economy or markets that you’d like to comment on that we didn’t cover?.*\-\s*speculation$",
    r"^Is there anything else about the economy or markets that you’d like to comment on that we didn’t cover?.*\-\s*response$",
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
    "risks",
    "horizon",
    "conflicts",
    "model",
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
PREDICTION_THRESHOLD_TIKTOK = 0.8


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
PREDICTION_THRESHOLD_X = 0.6

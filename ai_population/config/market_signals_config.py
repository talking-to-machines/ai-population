import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv(dotenv_path="ai_population/config/.env")

base_dir = os.path.dirname(os.path.abspath(__file__))

# Common configurations for market signals project
PIPELINE_EXECUTION_DATE = (
    "09-04-2026"  # test  # DD-MM-YYYY  # datetime.today().date().strftime("%d-%m-%Y")
)
MIN_FOLLOWER_COUNT = 5000
MIN_VIDEO_COUNT = 10
MIN_POSTS_COUNT = 10
NUM_POSTS_PER_KEYWORD = 100
NUM_POSTS_PER_PROFILE = 20
LATEST_K_POSTS_PER_PROFILE = 1000
PROFILE_SEARCH_START_DATE = "04-09-2026"  # MM-DD-YYYY  (Inclusive)
PROFILE_SEARCH_END_DATE = "04-10-2026"  # MM-DD-YYYY format  (Exclusive)
RUSSELL_4000_STOCK_TICKER_FILE = "russell4000_stock_tickers_shorten.csv"
SP_500_STOCK_TICKER_FILE = "sp500_stock_tickers_shorten.csv"
NASDAQ_100_STOCK_TICKER_FILE = "nasdaq100_stock_tickers_shorten.csv"
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
    r"What is your best estimate of the probability that the U.S. economy will enter a recession in the next 12 months.*\-\s*explanation$",
    r"What is your best estimate of the probability that the U.S. economy will enter a recession in the next 12 months.*\-\s*symbol$",
    r"What is your best estimate of the probability that the U.S. economy will enter a recession in the next 12 months.*\-\s*category$",
    r"What is your best estimate of the probability that the U.S. economy will enter a recession in the next 12 months.*\-\s*speculation$",
    r"Considering business conditions in the country as a whole, do you think that during the next 12 months we’ll have good times financially, or bad times, or what.*\-\s*explanation$",
    r"Considering business conditions in the country as a whole, do you think that during the next 12 months we’ll have good times financially, or bad times, or what.*\-\s*symbol$",
    r"Considering business conditions in the country as a whole, do you think that during the next 12 months we’ll have good times financially, or bad times, or what.*\-\s*category$",
    r"Considering business conditions in the country as a whole, do you think that during the next 12 months we’ll have good times financially, or bad times, or what.*\-\s*speculation$",
    r"In the next six months, do you expect business conditions to be better, worse, or the same.*\-\s*explanation$",
    r"In the next six months, do you expect business conditions to be better, worse, or the same.*\-\s*symbol$",
    r"In the next six months, do you expect business conditions to be better, worse, or the same.*\-\s*category$",
    r"In the next six months, do you expect business conditions to be better, worse, or the same.*\-\s*speculation$",
    r"In your view, is the overall investor sentiment currently bearish, neutral, or bullish.*\-\s*explanation$",
    r"In your view, is the overall investor sentiment currently bearish, neutral, or bullish.*\-\s*symbol$",
    r"In your view, is the overall investor sentiment currently bearish, neutral, or bullish.*\-\s*category$",
    r"In your view, is the overall investor sentiment currently bearish, neutral, or bullish.*\-\s*speculation$",
    r"I feel that the direction of the stock market over the next six months will be: Up (Bullish), No Change (Neutral), or Down (Bearish).*\-\s*explanation$",
    r"I feel that the direction of the stock market over the next six months will be: Up (Bullish), No Change (Neutral), or Down (Bearish).*\-\s*symbol$",
    r"I feel that the direction of the stock market over the next six months will be: Up (Bullish), No Change (Neutral), or Down (Bearish).*\-\s*category$",
    r"I feel that the direction of the stock market over the next six months will be: Up (Bullish), No Change (Neutral), or Down (Bearish).*\-\s*speculation$",
    r"In the next 6 months, do you expect U.S. stock market indices to rise, stay about the same, or fall.*\-\s*explanation$",
    r"In the next 6 months, do you expect U.S. stock market indices to rise, stay about the same, or fall.*\-\s*symbol$",
    r"In the next 6 months, do you expect U.S. stock market indices to rise, stay about the same, or fall.*\-\s*category$",
    r"In the next 6 months, do you expect U.S. stock market indices to rise, stay about the same, or fall.*\-\s*speculation$",
    r"In the next 12 months, do you expect interest rates on U.S. bonds to rise, stay about the same, or fall.*\-\s*explanation$",
    r"In the next 12 months, do you expect interest rates on U.S. bonds to rise, stay about the same, or fall.*\-\s*symbol$",
    r"In the next 12 months, do you expect interest rates on U.S. bonds to rise, stay about the same, or fall.*\-\s*category$",
    r"In the next 12 months, do you expect interest rates on U.S. bonds to rise, stay about the same, or fall.*\-\s*speculation$",
    r"Based on your general knowledge, please indicate on a scale of 0 to 100 how you expect the Energy industry to perform over the next 6 months.*\-\s*explanation$",
    r"Based on your general knowledge, please indicate on a scale of 0 to 100 how you expect the Energy industry to perform over the next 6 months.*\-\s*value$",
    r"Based on your general knowledge, please indicate on a scale of 0 to 100 how you expect the Energy industry to perform over the next 6 months.*\-\s*speculation$",
    r"Based on your general knowledge, please indicate on a scale of 0 to 100 how you expect the Materials industry to perform over the next 6 months.*\-\s*explanation$",
    r"Based on your general knowledge, please indicate on a scale of 0 to 100 how you expect the Materials industry to perform over the next 6 months.*\-\s*value$",
    r"Based on your general knowledge, please indicate on a scale of 0 to 100 how you expect the Materials industry to perform over the next 6 months.*\-\s*speculation$",
    r"Based on your general knowledge, please indicate on a scale of 0 to 100 how you expect the Industrials industry to perform over the next 6 months.*\-\s*explanation$",
    r"Based on your general knowledge, please indicate on a scale of 0 to 100 how you expect the Industrials industry to perform over the next 6 months.*\-\s*value$",
    r"Based on your general knowledge, please indicate on a scale of 0 to 100 how you expect the Industrials industry to perform over the next 6 months.*\-\s*speculation$",
    r"Based on your general knowledge, please indicate on a scale of 0 to 100 how you expect the Consumer Discretionary industry to perform over the next 6 months.*\-\s*explanation$",
    r"Based on your general knowledge, please indicate on a scale of 0 to 100 how you expect the Consumer Discretionary industry to perform over the next 6 months.*\-\s*value$",
    r"Based on your general knowledge, please indicate on a scale of 0 to 100 how you expect the Consumer Discretionary industry to perform over the next 6 months.*\-\s*speculation$",
    r"Based on your general knowledge, please indicate on a scale of 0 to 100 how you expect the Consumer Staples industry to perform over the next 6 months.*\-\s*explanation$",
    r"Based on your general knowledge, please indicate on a scale of 0 to 100 how you expect the Consumer Staples industry to perform over the next 6 months.*\-\s*value$",
    r"Based on your general knowledge, please indicate on a scale of 0 to 100 how you expect the Consumer Staples industry to perform over the next 6 months.*\-\s*speculation$",
    r"Based on your general knowledge, please indicate on a scale of 0 to 100 how you expect the Health Care industry to perform over the next 6 months.*\-\s*explanation$",
    r"Based on your general knowledge, please indicate on a scale of 0 to 100 how you expect the Health Care industry to perform over the next 6 months.*\-\s*value$",
    r"Based on your general knowledge, please indicate on a scale of 0 to 100 how you expect the Health Care industry to perform over the next 6 months.*\-\s*speculation$",
    r"Based on your general knowledge, please indicate on a scale of 0 to 100 how you expect the Financials industry to perform over the next 6 months.*\-\s*explanation$",
    r"Based on your general knowledge, please indicate on a scale of 0 to 100 how you expect the Financials industry to perform over the next 6 months.*\-\s*value$",
    r"Based on your general knowledge, please indicate on a scale of 0 to 100 how you expect the Financials industry to perform over the next 6 months.*\-\s*speculation$",
    r"Based on your general knowledge, please indicate on a scale of 0 to 100 how you expect the Information Technology industry to perform over the next 6 months.*\-\s*explanation$",
    r"Based on your general knowledge, please indicate on a scale of 0 to 100 how you expect the Information Technology industry to perform over the next 6 months.*\-\s*value$",
    r"Based on your general knowledge, please indicate on a scale of 0 to 100 how you expect the Information Technology industry to perform over the next 6 months.*\-\s*speculation$",
    r"Based on your general knowledge, please indicate on a scale of 0 to 100 how you expect the Telecommunication Services industry to perform over the next 6 months.*\-\s*explanation$",
    r"Based on your general knowledge, please indicate on a scale of 0 to 100 how you expect the Telecommunication Services industry to perform over the next 6 months.*\-\s*value$",
    r"Based on your general knowledge, please indicate on a scale of 0 to 100 how you expect the Telecommunication Services industry to perform over the next 6 months.*\-\s*speculation$",
    r"Based on your general knowledge, please indicate on a scale of 0 to 100 how you expect the Utilities industry to perform over the next 6 months.*\-\s*explanation$",
    r"Based on your general knowledge, please indicate on a scale of 0 to 100 how you expect the Utilities industry to perform over the next 6 months.*\-\s*value$",
    r"Based on your general knowledge, please indicate on a scale of 0 to 100 how you expect the Utilities industry to perform over the next 6 months.*\-\s*speculation$",
    r"Based on your general knowledge, please indicate on a scale of 0 to 100 how you expect the Real Estate industry to perform over the next 6 months.*\-\s*explanation$",
    r"Based on your general knowledge, please indicate on a scale of 0 to 100 how you expect the Real Estate industry to perform over the next 6 months.*\-\s*value$",
    r"Based on your general knowledge, please indicate on a scale of 0 to 100 how you expect the Real Estate industry to perform over the next 6 months.*\-\s*speculation$",
    r"Compared to the most recently released U.S. unemployment rate, do you expect the next unemployment rate release to be higher, lower, or about the same.*\-\s*explanation$",
    r"Compared to the most recently released U.S. unemployment rate, do you expect the next unemployment rate release to be higher, lower, or about the same.*\-\s*symbol$",
    r"Compared to the most recently released U.S. unemployment rate, do you expect the next unemployment rate release to be higher, lower, or about the same.*\-\s*category$",
    r"Compared to the most recently released U.S. unemployment rate, do you expect the next unemployment rate release to be higher, lower, or about the same.*\-\s*speculation$",
    r"What do you expect the next U.S. unemployment rate to be.*\-\s*explanation$",
    r"What do you expect the next U.S. unemployment rate to be.*\-\s*value$",
    r"What do you expect the next U.S. unemployment rate to be.*\-\s*speculation$",
    r"Compared to the most recently released U.S. inflation number, do you expect the next inflation release to be higher, lower, or about the same.*\-\s*explanation$",
    r"Compared to the most recently released U.S. inflation number, do you expect the next inflation release to be higher, lower, or about the same.*\-\s*symbol$",
    r"Compared to the most recently released U.S. inflation number, do you expect the next inflation release to be higher, lower, or about the same.*\-\s*category$",
    r"Compared to the most recently released U.S. inflation number, do you expect the next inflation release to be higher, lower, or about the same.*\-\s*speculation$",
    r"What do you expect the next U.S. annual inflation number to be.*\-\s*explanation$",
    r"What do you expect the next U.S. annual inflation number to be.*\-\s*value$",
    r"What do you expect the next U.S. annual inflation number to be.*\-\s*speculation$",
    r"At the next Federal Open Market Committee (FOMC) meeting, do you expect the Federal Reserve to lower interest rates, raise them, or leave them unchanged.*\-\s*explanation$",
    r"At the next Federal Open Market Committee (FOMC) meeting, do you expect the Federal Reserve to lower interest rates, raise them, or leave them unchanged.*\-\s*symbol$",
    r"At the next Federal Open Market Committee (FOMC) meeting, do you expect the Federal Reserve to lower interest rates, raise them, or leave them unchanged.*\-\s*category$",
    r"At the next Federal Open Market Committee (FOMC) meeting, do you expect the Federal Reserve to lower interest rates, raise them, or leave them unchanged.*\-\s*speculation$",
    r"What do you expect the target interest rate to be immediately after the next Federal Open Market Committee (FOMC) meeting.*\-\s*explanation$",
    r"What do you expect the target interest rate to be immediately after the next Federal Open Market Committee (FOMC) meeting.*\-\s*value$",
    r"What do you expect the target interest rate to be immediately after the next Federal Open Market Committee (FOMC) meeting.*\-\s*speculation$",
    r"Over the next five trading days, relative to today, do you expect the S&P 500 to be higher, lower, or about the same.*\-\s*explanation$",
    r"Over the next five trading days, relative to today, do you expect the S&P 500 to be higher, lower, or about the same.*\-\s*symbol$",
    r"Over the next five trading days, relative to today, do you expect the S&P 500 to be higher, lower, or about the same.*\-\s*category$",
    r"Over the next five trading days, relative to today, do you expect the S&P 500 to be higher, lower, or about the same.*\-\s*speculation$",
]

# Load Russell 4000 stock tickers
russell4000_stock_tickers = pd.read_csv(
    os.path.join(base_dir, "../config", RUSSELL_4000_STOCK_TICKER_FILE)
)
russell4000_stock_tickers.drop_duplicates(subset=["TICKER"], inplace=True)
russell4000_stock_tickers["combined_ticker"] = russell4000_stock_tickers.apply(
    lambda stock_row: f"{stock_row['COMNAM']} ({stock_row['TICKER']})", axis=1
)
russell4000_stock_ticker_list = russell4000_stock_tickers["combined_ticker"].to_list()
FINFLUENCER_DAILY_STOCK_PICK_REGEX_PATTERNS = [
    r"top[-\s]+conviction BUY.*\-\s*explanation$",
    r"top[-\s]+conviction BUY.*\-\s*stock ticker$",
    r"top[-\s]+conviction BUY.*\-\s*speculation$",
    r"top[-\s]+conviction SELL.*\-\s*explanation$",
    r"top[-\s]+conviction SELL.*\-\s*stock ticker$",
    r"top[-\s]+conviction SELL.*\-\s*speculation$",
]
for stock_ticker in russell4000_stock_tickers["TICKER"].to_list():
    FINFLUENCER_DAILY_STOCK_PICK_REGEX_PATTERNS.extend(
        [
            rf"\({stock_ticker}\).*\-\s*explanation$",
            rf"\({stock_ticker}\).*\-\s*recommendation$",
            rf"\({stock_ticker}\).*\-\s*confidence$",
            rf"\({stock_ticker}\).*\-\s*speculation$",
            rf"\({stock_ticker}\).*\-\s*expected holding period$",
            rf"\({stock_ticker}\).*\-\s*primary catalyst type$",
        ]
    )

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
    "stock_recommendation_interview_datetime",
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
FILTER_ORIGINAL_PROFILES_TIKTOK = True
ORIGINAL_PROFILES_TIKTOK = [
    "alexisanddean",
    "angelpublishing",
    "austinhankwitz",
    "bloombergbusiness",
    "byhistruth",
    "capital.growth",
    "carsandinvesting",
    "cashor.crash",
    "chrisstockdads",
    "cnbc",
    "cryptomasun",
    "einsteinofwallst",
    "financialboffins",
    "financialtimes",
    "fung.money",
    "gritcapital",
    "humphreytalks",
    "investmattallen",
    "jessicainskip",
    "jonerlichman",
    "joshtheceo",
    "joyeeyang0",
    "lauriewilebrown",
    "lisaremillard",
    "madymills",
    "mainstreetwolf",
    "mattshoss",
    "momentum.official",
    "mywallst",
    "oliviavoz",
    "overkilltrading",
    "overlookedalpha",
    "quiverquant",
    "roxana.maddahi",
    "saccofinancial",
    "sammystockz",
    "stastalksstocks",
    "stocksandsavings",
    "sumitsinvestmenttakes",
    "theeconomist",
    "thefifthperson",
    "therealrickdoyle",
    "thesafeinvestor37",
    "tik.stocks",
    "willy_lebon",
    "yahoofinance",
]


# X-specific configurations
PROJECT_NAME_X = "market-signals-x"
KEYWORD_SEARCH_FILE_X = f"x_keyword_search_{PIPELINE_EXECUTION_DATE}.csv"
PROFILE_METADATA_SEARCH_FILE_X = f"x_profile_metadata_{PIPELINE_EXECUTION_DATE}.csv"
FINFLUENCER_POOL_FILE_X = "x_verified_finfluencer_profiles_sample.csv"
ONBOARDING_RESULTS_FILE_X = f"x_onboarding_results_{PIPELINE_EXECUTION_DATE}.csv"
EXPERT_REFLECTION_FILE_X = f"x_expert_reflection_{PIPELINE_EXECUTION_DATE}.csv"
FINFLUENCER_PROFILE_METADATA_SEARCH_FILE_X = (
    f"x_finfluencer_profile_metadata_{PIPELINE_EXECUTION_DATE}.csv"
)
FINFLUENCER_PROFILE_SEARCH_FILE_X = (
    f"x_finfluencer_profile_search_{PIPELINE_EXECUTION_DATE}.csv"
)
FINFLUENCER_HISTORICAL_PROFILE_SEARCH_FILE_X = (
    f"../x_finfluencer_historical_profile_search.csv"
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
FINFLUENCER_DAILY_STOCK_PICK_FILE_X = (
    f"x_finfluencer_daily_stock_pick_{PIPELINE_EXECUTION_DATE}.csv"
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
FILTER_ORIGINAL_PROFILES_X = True
ORIGINAL_PROFILES_X = [
    "1OptionsTrading",
    "abnormalreturns",
    "AdeptMarket",
    "anymantrading",
    "AswathDamodaran",
    "BezosCrypto",
    "Blackopstocks",
    "BreakoutStocks",
    "BrianStutland",
    "ChandlerTrading",
    "Chariot_Invest",
    "ContrarianShort",
    "CrossStocks",
    "Darkminer71",
    "DataDInvesting",
    "davidmoadel",
    "daytradesignals",
    "DCDOWORK",
    "DevotedDividend",
    "dirtcheapstocks",
    "Dividend_Dr",
    "DKellerCMT",
    "dmdsplyinvestor",
    "DV_Situations",
    "ElliottForecast",
    "GlobalStockPick",
    "gurgavin",
    "hiddensmallcaps",
    "I_Am_The_ICT",
    "ideahive",
    "InvestorsLive",
    "ivanhoff2",
    "Jake__Wujastyk",
    "LindaRaschke",
    "LizAnnSonders",
    "MarketMovesMatt",
    "MasteredTrader",
    "MichaelGoodwell",
    "OMillionaires",
    "OnlyOTrades",
    "OptionAlpha",
    "optionscjp",
    "OptionsDepth",
    "OptionsHawk",
    "OptiontradinIQ",
    "PelosiTracker_",
    "profitly",
    "RealDayTrading",
    "ReturnsJourney",
    "SeekingAlpha",
    "Selling4Premium",
    "sentimentrader",
    "seth_fin",
    "SethCL",
    "StockMKTNewz",
    "StockOptionCole",
    "StocksOnSpaces",
    "StocksToTrade",
    "Stocktwits",
    "SuburbanDrone",
    "SwaggyStocks",
    "TAftermath2020",
    "TheAlphaThought",
    "TheStalwart",
    "TheTradingTank",
    "timothysykes",
    "traderstewie",
    "TradesTrey",
    "tradewithprof",
    "Trading0secrets",
    "TradingLounge",
    "TradingwithFun1",
    "TrendSpider",
    "TSXtrad3r",
    "ukarlewitz",
    "value_invest12",
    "ValueStockGeek",
    "ValueWolf",
    "VertiCallAlgo",
    "VROStocks",
    "vsourbh",
    "ZacksResearch",
    "zerohedge",
]
DAILY_STOCK_PICK_PROFILES_X = [
    "Tickeron",
    "strengthPlan",
    "ClassicRoy",
    "StockMarketMcro",
    "Michael34952",
    "BestTrader01",
    "DollarCostAvg",
    "GrindeOptions",
    "ValueSense_io",
    "SteveDJacobs",
    "BullMarketBoss",
    "twinsight_x",
    "WealthCoachMak",
    "WOLF_Financial",
    "MikeLongTerm",
    "solidintel_x",
    "King0ftheCharts",
    "israil_4life",
    "ChartingProdigy",
    "QualityInvest5",
    "EBUYUKARSLAN",
    "danshep55",
    "InvestmentGuru_",
    "TerpsTrader1",
    "TopStockAlerts1",
    "wallstengine",
    "DividendDynasty",
    "OpenOutcrier",
    "AnthonySandford",
    "CuriousPejjy",
    "BradMunchen",
    "MentoviaX",
    "tradealgo_",
    "spluscollective",
    "iamtomnash",
    "C_S_Skeptic",
    "Mr_Derivatives",
    "yasutaketin",
    "BenBSP",
    "commonsenseplay",
    "TraderJonesy",
    "Pharmdca",
    "TradingOutpost",
    "SchwabNetwork",
    "ChartMill",
    "EliteOptions2",
    "marketwirenews",
    "StockSavvyShay",
    "HedgieMarkets",
    "TheTranscript_",
    "DrStoxx",
    "TalkMarkets",
    "TheValueist",
    "dewmboom",
    "SpartanTrading",
    "alshfaw",
    "nikoliasgoninus",
    "epictrades1",
    "EdwardCoronaUSA",
    "SebastinPatron3",
    "CenterPointSec",
    "Options_Sandy",
    "YahooFinance",
    "sspencer_smb",
    "GDXTrader",
    "amitisinvesting",
    "YodaStockInvest",
    "thexcapitalist",
    "Street_Insider",
    "FinancewithIzzy",
    "Corgi4joy",
    "WallStDiaries",
    "equitydd",
    "CNBC",
    "KobeissiLetter",
    "BrandonVanZee",
    "BluthCapital",
    "SoJustFollowMe",
    "CarolKentSpeaks",
    "CNBCi",
    "cfromhertz",
    "danielisdizzy",
    "Benzinga",
    "tradertvneal",
    "SpecialSitsNews",
    "Akaletico",
    "wealthmatica",
    "momoblog0214",
    "MarcJacksonLA",
    "TheWuhanClan",
    "M44_1RJ",
    "neilksethi",
    "CNBCtech",
    "niccruzpatane",
    "Investingcom",
    "techradar",
    "marketsday",
    "samsolid57",
    "investwithsheng",
    "NolanGouveiapG",
]

import os, re
import pandas as pd
from dotenv import load_dotenv

load_dotenv(dotenv_path="ai_population/config/.env")

base_dir = os.path.dirname(os.path.abspath(__file__))

PROJECT_NAME_X = "joint-llm-swiss-x"
NUM_POSTS_PER_PROFILE = 250
PIPELINE_EXECUTION_DATE = "politicians-validation"
PROFILE_SEARCH_START_DATE = "01-01-2025"  # MM-DD-YYYY format
PROFILE_SEARCH_END_DATE = "09-01-2025"  # MM-DD-YYYY format

# Politician-Related Config
POLITICIAN_PROFILE_METADATA_SEARCH_FILE_X = (
    f"x_jointllm_politician_profile_metadata_{PIPELINE_EXECUTION_DATE}.csv"
)
POLITICIAN_PROFILE_SEARCH_FILE_X = (
    f"x_jointllm_politician_profile_search_{PIPELINE_EXECUTION_DATE}.csv"
)
POLITICIAN_POOL_FILE_X = f"x_jointllm_politician_pool.csv"
LOCAL_POLITICIAN_PROFILE_METADATA_FILE = (
    f"x_jointllm_politician_local_profile_metadata_{PIPELINE_EXECUTION_DATE}.csv"
)
LOCAL_POLITICIAN_PROFILE_POST_FILE = (
    f"x_jointllm_politician_local_profile_search_{PIPELINE_EXECUTION_DATE}.csv"
)
POLITICIAN_POST_DEMOGRAPHIC_INTERVIEW_FILE_X = (
    f"x_jointllm_politician_post_demographic_interview_{PIPELINE_EXECUTION_DATE}.csv"
)
POLITICIAN_POST_VOTING_INTERVIEW_FILE_X = (
    f"x_jointllm_politician_post_voting_interview_{PIPELINE_EXECUTION_DATE}.csv"
)

DEMOGRAPHIC_INTERVIEW_REGEX_PATTERNS = [
    r"^PERSON LIVING IN SWITZERLAND.*\-\s*explanation$",
    r"^PERSON LIVING IN SWITZERLAND.*\-\s*symbol$",
    r"^PERSON LIVING IN SWITZERLAND.*\-\s*category$",
    r"^PERSON LIVING IN SWITZERLAND.*\-\s*speculation$",
    r"^REGION.*\-\s*explanation$",
    r"^REGION.*\-\s*symbol$",
    r"^REGION.*\-\s*category$",
    r"^REGION.*\-\s*speculation$",
    r"^AGE.*\-\s*explanation$",
    r"^AGE.*\-\s*symbol$",
    r"^AGE.*\-\s*category$",
    r"^AGE.*\-\s*speculation$",
    r"^GENDER.*\-\s*explanation$",
    r"^GENDER.*\-\s*symbol$",
    r"^GENDER.*\-\s*category$",
    r"^GENDER.*\-\s*speculation$",
    r"^PERSONAL NET WORTH.*\-\s*explanation$",
    r"^PERSONAL NET WORTH.*\-\s*symbol$",
    r"^PERSONAL NET WORTH.*\-\s*category$",
    r"^PERSONAL NET WORTH.*\-\s*speculation$",
    r"^MARITAL STATUS.*\-\s*explanation$",
    r"^MARITAL STATUS.*\-\s*symbol$",
    r"^MARITAL STATUS.*\-\s*category$",
    r"^MARITAL STATUS.*\-\s*speculation$",
    r"^HIGHEST LEVEL OF EDUCATION.*\-\s*explanation$",
    r"^HIGHEST LEVEL OF EDUCATION.*\-\s*symbol$",
    r"^HIGHEST LEVEL OF EDUCATION.*\-\s*category$",
    r"^HIGHEST LEVEL OF EDUCATION.*\-\s*speculation$",
    r"^POLITICAL PARTY.*\-\s*explanation$",
    r"^POLITICAL PARTY.*\-\s*symbol$",
    r"^POLITICAL PARTY.*\-\s*category$",
    r"^POLITICAL PARTY.*\-\s*speculation$",
]

politician_election_info = pd.read_excel(
    os.path.join(
        base_dir, "../data/joint-llm-swiss-x/politician_election_info_en.xlsx"
    ),
    sheet_name="Tabellenblatt1",
)
politician_election_info = politician_election_info.fillna("")
_suffixes = ["explanation", "symbol", "category", "speculation"]
VOTING_PREFERENCE_INTERVIEW_REGEX_PATTERNS = [
    rf"^VOTE {re.escape(str(pid))}.*\-\s*{s}$"
    for pid in politician_election_info["id"].tolist()
    for s in _suffixes
]

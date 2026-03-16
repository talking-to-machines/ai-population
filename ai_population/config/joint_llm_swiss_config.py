import os, re
import pandas as pd
from dotenv import load_dotenv

load_dotenv(dotenv_path="ai_population/config/.env")

base_dir = os.path.dirname(os.path.abspath(__file__))

PROJECT_NAME = "joint-llm-swiss"
NUM_POSTS_PER_PROFILE = 500  # 250
NUM_POSTS_PER_PROFILE_FROM_KEYWORD_SEARCH = 20
NUM_POSTS_PER_KEYWORD = 100
POLITICIAN_PIPELINE = "politicians-validation"
VOTER_PIPELINE_X = "voters-x"
VOTER_PIPELINE_TIKTOK = "voters-tiktok"
PROFILE_SEARCH_START_DATE = "01-01-2024"  # MM-DD-YYYY format
PROFILE_SEARCH_END_DATE = "06-01-2025"  # MM-DD-YYYY format

# Voter-Related Config
VOTER_KEYWORD_SEARCH_FILE_X = f"x_jointllm_voter_keyword_search_{VOTER_PIPELINE_X}.csv"
VOTER_KEYWORD_SEARCH_FILE_TIKTOK = (
    f"tiktok_jointllm_voter_keyword_search_{VOTER_PIPELINE_TIKTOK}.csv"
)
VOTER_KEYWORD_PROFILE_METADATA_FILE_X = (
    f"x_jointllm_voter_keyword_profile_metadata_{VOTER_PIPELINE_X}.csv"
)
VOTER_KEYWORD_PROFILE_METADATA_FILE_TIKTOK = (
    f"tiktok_jointllm_voter_keyword_profile_metadata_{VOTER_PIPELINE_TIKTOK}.csv"
)
VOTER_KEYWORD_PROFILE_POSTS_FILE_X = (
    f"x_jointllm_voter_keyword_profile_posts_{VOTER_PIPELINE_X}.csv"
)
VOTER_KEYWORD_PROFILE_POSTS_FILE_TIKTOK = (
    f"tiktok_jointllm_voter_keyword_profile_posts_{VOTER_PIPELINE_TIKTOK}.csv"
)
VOTER_ENTITY_GEOGRAPHIC_EXCLUSION_CRITERIA_FILE_X = (
    f"x_jointllm_voter_entity_geographic_exclusion_criteria_{VOTER_PIPELINE_X}.csv"
)
VOTER_ENTITY_GEOGRAPHIC_EXCLUSION_CRITERIA_FILE_TIKTOK = f"tiktok_jointllm_voter_entity_geographic_exclusion_criteria_{VOTER_PIPELINE_TIKTOK}.csv"
VOTER_QUOTA_INCLUSION_CRITERIA_FILE_X = (
    f"x_jointllm_voter_quota_inclusion_criteria_{VOTER_PIPELINE_X}.csv"
)
VOTER_QUOTA_INCLUSION_CRITERIA_FILE_TIKTOK = (
    f"tiktok_jointllm_voter_quota_inclusion_criteria_{VOTER_PIPELINE_TIKTOK}.csv"
)
VOTER_ELIGIBLE_PROFILE_SEARCH_FILE_X = (
    f"x_jointllm_voter_eligible_profile_search_{VOTER_PIPELINE_X}.csv"
)
VOTER_ELIGIBLE_PROFILE_SEARCH_FILE_TIKTOK = (
    f"tiktok_jointllm_voter_eligible_profile_search_{VOTER_PIPELINE_TIKTOK}.csv"
)
VOTER_TARGET_STRATIFICATION_FRAME_X = (
    f"x_jointllm_voter_target_stratification_frame_{VOTER_PIPELINE_X}.csv"
)
VOTER_CURRENT_STRATIFICATION_FRAME_X = (
    f"x_jointllm_voter_current_stratification_frame_{VOTER_PIPELINE_X}.csv"
)
VOTER_TARGET_STRATIFICATION_FRAME_TIKTOK = (
    f"tiktok_jointllm_voter_target_stratification_frame_{VOTER_PIPELINE_TIKTOK}.csv"
)
VOTER_CURRENT_STRATIFICATION_FRAME_TIKTOK = (
    f"tiktok_jointllm_voter_current_stratification_frame_{VOTER_PIPELINE_TIKTOK}.csv"
)
VOTER_DIGITAL_POLLING_FILE_X = (
    f"x_jointllm_voter_digital_polling_{VOTER_PIPELINE_X}.csv"
)
VOTER_DIGITAL_POLLING_FILE_TIKTOK = (
    f"tiktok_jointllm_voter_digital_polling_{VOTER_PIPELINE_TIKTOK}.csv"
)
VOTER_SEARCH_TERMS_X = [
    "Bundesrat",
    "Nationalrat",
    "Ständerat",
    "SVP",
    "SPS",
    "FDP",
    "Grüne",
    "GLP",
    "Mitte",
    "Wahlen2023",
    "Nationalratswahlen",
    "Abstimmung",
    "Volksabstimmung",
    "Referendum",
    "Initiative",
    "Neutralität",
    "SchweizerNeutralität",
    "Bilateralen",
    "Asylpolitik",
    "Klimapolitik",
    "Conseilfédéral",
    "Conseilnational",
    "ConseildesÉtats",
    "UDC",
    "PS",
    "PLR",
    "Verts",
    "PVL",
    "Centre",
    "élections2023",
    "électionsfédérales",
    "votation",
    "votations",
    "référendum",
    "initiative",
    "neutralité",
    "neutralitésuisse",
    "bilatérales",
    "politiqued’asile",
    "politiqueclimatique",
    "Consigliodefederale",
    "Consiglionazionale",
    "ConsigliodegliStati",
    "UDC",
    "PS",
    "PLR",
    "Verdi",
    "VerdiLiberali",
    "Centro",
    "elezioni2023",
    "elezionifederali",
    "votazione",
    "votazioni",
    "referendum",
    "iniziativa",
    "neutralità",
    "neutralitàsvizzera",
    "bilaterali",
    "politicad’asilo",
    "politicaclimatica",
]
VOTER_SEARCH_TERMS_TIKTOK = [
    "bundesrat",
    "nationalrat",
    "ständerat",
    "svp",
    "sps",
    "fdp",
    "grüne",
    "glp",
    "mitte",
    "wahlen2023",
    "nationalratswahlen",
    "abstimmung",
    "volksabstimmung",
    "referendum",
    "initiative",
    "neutralität",
    "schweizerneutralität",
    "bilateralen",
    "asylpolitik",
    "klimapolitik",
    "conseilfédéral",
    "conseilnational",
    "conseildesétats",
    "udc",
    "ps",
    "plr",
    "verts",
    "pvl",
    "centre",
    "élections2023",
    "électionsfédérales",
    "votation",
    "votations",
    "référendum",
    "initiative",
    "neutralité",
    "neutralitésuisse",
    "bilatérales",
    "politiquedasile",
    "politiqueclimatique",
    "consigliodefederale",
    "consiglionazionale",
    "consigliodeglistati",
    "udc",
    "ps",
    "plr",
    "verdi",
    "verdiliberati",
    "centro",
    "elezioni2023",
    "elezionifederali",
    "votazione",
    "votazioni",
    "referendum",
    "iniziativa",
    "neutralità",
    "neutralitàsvizzera",
    "bilaterali",
    "politicadasilo",
    "politicaclimatica",
]
VOTER_ENTITY_GEOGRAPHIC_INCLUSION_REGEX_PATTERNS = [
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
    r"^HOUSEHOLD_INCOME.*\-\s*explanation$",
    r"^HOUSEHOLD_INCOME.*\-\s*symbol$",
    r"^HOUSEHOLD_INCOME.*\-\s*category$",
    r"^HOUSEHOLD_INCOME.*\-\s*speculation$",
    r"^CITIZENSHIP.*\-\s*explanation$",
    r"^CITIZENSHIP.*\-\s*symbol$",
    r"^CITIZENSHIP.*\-\s*category$",
    r"^CITIZENSHIP.*\-\s*speculation$",
    r"^ENTITY.*\-\s*explanation$",
    r"^ENTITY.*\-\s*symbol$",
    r"^ENTITY.*\-\s*category$",
    r"^ENTITY.*\-\s*speculation$",
    r"^EDUCATION.*\-\s*explanation$",
    r"^EDUCATION.*\-\s*symbol$",
    r"^EDUCATION.*\-\s*category$",
    r"^EDUCATION.*\-\s*speculation$",
    r"^PARTY_MEMBER.*\-\s*explanation$",
    r"^PARTY_MEMBER.*\-\s*symbol$",
    r"^PARTY_MEMBER.*\-\s*category$",
    r"^PARTY_MEMBER.*\-\s*speculation$",
]
politician_election_info = pd.read_excel(
    os.path.join(base_dir, "../data/joint-llm-swiss/politician_election_info_en.xlsx"),
    sheet_name="Tabellenblatt1",
)
politician_election_info = politician_election_info.fillna("")
_suffixes = ["explanation", "symbol", "category", "speculation"]
DIGITAL_POLLING_REGEX_PATTERNS = [
    rf"^VOTE {re.escape(str(pid))}.*\-\s*{s}$"
    for pid in politician_election_info["id"].tolist()
    for s in _suffixes
]

# Politician-Related Config
POLITICIAN_POOL_FILE_X = f"x_jointllm_politician_pool.csv"
POLITICIAN_POOL_FILE_TIKTOK = f"tiktok_jointllm_politician_pool.csv"
POLITICIAN_PROFILE_METADATA_SEARCH_FILE_X = (
    f"x_jointllm_politician_profile_metadata_{POLITICIAN_PIPELINE}.csv"
)
POLITICIAN_PROFILE_SEARCH_FILE_X = (
    f"x_jointllm_politician_profile_search_{POLITICIAN_PIPELINE}.csv"
)
POLITICIAN_PROFILE_METADATA_SEARCH_FILE_TIKTOK = (
    f"tiktok_jointllm_politician_profile_metadata_{POLITICIAN_PIPELINE}.csv"
)
POLITICIAN_PROFILE_SEARCH_FILE_TIKTOK = (
    f"tiktok_jointllm_politician_profile_search_{POLITICIAN_PIPELINE}.csv"
)
LOCAL_POLITICIAN_PROFILE_METADATA_FILE_X = (
    f"x_jointllm_politician_local_profile_metadata_{POLITICIAN_PIPELINE}.csv"
)
LOCAL_POLITICIAN_PROFILE_POST_FILE_X = (
    f"x_jointllm_politician_local_profile_search_{POLITICIAN_PIPELINE}.csv"
)
LOCAL_POLITICIAN_PROFILE_METADATA_FILE_TIKTOK = (
    f"tiktok_jointllm_politician_local_profile_metadata_{POLITICIAN_PIPELINE}.csv"
)
LOCAL_POLITICIAN_PROFILE_POST_FILE_TIKTOK = (
    f"tiktok_jointllm_politician_local_profile_search_{POLITICIAN_PIPELINE}.csv"
)
POLITICIAN_POST_DEMOGRAPHIC_INTERVIEW_FILE = (
    f"jointllm_politician_post_demographic_interview_{POLITICIAN_PIPELINE}.csv"
)
POLITICIAN_POST_DIGITAL_POLLING_INTERVIEW_FILE = (
    f"jointllm_politician_post_digital_polling_interview_{POLITICIAN_PIPELINE}.csv"
)

POLITICIAN_DEMOGRAPHIC_INTERVIEW_REGEX_PATTERNS = [
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
    r"^HOUSEHOLD_INCOME.*\-\s*explanation$",
    r"^HOUSEHOLD_INCOME.*\-\s*symbol$",
    r"^HOUSEHOLD_INCOME.*\-\s*category$",
    r"^HOUSEHOLD_INCOME.*\-\s*speculation$",
    r"^CITIZENSHIP.*\-\s*explanation$",
    r"^CITIZENSHIP.*\-\s*symbol$",
    r"^CITIZENSHIP.*\-\s*category$",
    r"^CITIZENSHIP.*\-\s*speculation$",
    r"^EDUCATION.*\-\s*explanation$",
    r"^EDUCATION.*\-\s*symbol$",
    r"^EDUCATION.*\-\s*category$",
    r"^EDUCATION.*\-\s*speculation$",
    r"^PARTY_MEMBER.*\-\s*explanation$",
    r"^PARTY_MEMBER.*\-\s*symbol$",
    r"^PARTY_MEMBER.*\-\s*category$",
    r"^PARTY_MEMBER.*\-\s*speculation$",
]

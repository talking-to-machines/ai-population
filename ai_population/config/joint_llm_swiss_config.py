import os, re
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

load_dotenv(dotenv_path="ai_population/config/.env")

base_dir = os.path.dirname(os.path.abspath(__file__))

PROJECT_NAME = "joint-llm-swiss"
NUM_POSTS_PER_PROFILE = 350  # 250
NUM_POSTS_PER_PROFILE_FROM_KEYWORD_SEARCH = 20
MAX_NUM_POSTS_PER_KEYWORD = 100
POLITICIAN_PIPELINE = "politicians-validation-round2"
VOTER_PIPELINE_X = "voters-x"
VOTER_PIPELINE_TIKTOK = "voters-tiktok"
PROFILE_SEARCH_START_DATE = "01-01-2024"  # MM-DD-YYYY format
PROFILE_SEARCH_END_DATE = "06-01-2025"  # MM-DD-YYYY format
PROFILE_SEARCH_TODAY = datetime.today().strftime("%m-%d-%Y")

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
    "Schweizer Bundesrat",
    "Bundesrat Schweiz",
    "Bundesrätin Schweiz",
    "Nationalrat Schweiz",
    "Nationalrätin Schweiz",
    "Ständerat Schweiz",
    "Ständerätin Schweiz",
    "Bundeshaus Bern",
    "Abstimmungssonntag",
    "Volksabstimmung Schweiz",
    "Volksinitiative Schweiz",
    "Schweizer Referendum",
    "Schweizer Initiative",
    "direkte Demokratie Schweiz",
    "Föderalismus Schweiz",
    "Eidgenossenschaft",
    "Schweizerische Volkspartei",
    "SVP Schweiz",
    "Sozialdemokratische Partei der Schweiz",
    "SP Schweiz",
    "FDP Schweiz",
    "FDP.Die Liberalen",
    "Grüne Schweiz",
    "GLP Schweiz",
    "Grünliberale Schweiz",
    "Die Mitte Schweiz",
    "EVP Schweiz",
    "Nationalratswahlen Schweiz",
    "Schweizer Neutralität",
    "Bilaterale Abkommen Schweiz",
    "Bilaterale III Schweiz",
    "Asylpolitik Schweiz",
    "Klimapolitik Schweiz",
    "Conseil fédéral suisse",
    "Conseillère fédérale suisse",
    "Conseil national suisse",
    "Conseillère nationale suisse",
    "Conseil des États suisse",
    "Assemblée fédérale suisse",
    "Palais fédéral suisse",
    "votation suisse",
    "initiative populaire suisse",
    "référendum suisse",
    "démocratie directe suisse",
    "dimanche de vote suisse",
    "Confédération suisse",
    "UDC Suisse",
    "Parti socialiste suisse",
    "PS Suisse",
    "PLR Suisse",
    "Parti libéral-radical suisse",
    "Verts Suisse",
    "PVL Suisse",
    "Le Centre Suisse",
    "élections fédérales suisse",
    "neutralité suisse",
    "accords bilatéraux suisse",
    "accords bilatéraux III suisse",
    "politique d'asile suisse",
    "politique climatique suisse",
    "Consiglio federale svizzero",
    "Consigliera federale svizzera",
    "Consiglio nazionale svizzero",
    "Consiglio degli Stati svizzero",
    "Assemblea federale svizzera",
    "Palazzo federale svizzero",
    "votazione svizzera",
    "iniziativa popolare svizzera",
    "referendum svizzero",
    "democrazia diretta Svizzera",
    "Confederazione Svizzera",
    "UDC Svizzera",
    "Partito socialista svizzero",
    "PS Svizzera",
    "PLR Svizzera",
    "Partito liberale-radicale svizzero",
    "Verdi Svizzera",
    "PVL Svizzera",
    "Il Centro Svizzera",
    "Lega dei Ticinesi",
    "elezioni federali svizzera",
    "neutralità svizzera",
    "accordi bilaterali svizzera",
    "accordi bilaterali III svizzera",
    "politica d'asilo svizzera",
    "politica climatica svizzera",
]
VOTER_SEARCH_TERMS_TIKTOK = [
    "SchweizerPolitik",
    "CHPolitik",
    "SchweizPolitik",
    "SchweizerBundesrat",
    "BundesratSchweiz",
    "Bundeshaus",
    "Nationalrat",
    "Ständerat",
    "Abstimmungssonntag",
    "AbstimmungSchweiz",
    "Volksabstimmung",
    "Volksinitiative",
    "SVPSchweiz",
    "SPSchweiz",
    "FDPCH",
    "GrüneSchweiz",
    "GLPSchweiz",
    "DieMitteSchweiz",
    "EVPSchweiz",
    "SchweizerNeutralität",
    "BilateraleSchweiz",
    "BilateraleIII",
    "AsylpolitikCH",
    "KlimapolitikSchweiz",
    "DirekteDemokratieCH",
    "ReferendumCH",
    "WahlenCH",
    "Eidgenossenschaft",
    "PolitiqueSuisse",
    "SuissePolitique",
    "ConseilFederalSuisse",
    "ConseilNationalSuisse",
    "UDCSuisse",
    "PSSuisse",
    "PLRSuisse",
    "VertsSuisse",
    "PVLSuisse",
    "LeCentreSuisse",
    "VotationSuisse",
    "ReferendumSuisse",
    "InitiativePopulaire",
    "NeutraliteSuisse",
    "AccordsBilateraux",
    "PolitiqueDAsile",
    "PolitiqueClimatiqueCH",
    "DemocratieDirect",
    "PoliticaSvizzera",
    "SvizzeraPolitica",
    "ConsiglioFederaleSvizzero",
    "ConsiglioNazionaleSvizzero",
    "UDCSvizzera",
    "PSSvizzera",
    "PLRSvizzera",
    "VerdiSvizzera",
    "CentroSvizzera",
    "LegaDeiTicinesi",
    "VotazioneSvizzera",
    "ReferendumSvizzero",
    "IniziativaPopolare",
    "NeutralitaSvizzera",
    "AccordiBilaterali",
    "DemocratiaDiretta",
]
VOTER_DEMOGRAPHIC_INTERVIEW_REGEX_PATTERNS = [
    r"^PERSON_LIVING_IN_SWITZERLAND.*\-\s*explanation$",
    r"^PERSON_LIVING_IN_SWITZERLAND.*\-\s*symbol$",
    r"^PERSON_LIVING_IN_SWITZERLAND.*\-\s*category$",
    r"^PERSON_LIVING_IN_SWITZERLAND.*\-\s*speculation$",
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
    r"^TURNOUT_FEDERAL.*\-\s*explanation$",
    r"^TURNOUT_FEDERAL.*\-\s*symbol$",
    r"^TURNOUT_FEDERAL.*\-\s*category$",
    r"^TURNOUT_FEDERAL.*\-\s*speculation$",
    r"^VOTE_FEDERAL.*\-\s*explanation$",
    r"^VOTE_FEDERAL.*\-\s*symbol$",
    r"^VOTE_FEDERAL.*\-\s*category$",
    r"^VOTE_FEDERAL.*\-\s*speculation$",
]
referendums = pd.read_excel(
    os.path.join(base_dir, "../data/joint-llm-swiss/referendums.xlsx"),
    sheet_name="referendums",
)
referendums = referendums.fillna("")
_suffixes = ["explanation", "symbol", "category", "speculation"]
_prefixes = ["VOTE", "TURNOUT"]
DIGITAL_POLLING_REGEX_PATTERNS = [
    rf"^{prefix}_{re.escape(str(pid))}.*\-\s*{s}$"
    for prefix in _prefixes
    for pid in referendums["business_number"].tolist()
    for s in _suffixes
]
DIGITAL_POLLING_REGEX_PATTERNS += [
    r"^TURNOUT_FEDERAL.*\-\s*explanation$",
    r"^TURNOUT_FEDERAL.*\-\s*symbol$",
    r"^TURNOUT_FEDERAL.*\-\s*category$",
    r"^TURNOUT_FEDERAL.*\-\s*speculation$",
    r"^VOTE_FEDERAL.*\-\s*explanation$",
    r"^VOTE_FEDERAL.*\-\s*symbol$",
    r"^VOTE_FEDERAL.*\-\s*category$",
    r"^VOTE_FEDERAL.*\-\s*speculation$",
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
    r"^PERSON_LIVING_IN_SWITZERLAND.*\-\s*explanation$",
    r"^PERSON_LIVING_IN_SWITZERLAND.*\-\s*symbol$",
    r"^PERSON_LIVING_IN_SWITZERLAND.*\-\s*category$",
    r"^PERSON_LIVING_IN_SWITZERLAND.*\-\s*speculation$",
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
    r"^EDUCATION.*\-\s*explanation$",
    r"^EDUCATION.*\-\s*symbol$",
    r"^EDUCATION.*\-\s*category$",
    r"^EDUCATION.*\-\s*speculation$",
    r"^PARTY_MEMBER.*\-\s*explanation$",
    r"^PARTY_MEMBER.*\-\s*symbol$",
    r"^PARTY_MEMBER.*\-\s*category$",
    r"^PARTY_MEMBER.*\-\s*speculation$",
]

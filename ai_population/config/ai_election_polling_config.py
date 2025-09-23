import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Common configurations for AI Election Polling
PIPELINE_EXECUTION_DATE = datetime.today().date().strftime("%d-%m-%Y")
NUM_POSTS_PER_KEYWORD = 20
NUM_POSTS_PER_PROFILE = 20
TEMPORAL_INCLUSION_PERIOD = 7  # Profiles polled within the last 7 days will be excluded
PROFILE_SEARCH_START_DATE = "06-23-2025"  # MM-DD-YYYY format
PROFILE_SEARCH_END_DATE = "06-24-2025"  # MM-DD-YYYY format
ENTITY_GEOGRAPHIC_INCLUSION_REGEX_PATTERNS = [
    r"^Is this an account of a real-life existing person, or of another kind of entity.*\-\s*explanation$",
    r"^Is this an account of a real-life existing person, or of another kind of entity.*\-\s*symbol$",
    r"^Is this an account of a real-life existing person, or of another kind of entity.*\-\s*category$",
    r"^Is this an account of a real-life existing person, or of another kind of entity.*\-\s*speculation$",
    r"^Does the user of this.*\-\s*explanation$",
    r"^Does the user of this.*\-\s*symbol$",
    r"^Does the user of this.*\-\s*category$",
    r"^Does the user of this.*\-\s*speculation$",
    r"^If the response to Question 2 is “Yes,” specify the state (province) the user is living in.*\-\s*explanation$",
    r"^If the response to Question 2 is “Yes,” specify the state (province) the user is living in.*\-\s*speculation$",
    r"^If the response to Question 2 is “Yes,” specify the state (province) the user is living in.*\-\s*value$",
]
AI_ELECTION_INTERVIEW_REGEX_PATTERNS = [
    r"^XXX.*\-\s*explanation$",
    r"^XXX.*\-\s*symbol$",
    r"^XXX.*\-\s*category$",
    r"^XXX.*\-\s*speculation$",
    r"^YYY.*\-\s*explanation$",
    r"^YYY.*\-\s*speculation$",
    r"^YYY.*\-\s*value$",
]


# File directories for X (formerly Twitter) AI Election Polling
KEYWORD_SEARCH_FILE_X = f"x_keyword_search_{PIPELINE_EXECUTION_DATE}.csv"
PROFILE_METADATA_SEARCH_FILE_X = (
    f"x_profile_metadata_search_{PIPELINE_EXECUTION_DATE}.csv"
)
TEMPORAL_INCLUSION_CRITERIA_FILE_X = (
    f"x_temporal_inclusion_criteria_{PIPELINE_EXECUTION_DATE}.csv"
)
POLLED_PROFILES_X = f"x_polled_profiles.csv"
NULL_GEOGRAPHY_EXCLUSION_CRITERIA_FILE_X = (
    f"x_null_geography_exclusion_criteria_{PIPELINE_EXECUTION_DATE}.csv"
)
ENTITY_GEOGRAPHIC_INCLUSION_CRITERIA_FILE_X = (
    f"x_entity_geographic_inclusion_criteria_{PIPELINE_EXECUTION_DATE}.csv"
)
QUOTA_INCLUSION_CRITERIA_FILE_X = (
    f"x_quota_inclusion_criteria_{PIPELINE_EXECUTION_DATE}.csv"
)
ELIGIBLE_PROFILE_SEARCH_FILE_X = (
    f"x_eligible_profile_search_{PIPELINE_EXECUTION_DATE}.csv"
)
DIGITAL_POLLING_FILE_X = f"x_digital_polling_{PIPELINE_EXECUTION_DATE}.csv"


# File directories for Tiktok AI Election Polling
KEYWORD_SEARCH_FILE_TIKTOK = f"tiktok_keyword_search_{PIPELINE_EXECUTION_DATE}.csv"
PROFILE_METADATA_SEARCH_FILE_TIKTOK = (
    f"tiktok_profile_metadata_search_{PIPELINE_EXECUTION_DATE}.csv"
)
TEMPORAL_INCLUSION_CRITERIA_FILE_TIKTOK = (
    f"tiktok_temporal_inclusion_criteria_{PIPELINE_EXECUTION_DATE}.csv"
)
POLLED_PROFILES_TIKTOK = f"tiktok_polled_profiles.csv"
NULL_GEOGRAPHY_EXCLUSION_CRITERIA_FILE_TIKTOK = (
    f"tiktok_null_geography_exclusion_criteria_{PIPELINE_EXECUTION_DATE}.csv"
)
ENTITY_GEOGRAPHIC_INCLUSION_CRITERIA_FILE_TIKTOK = (
    f"tiktok_entity_geographic_inclusion_criteria_{PIPELINE_EXECUTION_DATE}.csv"
)
QUOTA_INCLUSION_CRITERIA_FILE_TIKTOK = (
    f"tiktok_quota_inclusion_criteria_{PIPELINE_EXECUTION_DATE}.csv"
)
ELIGIBLE_PROFILE_SEARCH_FILE_TIKTOK = (
    f"tiktok_eligible_profile_search_{PIPELINE_EXECUTION_DATE}.csv"
)
DIGITAL_POLLING_FILE_TIKTOK = f"tiktok_digital_polling_{PIPELINE_EXECUTION_DATE}.csv"


# AI Election Polling (Canada-specific)
PROJECT_NAME_TIKTOK_CANADA = "ai-elections-polling-tiktok-canada"
PROJECT_NAME_X_CANADA = "ai-elections-polling-x-canada"
SEARCH_TERMS_CANADA = []


# AI Election Polling (Chile-specific)
PROJECT_NAME_TIKTOK_CHILE = "ai-elections-polling-tiktok-chile"
PROJECT_NAME_X_CHILE = "ai-elections-polling-x-chile"
SEARCH_TERMS_CHILE = [
    "A la cochiguaga",
    "Apitutado",
    "Arreglín",
    "Chirimoyo",
    "Con la teja pasá",
    "Cortar la colita",
    "Esto muere aquí",
    "Hacer la guagua",
    "Tener santos en la corte",
    "Topón pa dentro",
    "Triangulación",
    "Una mano lava la otra, y las dos lavan la cara",
    "Hacer la vista gorda",
    "Pacogate",
    "Milicogate",
    "Alimentar el pingüino",
    "Arreglín",
    "Cohecho",
    "Soborno",
    "Mojado",
    "Chanchullo",
    "Levantamiento de fondos",
    "Cuoteo",
    "Fraude electoral",
    "Tráfico de influencias",
    "Corrupción",
    "Trucho",
    "Compra de votos",
    "Caso audios",
    "Nepotismo",
    "Amiguismo",
    "Extorsion",
    "Impunidad",
]

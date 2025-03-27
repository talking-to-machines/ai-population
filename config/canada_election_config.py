import os
from dotenv import load_dotenv

load_dotenv()

PROJECT = "canada-elections"
SEARCH_TERMS_FILE = "canada_election_search_terms.txt"
VIDEO_METADATA_FILE = "video_metadata.csv"
PROFILE_METADATA_FILE = "profile_metadata.csv"
POLLED_PROFILES_FILE = "polled_profiles.csv"
TEMPORAL_INCLUSION_PERIOD = 7  # Profiles polled within the last 7 days will be excluded
PROFILE_METADATA_POST_PROFILE_PROMPT_FILE = "profile_metadata_post_profile_prompt.csv"
PROFILE_METADATA_POST_TEMPORAL_INCLUSION_FILE = (
    "profile_metadata_post_temporal_inclusion.csv"
)
PROFILE_METADATA_POST_GEOGRAPHY_EXCLUSION_FILE = (
    "profile_metadata_post_geography_exclusion.csv"
)
PROFILE_METADATA_POST_ENTITY_GEOGRAPHIC_INCLUSION_FILE = (
    "profile_metadata_post_entity_geographic_inclusion.csv"
)
# PROFILES_FILE = "canada_elections_profiles.txt"
# POOL_TYPE = "LATEST"  # LATEST or STORAGE
# SURVEY_FILE = "canada_elections_survey.csv"
# SURVEY_PROGRESS_LOG = "survey_progress_log.pkl"

import os
from dotenv import load_dotenv

load_dotenv()

PROJECT = "chile-elections"
SEARCH_TERMS_FILE = "chile_election_search_terms.txt"
KEYWORD_SEARCH_VIDEO_METADATA_FILE = "keyword_search_video_metadata.csv"
# PROFILE_SEARCH_VIDEO_METADATA_FILE = "profile_search_video_metadata.csv"
KEYWORD_SEARCH_PROFILE_METADATA_FILE = "keyword_search_profile_metadata.csv"
# PROFILE_SEARCH_PROFILE_METADATA_FILE = "profile_search_profile_metadata.csv"
# POLLED_PROFILES_FILE = "polled_profiles.csv"
TEMPORAL_INCLUSION_PERIOD = 7  # Profiles polled within the last 7 days will be excluded
# PROFILE_METADATA_POST_PROFILE_PROMPT_FILE = "profile_metadata_post_profile_prompt.csv"
# PROFILE_METADATA_POST_TEMPORAL_INCLUSION_FILE = (
#     "profile_metadata_post_temporal_inclusion.csv"
# )
# PROFILE_METADATA_POST_GEOGRAPHY_EXCLUSION_FILE = (
#     "profile_metadata_post_geography_exclusion.csv"
# )
# PROFILE_METADATA_POST_ENTITY_GEOGRAPHIC_INCLUSION_FILE = (
#     "profile_metadata_post_entity_geographic_inclusion.csv"
# )
# PROFILE_METADATA_POST_QUOTA_INCLUSION_FILE = "profile_metadata_post_quota_inclusion.csv"
# PROFILE_METADATA_POST_POLLING_FILE = "polling_results.csv"

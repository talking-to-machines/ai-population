import os
import pandas as pd
from tqdm import tqdm

tqdm.pandas()
from config.canada_election_config import (
    PROJECT,
    SEARCH_TERMS_FILE,
    VIDEO_METADATA_FILE,
    PROFILE_METADATA_FILE,
)

from src.utils import build_profile_prompt
from src.keyword_search import perform_keyword_search
from src.video_transcription import perform_video_transcription

base_dir = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    # Step 1: Get Pool
    print("Step 1: Get Pool")
    ## Perform key word search for TikTok videos discussing Canada elections
    print("Performing key word search...")
    perform_keyword_search(
        project_name=PROJECT,
        search_terms_file=SEARCH_TERMS_FILE,
        profile_metadata_file=PROFILE_METADATA_FILE,
        video_metadata_file=VIDEO_METADATA_FILE,
    )
    print()

    ## Perform audio transcription of new videos
    print("Performing audio transcription...")
    perform_video_transcription(
        project_name=PROJECT, video_metadata_file=VIDEO_METADATA_FILE
    )
    print()

    # Step 2: Poll Users
    print("Step 2: Poll Users")
    ## Build user profile prompt
    print("Build user profile prompt...")
    build_profile_prompt(
        project_name=PROJECT,
        profile_metadata_file=PROFILE_METADATA_FILE,
        video_metadata_file=VIDEO_METADATA_FILE,
    )
    print()

    ## Apply temporal inclusion criteria (limit number of survey responses from a single user within a given timeframe)
    print("Apply temporal inclusion criteria...")
    print()

    ## Apply null geogrpahy exclusion criteria (remove profiles without self-reported location information)
    print("Apply null geography exclusion criteria...")
    print()

    ## Apply entity inclusion criteria (exclude profiles that do not belong to an individual (i.e., organisations, bots, etc)
    print("Apply entity inclusion criteria...")
    print()

    ## Apply geographic inclusion criteria (filter out profiles that are unlikely to reside in Level 1 geography (i.e., Canada))
    print("Apply geographic inclusion criteria...")
    print()

    # Apply quota inclusion criteria (performing quota sampling to ensure that the sample is representative of the Canadian population)
    print("Enforcing quota inclusion criteria...")
    print()

    # Perform digital interview on Canada election
    print("Performing digital interview on Canada election...")
    print()

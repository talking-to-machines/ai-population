import os
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm

tqdm.pandas()
from config.canada_election_config import (
    PROJECT,
    SEARCH_TERMS_FILE,
    VIDEO_METADATA_FILE,
    PROFILE_METADATA_FILE,
    SAMPLED_PROFILE_METADATA_FILE,
    POLLED_PROFILES_FILE,
    TEMPORAL_INCLUSION_PERIOD,
)

from src.utils import build_profile_prompt
from src.keyword_search import perform_keyword_search
from src.video_transcription import perform_video_transcription

base_dir = os.path.dirname(os.path.abspath(__file__))


def apply_temporal_inclusion_criteria(
    project_name: str,
    profile_metadata_file: str,
    sampled_profile_metadata_file: str,
    polled_profiles_file: str,
) -> None:
    print("Load profile metadata...")
    profile_metadata = pd.read_csv(
        f"{base_dir}/../data/{project_name}/{profile_metadata_file}"
    )

    print("Exclude profiles that have been polled within the last N days...")
    polled_profiles_file_path = Path(
        f"{base_dir}/../data/{project_name}/{polled_profiles_file}"
    )
    if polled_profiles_file_path.exists():
        polled_profiles = pd.read_csv(polled_profiles_file_path)
        polled_profiles["poll_date"] = pd.to_datetime(polled_profiles["poll_date"])

        # Identify profiles that were polled within the last N days
        recently_polled_profiles = polled_profiles[
            polled_profiles["poll_date"]
            >= datetime.today() - timedelta(days=TEMPORAL_INCLUSION_PERIOD)
        ].reset_index(drop=True)

        # Exclude profiles that were polled within the last N days
        sampled_profile_metadata = profile_metadata[
            ~profile_metadata["profile"].isin(recently_polled_profiles["profile"])
        ]
        sampled_profile_metadata.to_csv(
            f"{base_dir}/../data/{project_name}/{sampled_profile_metadata_file}",
            index=False,
        )

        # Update polled profiles with profiles that will be polled in the current survey iteration
        newly_polled_profiles = sampled_profile_metadata[["profile"]]
        newly_polled_profiles["poll_date"] = datetime.today().date()
        updated_polled_profiles = pd.concat(
            [recently_polled_profiles, newly_polled_profiles], ignore_index=True
        )
        updated_polled_profiles.to_csv(
            f"{base_dir}/../data/{project_name}/{polled_profiles_file}", index=False
        )

    else:  # If no profiles have been polled yet, all existing profiles will be polled
        sampled_profile_metadata = profile_metadata
        sampled_profile_metadata.to_csv(
            f"{base_dir}/../data/{project_name}/{sampled_profile_metadata_file}",
            index=False,
        )

        newly_polled_profiles = sampled_profile_metadata[["profile"]]
        newly_polled_profiles["poll_date"] = datetime.today().date()
        newly_polled_profiles.to_csv(
            f"{base_dir}/../data/{project_name}/{polled_profiles_file}", index=False
        )

    return None


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
    apply_temporal_inclusion_criteria(
        project_name=PROJECT,
        profile_metadata_file=PROFILE_METADATA_FILE,
        sampled_profile_metadata_file=SAMPLED_PROFILE_METADATA_FILE,
        polled_profiles_file=POLLED_PROFILES_FILE,
    )
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

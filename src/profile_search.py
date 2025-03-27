import os
import pandas as pd
from apify_client import ApifyClient
from src.utils import load_text_file, update_video_metadata, update_profile_metadata
from config.base_config import (
    APIFY_API,
    APIFY_ACTOR_ID,
    PROFILE_SEARCH_RESULTS_PER_PAGE,
)
from src.video_transcription import perform_video_transcription


def perform_profile_search(
    project_name: str,
    profile_metadata_file: str,
    video_metadata_file: str,
    profile_list: list = [],
    profile_list_file: str = None,
    perform_audio_transcription: bool = True,
    return_videos: bool = False,
) -> pd.DataFrame:
    # Create the project subfolder within the data folder if it does not exist
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(base_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "data", project_name), exist_ok=True)

    # Define search parameters
    if profile_list_file:
        profile_list = load_text_file(profile_list_file)

    # Initialize the ApifyClient with your API token
    client = ApifyClient(APIFY_API)

    # Prepare the Actor input
    run_input = {
        "excludePinnedPosts": False,
        "profileScrapeSections": ["videos"],
        "profileSorting": "latest",
        "profiles": profile_list,
        "resultsPerPage": PROFILE_SEARCH_RESULTS_PER_PAGE,
        "shouldDownloadCovers": False,
        "shouldDownloadSlideshowImages": False,
        "shouldDownloadSubtitles": False,
        "shouldDownloadVideos": False,
    }

    # Run the Actor and wait for it to finish
    print("Performing profile search using Apify...")
    run = client.actor(APIFY_ACTOR_ID).call(run_input=run_input)

    # Update video metadata store
    print("Updating video metadata...")
    update_video_metadata(
        project_name=project_name,
        video_metadata_file=video_metadata_file,
        client=client,
        run=run,
        profile_search=True,
        filtering_list=profile_list,
    )

    # Update profile metadata store
    print("Updating profile metadata...")
    update_profile_metadata(
        project_name=project_name,
        profile_metadata_file=profile_metadata_file,
        video_metadata_file=video_metadata_file,
    )

    # Perform audio transcription of new videos
    if perform_audio_transcription:
        print("Performing audio transcription...")
        perform_video_transcription(
            project_name=project_name, video_metadata_file=video_metadata_file
        )

    # Extract videos from profile list
    if return_videos:
        updated_video_metadata = pd.read_csv(
            f"{base_dir}/../data/{project_name}/{video_metadata_file}"
        )
        filtered_video_metadata = updated_video_metadata[
            updated_video_metadata["profile"].isin(profile_list)
        ].reset_index(drop=True)

        return filtered_video_metadata

    else:
        return None

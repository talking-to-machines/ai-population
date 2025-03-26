import os
from apify_client import ApifyClient
from src.utils import load_text_file, update_video_metadata, update_profile_metadata
from config.market_signals_config import *


if __name__ == "__main__":
    # Create the project subfolder within the data folder if it does not exist
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(base_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "data", PROJECT), exist_ok=True)

    # Define search parameters
    PROFILES = load_text_file(PROFILES_FILE)

    # Initialize the ApifyClient with your API token
    client = ApifyClient(APIFY_API)

    # Prepare the Actor input
    run_input = {
        "excludePinnedPosts": False,
        "profileScrapeSections": ["videos"],
        "profileSorting": "latest",
        "profiles": PROFILES,
        "resultsPerPage": 25,
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
        project_name=PROJECT,
        video_metadata_file=PROFILESEARCH_VIDEO_METADATA_FILE,
        client=client,
        run=run,
        profile_search=True,
        filtering_list=PROFILES,
    )

    # Update profile metadata store
    print("Updating profile metadata...")
    update_profile_metadata(
        project_name=PROJECT,
        profile_metadata_file=PROFILESEARCH_PROFILE_METADATA_FILE,
        video_metadata_file=PROFILESEARCH_VIDEO_METADATA_FILE,
    )

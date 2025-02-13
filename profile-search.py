import os
from apify_client import ApifyClient
from utils import load_text_file, update_video_metadata, update_profile_metadata
from config import (
    PROJECT,
    APIFY_API,
    APIFY_ACTOR_ID,
    PROFILES_FILE,
    OLDEST_POST_DATE,
    NEWEST_POST_DATE,
)


if __name__ == "__main__":
    # Create the project subfolder within the data folder if it does not exist
    folder_path = os.path.join("data", PROJECT)
    os.makedirs("data", exist_ok=True)
    os.makedirs(folder_path, exist_ok=True)

    # Define search parameters
    PROFILES = load_text_file(PROFILES_FILE)

    # Initialize the ApifyClient with your API token
    client = ApifyClient(APIFY_API)

    # Prepare the Actor input
    run_input = {
        "excludePinnedPosts": False,
        "newestPostDate": NEWEST_POST_DATE,
        "oldestPostDate": OLDEST_POST_DATE,
        "profileScrapeSections": ["videos", "reposts"],
        "profiles": PROFILES,
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
    update_video_metadata(client, run, PROJECT, profile_search=True)

    # Update profile metadata store
    print("Updating profile metadata...")
    update_profile_metadata(PROJECT, profile_search=True)

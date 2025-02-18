import os
from apify_client import ApifyClient
from utils import (
    load_text_file,
    update_video_metadata,
    update_profile_metadata,
    identify_top_influencers,
)
from config import (
    RESULTS_PER_PAGE,
    TOP_N_PROFILES,
    PROJECT,
    APIFY_API,
    APIFY_ACTOR_ID,
    SEARCH_TERMS_FILE,
)


if __name__ == "__main__":
    # Create the project subfolder within the data folder if it does not exist
    folder_path = os.path.join("data", PROJECT)
    os.makedirs("data", exist_ok=True)
    os.makedirs(folder_path, exist_ok=True)

    # Define search parameters
    SEARCH_TERMS = load_text_file(SEARCH_TERMS_FILE)

    # Initialize the ApifyClient with your API token
    client = ApifyClient(APIFY_API)

    # Prepare the Actor input
    run_input = {
        "excludePinnedPosts": False,
        "resultsPerPage": RESULTS_PER_PAGE,
        "searchQueries": SEARCH_TERMS,
        "searchSection": "/video",
        "shouldDownloadCovers": False,
        "shouldDownloadSlideshowImages": False,
        "shouldDownloadSubtitles": False,
        "shouldDownloadVideos": False,
    }

    # Run the Actor and wait for it to finish
    print("Performing key word search using Apify...")
    run = client.actor(APIFY_ACTOR_ID).call(run_input=run_input)

    # Update video metadata store
    print("Updating video metadata...")
    update_video_metadata(
        client, run, profile_search=False, filtering_list=SEARCH_TERMS
    )

    # Update profile metadata store
    print("Updating profile metadata...")
    update_profile_metadata(profile_search=False)

    # Identify top n influencial profiles based on keyword search
    print(
        f"Identifying top {TOP_N_PROFILES} influencial profiles from keyword search..."
    )
    identify_top_influencers(TOP_N_PROFILES)

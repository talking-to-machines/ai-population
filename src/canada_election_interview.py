import os
from tqdm import tqdm

tqdm.pandas()
from config.canada_election_config import (
    PROJECT,
    SEARCH_TERMS_FILE,
    KEYWORDSEARCH_VIDEO_METADATA_FILE,
    KEYWORDSEARCH_PROFILE_METADATA_FILE,
)

# from src.utils import (
#     extract_llm_responses,
#     extract_stock_recommendations,
#     perform_profile_interview,
# )
from src.keyword_search import perform_keyword_search

base_dir = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    # Perform key word search for TikTok videos discussing Canada elections
    print("Performing key word search for videos discussing Canada elections...")
    perform_keyword_search(
        project_name=PROJECT,
        search_terms_file=SEARCH_TERMS_FILE,
        profile_metadata_file=KEYWORDSEARCH_PROFILE_METADATA_FILE,
        video_metadata_file=KEYWORDSEARCH_VIDEO_METADATA_FILE,
    )
    print()

    # Perform audio transcription of new videos
    print("Performing audio transcription of new videos...")
    print()

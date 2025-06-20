# import os
# from apify_client import ApifyClient
# from src.utils import (
#     load_text_file,
#     update_video_metadata,
#     update_profile_metadata,
# )
# from config.base_config import (
#     KEYWORD_SEARCH_RESULTS_PER_PAGE,
#     APIFY_API,
#     APIFY_ACTOR_ID,
# )
# from src.video_transcription import perform_video_transcription


# def perform_keyword_search(
#     project_name: str,
#     search_terms_file: str,
#     profile_metadata_file: str,
#     video_metadata_file: str,
#     perform_audio_transcription: bool = True,
# ) -> None:
#     # # Create the project subfolder within the data folder if it does not exist
#     # base_dir = os.path.dirname(os.path.abspath(__file__))
#     # os.makedirs(os.path.join(base_dir, "../data"), exist_ok=True)
#     # os.makedirs(os.path.join(base_dir, "../data", project_name), exist_ok=True)

#     # # Define search parameters
#     # search_terms = load_text_file(search_terms_file)

#     # # Initialize the ApifyClient with your API token
#     # client = ApifyClient(APIFY_API)

#     # # Prepare the Actor input
#     # run_input = {
#     #     "excludePinnedPosts": False,
#     #     "resultsPerPage": KEYWORD_SEARCH_RESULTS_PER_PAGE,
#     #     "searchQueries": search_terms,
#     #     "searchSection": "/video",
#     #     "shouldDownloadCovers": False,
#     #     "shouldDownloadSlideshowImages": False,
#     #     "shouldDownloadSubtitles": False,
#     #     "shouldDownloadVideos": False,
#     # }

#     # # Run the Actor and wait for it to finish
#     # print("Performing key word search using Apify...")
#     # run = client.actor(APIFY_ACTOR_ID).call(run_input=run_input)

#     # # Update video metadata store
#     # print("Updating video metadata...")
#     # update_video_metadata(
#     #     project_name=project_name,
#     #     video_metadata_file=video_metadata_file,
#     #     client=client,
#     #     run=run,
#     #     profile_search=False,
#     #     filtering_list=search_terms,
#     # )

#     # # Update profile metadata store
#     # print("Updating profile metadata...")
#     # update_profile_metadata(
#     #     project_name=project_name,
#     #     profile_metadata_file=profile_metadata_file,
#     #     video_metadata_file=video_metadata_file,
#     # )

#     # Perform audio transcription of new videos
#     if perform_audio_transcription:
#         print("Performing audio transcription...")
#         perform_video_transcription(
#             project_name=project_name, video_metadata_file=video_metadata_file
#         )

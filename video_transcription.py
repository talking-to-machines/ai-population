import os
import pandas as pd
from utils import download_video, transcribe_videos
from config import PROJECT
from tqdm import tqdm

# Enable the progress_apply method
tqdm.pandas()

if __name__ == "__main__":
    print("Creating video downloads folder...")
    # Create the video downloads folder for project if it does not exist
    video_download_folder_path = os.path.join("data", PROJECT, "video-downloads")
    os.makedirs(video_download_folder_path, exist_ok=True)

    # Load video metadata
    print("Loading video metadata...")
    video_metadata_file_path = f"data/{PROJECT}/profilesearch_video_metadata.csv"
    if not os.path.exists(video_metadata_file_path):
        raise FileNotFoundError(
            "Run profile_search.py to generate video metadata first."
        )
    else:
        video_metadata = pd.read_csv(video_metadata_file_path)

    if "video_transcript" not in video_metadata.columns:
        video_metadata.dropna(subset=["id"], inplace=True)
        video_metadata["id"] = video_metadata["id"].astype(int)
        video_metadata["id"] = video_metadata["id"].astype(str)
        video_metadata["video_filename"] = video_metadata["id"].apply(
            lambda x: x + ".mp4"
        )
        video_metadata["video_transcript"] = None

    # Filter out videos that have not been transcribed
    print("Filtering videos that have not been transcribed...")
    video_metadata_without_transcript = (
        video_metadata[video_metadata["video_transcript"].isnull()]
        .copy()
        .reset_index(drop=True)
    )
    video_metadata_without_transcript.dropna(subset=["id"], inplace=True)
    video_metadata_without_transcript["id"] = video_metadata_without_transcript[
        "id"
    ].astype(int)
    video_metadata_without_transcript["id"] = video_metadata_without_transcript[
        "id"
    ].astype(str)
    video_metadata_without_transcript["video_filename"] = (
        video_metadata_without_transcript["id"].apply(lambda x: x + ".mp4")
    )

    # Download videos that have not been transcribed and perform transcription
    print("Downloading videos that have not been transcribed...")
    video_metadata_without_transcript.progress_apply(
        download_video, args=(PROJECT,), axis=1
    )
    print("Transcribing videos...")
    video_metadata_without_transcript["video_transcript"] = (
        video_metadata_without_transcript.progress_apply(
            transcribe_videos, args=(PROJECT,), axis=1
        )
    )

    # Merge newly transcribed videos with existing video metadata
    print(
        "Merge newly transcribed videos with existing video metadata and saving it..."
    )
    video_metadata_with_transcript = video_metadata[
        ~video_metadata["video_transcript"].isnull()
    ].reset_index(drop=True)
    video_metadata = pd.concat(
        [video_metadata_with_transcript, video_metadata_without_transcript],
        ignore_index=True,
    )
    video_metadata.to_csv(
        f"data/{PROJECT}/profilesearch_video_metadata_test.csv", index=False
    )

    # Clean up downloaded videos to save disk space
    print("Disk clean up of downloaded videos...")
    for file in os.listdir(video_download_folder_path):
        file_path = os.path.join(video_download_folder_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

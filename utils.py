import pandas as pd
import os
import ast
import yt_dlp
from pydub import AudioSegment
from apify_client import ApifyClient
from openai import OpenAI

openai_client = OpenAI()


def load_text_file(file_path) -> list:
    """
    Load search terms for market signals or profile list from text file.

    Args:
        file_path (str): The path to the text file containing search terms/profiles, one per line.

    Returns:
        list: A list of search terms/profiles as strings.
    """
    with open(file_path, "r") as file:
        return [line.strip() for line in file]


def update_video_metadata(
    client: ApifyClient,
    run: dict,
    project_name: str,
    profile_search: bool,
    filtering_list: list,
) -> None:
    """
    Updates the video metadata by fetching new data, appending it to the existing data,
    and removing duplicates.

    Args:
        client (ApifyClient): The Apify client used to fetch video metadata.
        run (dict): The run object containing the default dataset ID.
        project_name (str): The name of the project, used to define the file path.
        profile_search (bool): A boolean indicating whether the search was for profiles or not.
        filtering_list (list): A list of search terms or profiles used to filter the search results.
    """
    # Fetch extracted video metadata
    video_metadata = pd.DataFrame(
        list(client.dataset(run["defaultDatasetId"]).iterate_items())
    )

    # Filter out videos based on search terms or profiles to remove irrelevant entries
    if profile_search:
        video_metadata.rename(columns={"input": "profile"}, inplace=True)
        video_metadata = video_metadata[
            video_metadata["profile"].isin(filtering_list)
        ].reset_index(drop=True)
    else:  # keyword search
        video_metadata = video_metadata[
            video_metadata["searchQuery"].isin(filtering_list)
        ].reset_index(drop=True)

    # Append extraction time to extracted video metadata
    video_metadata["extractionTime"] = pd.Timestamp.utcnow()

    # Define the file path
    if profile_search:
        video_metadata_path = f"data/{project_name}/profilesearch_video_metadata.csv"
    else:
        video_metadata_path = f"data/{project_name}/keywordsearch_video_metadata.csv"

    if os.path.exists(video_metadata_path):
        # Load existing video metadata file
        old_video_metadata = pd.read_csv(video_metadata_path)
        old_video_metadata["id"] = old_video_metadata["id"].astype("str")

        # Append new data
        video_metadata = pd.concat([old_video_metadata, video_metadata])

    # Remove duplicated video entries based on video ID, keeping the latest entry
    video_metadata.drop_duplicates(
        subset="id",
        keep="last",
        inplace=True,
    )

    # Save updated video metadata
    video_metadata.to_csv(video_metadata_path, index=False)

    return None


def update_profile_metadata(project_name: str, profile_search: bool) -> None:
    """
    Updates the profile metadata for a given project by processing the video metadata.

    Args:
        project_name (str): The name of the project for which the profile metadata is to be updated.
        profile_search (bool): A boolean indicating whether the search was for profiles or not.
    """
    # Load video metadata file
    video_metadata_path = f"data/{project_name}/video_metadata.csv"
    video_metadata = pd.read_csv(video_metadata_path)

    # Extract the authorMeta field
    profile_metadata = video_metadata[["authorMeta", "extractionTime"]]

    # Convert the authorMeta dictionary to separate columns
    profile_metadata.loc[:, "authorMeta"] = profile_metadata["authorMeta"].apply(
        lambda x: ast.literal_eval(x)
    )
    profile_metadata = pd.json_normalize(profile_metadata["authorMeta"]).join(
        profile_metadata["extractionTime"]
    )
    profile_metadata.rename(columns={"name": "profile"}, inplace=True)
    profile_metadata["id"] = profile_metadata["id"].astype("str")

    # Remove duplicates based on profile ID, keeping the latest entry
    profile_metadata.drop_duplicates(
        subset="id",
        keep="last",
        inplace=True,
    )

    # Drop invalid profiles
    profile_metadata = profile_metadata[profile_metadata["id"] != "nan"].reset_index(
        drop=True
    )

    # Save profile metadata locally, overwrite existing profile metadata if it exist
    if profile_search:
        profile_metadata_path = (
            f"data/{project_name}/profilesearch_profile_metadata.csv"
        )
    else:
        profile_metadata_path = (
            f"data/{project_name}/keywordsearch_profile_metadata.csv"
        )
    profile_metadata.to_csv(profile_metadata_path, index=False)

    return None


def identify_top_influencers(project_name: str, top_n_profiles: int) -> None:
    """
    Identifies the top N influencers based on the number of followers from a profile metadata file
    and saves their profiles to a text file.

    Args:
        project_name (str): The name of the project, used to locate the profile metadata file.
        top_n_profiles (int): The number of top profiles to identify based on the number of followers.

    Returns:
        None
    """
    # Load profile metadata file based on keyword search
    profile_metadata_path = f"data/{project_name}/keywordsearch_profile_metadata.csv"
    profile_metadata = pd.read_csv(profile_metadata_path)

    # Sort profiles based on number of followers
    profile_metadata_sorted = profile_metadata.sort_values(
        by="fans", ascending=False
    ).reset_index(drop=True)

    # Identify top n profiles based on number of followers
    profile_metadata_top_n_profiles = profile_metadata_sorted.head(top_n_profiles)

    # Save top n profiles to a text file
    profiles = profile_metadata_top_n_profiles["profile"].tolist()
    profiles_path = f"{project_name}-profiles.txt"

    with open(profiles_path, "w") as file:
        for profile in profiles:
            file.write(f"{profile}\n")

    return None


def download_video(row: pd.Series, PROJECT: str) -> None:
    """
    Downloads a TikTok video using the provided information in the row.

    Args:
        row (pd.Series): A pandas Series containing the video information, including the 'webVideoUrl' and 'video_filename'.
        PROJECT (str): The project name used to construct the output file path.

    Returns:
        None
    """
    # The TikTok video link
    video_url = row["webVideoUrl"]

    # Output file name
    output_file = f"data/{PROJECT}/video-downloads/{row['video_filename']}"

    # Options for yt-dlp
    ydl_opts = {
        "outtmpl": output_file,  # Save the video with this file name
        "format": "best",  # Download the best quality available
    }

    # Download the video
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
    except Exception as e:
        print(f"An error occurred downloading {video_url}:", str(e))


def optimize_audio_file(input_file_path: str, output_file_path: str) -> None:
    """
    Optimize an audio file by downsampling it to 16 kHz and converting it to mono.

    Args:
        input_file_path (str): The path to the input audio file.
        output_file_path (str): The path where the optimized audio file will be saved.

    Returns:
        None
    """
    # Load the audio file
    audio = AudioSegment.from_file(input_file_path)

    # Downsample the audio to 16 kHz and convert to mono
    audio = audio.set_frame_rate(16000).set_channels(1)

    # Export the optimized audio file
    audio.export(output_file_path, format="wav")


def transcribe_videos(row: pd.Series, PROJECT: str) -> str:
    """
    Transcribes the audio from a video file using the OpenAI Whisper model.
    Args:
        row (pd.Series): A pandas Series containing information about the video file.
                         It must include a 'video_filename' key with the name of the video file.
        PROJECT (str): The name of the project, used to construct the file paths.
    Returns:
        str: The transcription of the audio if successful, otherwise None.
    Raises:
        FileNotFoundError: If the input video file is not found.
        Exception: For other errors encountered during transcription, including file size issues.
    """
    input_file_path = f"downloads/{PROJECT}/{row['video_filename']}"
    optimized_file_path = f"downloads/{PROJECT}/optimized_{row['video_filename']}"

    try:
        with open(input_file_path, "rb") as audio_file:
            transcription = openai_client.audio.transcriptions.create(
                model="whisper-1", file=audio_file, response_format="text"
            )
        return transcription

    except FileNotFoundError:
        return None

    except Exception as e:
        if e.status_code == 413:
            print(
                f"Error: File {row['video_filename']} is too large to process. Optimizing the audio file..."
            )
            # Optimize the audio file
            optimize_audio_file(input_file_path, optimized_file_path)
            try:
                with open(optimized_file_path, "rb") as audio_file:
                    transcription = openai_client.audio.transcriptions.create(
                        model="whisper-1", file=audio_file, response_format="text"
                    )
                return transcription
            except Exception as e:
                print(
                    f"Error: File {optimized_file_path} is still too large after optimisation: {e}"
                )
                return None
        else:
            print(f"Error encountered when transcribing {row['video_filename']}: {e}")
            return None

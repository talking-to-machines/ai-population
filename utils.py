import pandas as pd
import os
import ast
from apify_client import ApifyClient


def load_search_terms(file_path):
    """
    Load search terms for market signals from text file.

    Args:
        file_path (str): The path to the text file containing search terms, one per line.

    Returns:
        list: A list of search terms as strings.
    """
    with open(file_path, "r") as file:
        return [line.strip() for line in file]


def update_video_metadata(client: ApifyClient, run: dict, project_name: str):
    """
    Updates the video metadata by fetching new data, appending it to the existing data,
    and removing duplicates.

    Args:
        client (ApifyClient): The Apify client used to fetch video metadata.
        run (dict): The run object containing the default dataset ID.
        project_name (str): The name of the project, used to define the file path.
    """
    # Fetch extracted video metadata
    video_metadata = pd.DataFrame(
        list(client.dataset(run["defaultDatasetId"]).iterate_items())
    )

    # Append extraction time to extracted video metadata
    video_metadata["extractionTime"] = pd.Timestamp.utcnow()

    # Define the file path
    video_metadata_path = f"data/{project_name}/video_metadata.csv"

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


def update_profile_metadata(project_name: str):
    """
    Updates the profile metadata for a given project by processing the video metadata.

    Args:
        project_name (str): The name of the project for which the profile metadata is to be updated.
    """
    # Load video metadata file
    video_metadata_path = f"data/{project_name}/video_metadata.csv"
    video_metadata = pd.read_csv(video_metadata_path)

    # Extract the authorMeta field
    profile_metadata = video_metadata[["authorMeta", "extractionTime"]]

    # Convert the authorMeta dictionary to separate columns
    profile_metadata.loc[:, "authorMeta"] = profile_metadata["authorMeta"].apply(lambda x: ast.literal_eval(x))
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
    profile_metadata_path = f"data/{project_name}/profile_metadata.csv"
    profile_metadata.to_csv(profile_metadata_path, index=False)

    return None

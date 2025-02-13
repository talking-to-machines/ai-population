import pandas as pd
import os
import ast
from apify_client import ApifyClient


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
    client: ApifyClient, run: dict, project_name: str, profile_search: bool
) -> None:
    """
    Updates the video metadata by fetching new data, appending it to the existing data,
    and removing duplicates.

    Args:
        client (ApifyClient): The Apify client used to fetch video metadata.
        run (dict): The run object containing the default dataset ID.
        project_name (str): The name of the project, used to define the file path.
        profile_search (bool): A boolean indicating whether the search was for profiles or not.
    """
    # Fetch extracted video metadata
    video_metadata = pd.DataFrame(
        list(client.dataset(run["defaultDatasetId"]).iterate_items())
    )
    if profile_search:
        video_metadata.rename(columns={"input": "profile"}, inplace=True)

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

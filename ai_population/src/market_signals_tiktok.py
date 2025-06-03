import os
import pandas as pd
import re
import string
import requests
import time
from tqdm import tqdm

tqdm.pandas()
from datetime import datetime
from ai_population.config.market_signals_config import (
    PIPELINE_EXECUTION_DATE,
    MIN_FOLLOWER_COUNT,
    MIN_VIDEO_COUNT,
    NUM_POST_PER_KEYWORD,
    NUM_RESULTS_PER_PROFILE,
    PROFILE_SEARCH_START_DATE,
    PROFILE_SEARCH_END_DATE,
    PROJECT_NAME_TIKTOK,
    SEARCH_TERMS_TIKTOK,
    FINFLUENCER_POOL_FILE_TIKTOK,
    KEYWORD_SEARCH_FILE_TIKTOK,
    PROFILE_METADATA_SEARCH_FILE_TIKTOK,
    ONBOARDING_RESULTS_FILE_TIKTOK,
    FINFLUENCER_PROFILE_METADATA_SEARCH_FILE_TIKTOK,
    FINFLUENCER_PROFILE_SEARCH_FILE_TIKTOK,
    FINFLUENCER_EXPERT_REFLECTION_FILE_TIKTOK,
    FINFLUENCER_STOCK_MENTIONS_FILE_TIKTOK,
    FINFLUENCER_POST_INTERVIEW_FILE_TIKTOK,
    FINFLUENCER_STOCK_RECOMMENDATION_FILE_TIKTOK,
    RUSSELL_4000_STOCK_TICKER_FILE,
)
from ai_population.config.base_config import (
    WAIT_TIME_BETWEEN_RETRIEVAL_REQUESTS,
    BRIGHTDATA_API,
    GPT_MODEL,
)
from ai_population.src.utils import (
    extract_llm_responses,
    extract_stock_recommendations,
    perform_profile_interview,
    perform_video_transcription,
)
from ai_population.prompts.prompt_template import (
    tiktok_finfluencer_onboarding_system_prompt,
    tiktok_finfluencer_onboarding_user_prompt,
    tiktok_portfoliomanager_reflection_system_prompt,
    tiktok_portfoliomanager_reflection_user_prompt,
    tiktok_investmentadvisor_reflection_system_prompt,
    tiktok_investmentadvisor_reflection_user_prompt,
    tiktok_financialanalyst_reflection_system_prompt,
    tiktok_financialanalyst_reflection_user_prompt,
    tiktok_economist_reflection_system_prompt,
    tiktok_economist_reflection_user_prompt,
    tiktok_finfluencer_interview_system_prompt,
    tiktok_finfluencer_interview_user_prompt,
)

base_dir = os.path.dirname(os.path.abspath(__file__))


def perform_tiktok_onboarding_interview(
    project_name: str, profile_metadata_file: str, video_file: str, output_file: str
) -> None:
    # Perform financial influencer identification interview
    perform_profile_interview(
        project_name=project_name,
        gpt_model=GPT_MODEL,
        profile_metadata_file=profile_metadata_file,
        video_file=video_file,
        output_file=output_file,
        system_prompt_template=tiktok_finfluencer_onboarding_system_prompt,
        user_prompt_template=tiktok_finfluencer_onboarding_user_prompt,
        llm_response_field="onboarding_llm_response",
        interview_type="tiktok_finfluencer_onboarding",
    )

    # Preprocess onboarding results
    onboarding_results = pd.read_csv(
        os.path.join(base_dir, "../data", project_name, output_file)
    )
    extracted_responses = onboarding_results["onboarding_llm_response"].apply(
        extract_llm_responses
    )
    onboarding_results = pd.concat([onboarding_results, extracted_responses], axis=1)

    # Save identified financial influencers
    onboarding_results.to_csv(
        os.path.join(base_dir, "../data", project_name, output_file),
        index=False,
    )


def generate_expert_reflections(
    project_name: str,
    role: str,
    profile_metadata_file: str,
    video_file: str,
    output_file: str,
) -> None:
    if role == "portfolio_manager":
        system_prompt_template = tiktok_portfoliomanager_reflection_system_prompt
        user_prompt_template = tiktok_portfoliomanager_reflection_user_prompt
        llm_response_field = "expert_reflection_portfoliomanager"
        interview_type = "portfoliomanager_reflection"

    elif role == "investment_advisor":
        system_prompt_template = tiktok_investmentadvisor_reflection_system_prompt
        user_prompt_template = tiktok_investmentadvisor_reflection_user_prompt
        llm_response_field = "expert_reflection_investmentadvisor"
        interview_type = "investmentadvisor_reflection"

    elif role == "financial_analyst":
        system_prompt_template = tiktok_financialanalyst_reflection_system_prompt
        user_prompt_template = tiktok_financialanalyst_reflection_user_prompt
        llm_response_field = "expert_reflection_financialanalyst"
        interview_type = "financialanalyst_reflection"

    elif role == "economist":
        system_prompt_template = tiktok_economist_reflection_system_prompt
        user_prompt_template = tiktok_economist_reflection_user_prompt
        llm_response_field = "expert_reflection_economist"
        interview_type = "economist_reflection"

    else:
        raise ValueError(f"Role {role} is not supported.")

    perform_profile_interview(
        project_name=project_name,
        gpt_model=GPT_MODEL,
        profile_metadata_file=profile_metadata_file,
        video_file=video_file,
        output_file=output_file,
        system_prompt_template=system_prompt_template,
        user_prompt_template=user_prompt_template,
        llm_response_field=llm_response_field,
        interview_type=interview_type,
    )


def extract_stock_mentions_from_transcripts(row: pd.Series, russell_4000_stock) -> str:
    """
    Extracts stock mentions from video transcripts and formats them into a structured string.

    This function processes a row containing combined video transcripts, identifies mentions
    of stocks listed in the Russell 4000 dataset, and returns a formatted string summarizing
    the stock mentions along with their tickers and the creation dates of the videos.

    Args:
        row (pd.Series): A pandas Series containing the column "transcripts_combined",
                         which holds the combined video transcripts.
        russell_4000_stock (pd.DataFrame): A DataFrame containing stock information with
                                           columns "COMNAM" (full stock name),
                                           "SHORTEN_COMNAM" (shortened stock name),
                                           and "TICKER" (stock ticker).

    Returns:
        str: A formatted string summarizing the stock mentions, including the stock name,
             ticker, and video creation date. Each stock mention is separated by double newlines.
    """
    # Split the transcripts by double newline
    transcript_chunks = row["transcripts_combined"].strip().split("\n\n")

    # Prepare a list for storing the matched results
    found_mentions = []

    for chunk in transcript_chunks:
        # Initialize variables for creation date and transcript text
        creation_date = "Unknown"
        transcript_text = ""

        # Extract creation date using a regular expression
        creation_date_match = re.search(r"Creation Date:\s*(.+)", chunk)
        if creation_date_match:
            creation_date = creation_date_match.group(1).strip()

        # Extract video transcript using a regular expression
        transcript_match = re.search(r"Video Transcript:\s*(.+)", chunk)
        if transcript_match:
            transcript_text = transcript_match.group(1).strip()

        # Skip processing if no transcript text is found
        if not transcript_text:
            continue

        # Check each stock in the Russell 4000
        for _, row in russell_4000_stock.iterrows():
            full_stock_name = row["COMNAM"].strip()
            shorted_stock_name = row["SHORTEN_COMNAM"].strip()
            stock_ticker = row["TICKER"].strip()

            # Check if stock name is found in transcript chunk
            transcript_text = transcript_text.translate(
                str.maketrans(string.punctuation, " " * len(string.punctuation))
            )  # Remove all punctuation from transcript_text
            name_match = (
                re.search(
                    rf"\b{re.escape(full_stock_name.lower())}\b",
                    transcript_text.lower(),
                )
                is not None
                or re.search(
                    rf"\b{re.escape(shorted_stock_name.lower())}\b",
                    transcript_text.lower(),
                )
                is not None
                or re.search(
                    rf"\b{re.escape(stock_ticker.lower())}\b", transcript_text.lower()
                )
                is not None
            )

            if name_match:
                found_mentions.append(
                    {
                        "stock_name": full_stock_name,
                        "stock_ticker": stock_ticker,
                        "video_creation_date": creation_date,
                    }
                )

    # Build a DataFrame from the matches
    stock_mentions_df = pd.DataFrame(
        found_mentions, columns=["stock_name", "stock_ticker", "video_creation_date"]
    )

    # Remove duplicates if you only want unique (stock, date) pairs
    stock_mentions_df = stock_mentions_df.drop_duplicates().reset_index(drop=True)

    # Create a formatted text string containing all the stocks mentioned and the questions for each stock
    stock_mentions_formatted_str = ""
    stock_question_template = """**stock name: {stock_name}**
**stock ticker: {stock_ticker}**
**mention date: {video_creation_date}**"""

    for i in range(len(stock_mentions_df)):
        if i != 0:
            stock_mentions_formatted_str += "\n\n"
        stock_mentions_formatted_str += stock_question_template.format(
            stock_name=stock_mentions_df.loc[i, "stock_name"],
            stock_ticker=stock_mentions_df.loc[i, "stock_ticker"],
            video_creation_date=stock_mentions_df.loc[i, "video_creation_date"],
        )

    return stock_mentions_formatted_str


def extract_stock_mentions(
    project_name: str, input_file: str, output_file: str
) -> None:
    """
    Extract stock mentions from fininfluencer profile data and save the results to a file.

    This function processes a CSV file containing fininfluencer profile data, identifies
    stock mentions in past video transcripts using a predefined list of stock tickers,
    and saves the updated data with stock mentions to a new CSV file.

    Args:
        project_name (str): The name of the project directory containing the input and output files.
        input_file (str): The name of the input CSV file containing fininfluencer profile data.
        output_file (str): The name of the output CSV file to save the processed data with stock mentions.

    Returns:
        None: This function does not return any value. It performs file I/O operations.
    """
    # Load fininfluencer profile data
    fininfluencer_profile_data = pd.read_csv(
        os.path.join(base_dir, "../data", project_name, input_file)
    )

    # Extract stocks mention in past videos
    russell_4000_stock = pd.read_csv(
        os.path.join(base_dir, "../config", RUSSELL_4000_STOCK_TICKER_FILE)
    )
    fininfluencer_profile_data["stock_mentions"] = (
        fininfluencer_profile_data.progress_apply(
            extract_stock_mentions_from_transcripts, args=(russell_4000_stock,), axis=1
        )
    )

    # Save formatted post reflection results
    fininfluencer_profile_data.to_csv(
        os.path.join(base_dir, "../data", project_name, output_file), index=False
    )


def perform_tiktok_finfluencer_interview(
    project_name: str, profile_metadata_file: str, video_file: str, output_file: str
) -> None:

    perform_profile_interview(
        project_name=project_name,
        gpt_model=GPT_MODEL,
        profile_metadata_file=profile_metadata_file,
        video_file=video_file,
        output_file=output_file,
        system_prompt_template=tiktok_finfluencer_interview_system_prompt,
        user_prompt_template=tiktok_finfluencer_interview_user_prompt,
        llm_response_field="tiktok_finfluencer_interview",
        interview_type="tiktok_finfluencer_interview",
    )

    # Preprocess post interview results
    post_interview_results = pd.read_csv(
        os.path.join(base_dir, "../data", project_name, output_file)
    )
    extracted_responses = post_interview_results["tiktok_finfluencer_interview"].apply(
        extract_llm_responses,
        args=(
            [
                "stock name",
                "A list of Russell 4000 stocks was extracted from your past video transcripts",
            ],
        ),
    )
    post_interview_results = pd.concat(
        [post_interview_results, extracted_responses], axis=1
    )

    # Extract stock recommendations
    combined_stock_recommendations = pd.DataFrame()
    for i in range(len(post_interview_results)):
        profile_stock_recommendations = extract_stock_recommendations(
            post_interview_results.iloc[i],
            llm_response_field="tiktok_finfluencer_interview",
        )

        if profile_stock_recommendations is None:  # No stock recommendations
            continue

        profile_stock_recommendations["account_id"] = post_interview_results.loc[
            i, "account_id"
        ]
        profile_stock_recommendations["profile_url"] = post_interview_results.loc[
            i, "url"
        ]
        profile_stock_recommendations["followers"] = post_interview_results.loc[
            i, "followers"
        ]
        profile_stock_recommendations["influence"] = post_interview_results.loc[
            i,
            "Indicate on a scale of 0 to 100, how influential this influencer is – 0 means not at all influential and 100 means very influential with millions of followers and mainstream recognition? - value",
        ]
        profile_stock_recommendations["credibility"] = post_interview_results.loc[
            i,
            "Indicate on a scale of 0 to 100, how credible or authoritative this influencer is – 0 means not at all credible or authoritative and 100 means very credible and authoritative? - value",
        ]

        combined_stock_recommendations = pd.concat(
            [combined_stock_recommendations, profile_stock_recommendations],
            ignore_index=True,
        )

    # Remove duplicated entries and stocks that were not mentioned in the video transcripts
    valid_stock_recommendations = (
        combined_stock_recommendations.drop_duplicates().reset_index(drop=True)
    )

    # Sort by profile and mention date (descending order within each profile)
    valid_stock_recommendations["mention date"] = pd.to_datetime(
        valid_stock_recommendations["mention date"]
    )
    valid_stock_recommendations = valid_stock_recommendations.sort_values(
        by=["account_id", "mention date"], ascending=[True, False]
    ).reset_index(drop=True)

    # Remove stocks that are not mentioned by the influencer
    valid_stock_recommendations = valid_stock_recommendations[
        combined_stock_recommendations["mentioned by influencer"] == "Yes"
    ].reset_index(drop=True)

    # Save formatted interview results and stock recommendations
    post_interview_results.to_csv(
        os.path.join(base_dir, "../data", project_name, output_file), index=False
    )
    valid_stock_recommendations.to_csv(
        os.path.join(
            base_dir,
            "../data",
            project_name,
            FINFLUENCER_STOCK_RECOMMENDATION_FILE_TIKTOK,
        ),
        index=False,
    )


def perform_tiktok_keyword_search(
    project_name: str,
    search_terms: list,
    output_file_path: str,
) -> pd.DataFrame:
    """
    Perform a TikTok keyword search using the Bright Data API and save the results to a CSV file.

    Args:
        project_name (str): The name of the project. A subfolder with this name will be created
            within the data folder to store the output file.
        search_terms (list): The list containing the search terms,
            one term per line.
        output_file_path (str): The file path where the resulting CSV file will be saved.

    Returns:
        pd.DataFrame: Returns the keyword search results as a pandas Dataframe.

    Raises:
        requests.exceptions.RequestException: If there is an issue with the API request.
        KeyError: If the response from the API does not contain the expected keys.
        ValueError: If the response data is not in the expected format.
    """
    # Create the project subfolder within the data folder if it does not exist
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(base_dir, "../data"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "../data", project_name), exist_ok=True)

    # Initialise keyword search job
    data = [
        {"search_keyword": keyword, "num_of_posts": NUM_POST_PER_KEYWORD, "country": ""}
        for keyword in search_terms
    ]
    response = requests.post(
        "https://api.brightdata.com/datasets/v3/trigger",
        headers={
            "Authorization": f"Bearer {BRIGHTDATA_API}",
            "Content-Type": "application/json",
        },
        params={
            "dataset_id": "gd_lu702nij2f790tmv9h",
            "format": "csv",
            "uncompressed_webhook": "true",
            "force_deliver": "true",
            "include_errors": "true",
            "type": "discover_new",
            "discover_by": "keyword",
        },
        json=data,
    )
    snapshot_id = response.json().get("snapshot_id")

    # Retrieve keyword search results
    response_json = {"status": "running"}
    while isinstance(response_json, dict) and response_json.get("status") == "running":
        response = requests.get(
            f"https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}",
            headers={
                "Authorization": f"Bearer {BRIGHTDATA_API}",
            },
            params={
                "format": "json",
            },
        )
        response_json = response.json()
        time.sleep(WAIT_TIME_BETWEEN_RETRIEVAL_REQUESTS)

    keyword_search_results = pd.DataFrame(response_json)
    if "warning_code" in keyword_search_results.columns:
        keyword_search_results = keyword_search_results[
            keyword_search_results["warning_code"] != "dead_page"
        ].reset_index(drop=True)
    if "error_code" in keyword_search_results.columns:
        keyword_search_results = keyword_search_results[
            keyword_search_results["error_code"] != "crawl_failed"
        ].reset_index(drop=True)
    keyword_search_results.to_csv(
        os.path.join(base_dir, "../data", project_name, output_file_path), index=False
    )

    return keyword_search_results


def perform_tiktok_profile_search(
    project_name: str,
    input_file_path: str,
    output_file_path: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Perform a TikTok profile search and retrieve posts data for specified profiles.

    This function triggers a profile search job using the Bright Data API, retrieves
    the results, and saves them to a CSV file. It also ensures the necessary project
    subfolder structure exists within the data directory.

    Args:
        project_name (str): Name of the project, used to create a subfolder in the data directory.
        input_file_path (str): Path to the CSV file containing TikTok account IDs under the column "account_id".
        output_file_path (str): Path to save the retrieved profile search results as a CSV file.
        start_date (str): The start date for the profile search.
        end_date (str): The end date for the profile search.

    Returns:
        pd.DataFrame: A DataFrame containing the profile search results.
    """
    # Create the project subfolder within the data folder if it does not exist
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(base_dir, "../data"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "../data", project_name), exist_ok=True)

    # Define search parameters
    profile_list = pd.read_csv(
        os.path.join(base_dir, "../data", project_name, input_file_path)
    )["account_id"].tolist()

    # Initialise profile search job
    data = [
        {
            "url": f"https://www.tiktok.com/@{profile}",
            "num_of_posts": NUM_RESULTS_PER_PROFILE,
            "posts_to_not_include": "",
            "start_date": start_date,
            "end_date": end_date,
            "what_to_collect": "Posts",
            "post_type": "Video Posts",
            "country": "",
        }
        for profile in profile_list
    ]
    response = requests.post(
        "https://api.brightdata.com/datasets/v3/trigger",
        headers={
            "Authorization": f"Bearer {BRIGHTDATA_API}",
            "Content-Type": "application/json",
        },
        params={
            "dataset_id": "gd_lu702nij2f790tmv9h",
            "include_errors": "true",
            "type": "discover_new",
            "discover_by": "profile_url",
        },
        json=data,
    )
    snapshot_id = response.json().get("snapshot_id")

    # Retrieve profile search results
    response_json = {"status": "running"}
    while isinstance(response_json, dict) and response_json.get("status") == "running":
        response = requests.get(
            f"https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}",
            headers={
                "Authorization": f"Bearer {BRIGHTDATA_API}",
            },
            params={
                "format": "json",
            },
        )
        response_json = response.json()
        time.sleep(WAIT_TIME_BETWEEN_RETRIEVAL_REQUESTS)

    profile_search_results = pd.DataFrame(response_json)
    if "warning_code" in profile_search_results.columns:
        profile_search_results = profile_search_results[
            profile_search_results["warning_code"] != "dead_page"
        ].reset_index(drop=True)
    if "error_code" in profile_search_results.columns:
        profile_search_results = profile_search_results[
            profile_search_results["error_code"] != "crawl_failed"
        ].reset_index(drop=True)

    profile_search_results.to_csv(
        os.path.join(base_dir, "../data", project_name, output_file_path), index=False
    )

    return profile_search_results


def perform_tiktok_profile_metadata_search(
    project_name: str, input_file_path: str, output_file_path: str = ""
) -> pd.DataFrame:
    """
    Perform a TikTok profile metadata search using Bright Data API and save the results to a CSV file.

    Args:
        project_name (str): Name of the project. Used to create a subfolder within the data directory.
        input_file_path (str): Path to the input DataFrame containing TikTok account IDs. Must include a column named 'account_id'.
        output_file_path (str, optional): Path to save the output CSV file. Defaults to an empty string.

    Returns:
        pd.DataFrame: DataFrame containing the TikTok profile metadata search results.

    Raises:
        AssertionError: If the input DataFrame does not contain the 'account_id' column.
    """
    # Create the project subfolder within the data folder if it does not exist
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(base_dir, "../data"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "../data", project_name), exist_ok=True)

    # Define list of profiles for search
    profile_data = pd.read_csv(
        os.path.join(base_dir, "../data", project_name, input_file_path)
    )
    assert (
        "account_id" in profile_data.columns
    ), "Input file must contain 'account_id' column."
    profile_list = list(set(profile_data["account_id"].tolist()))

    # Initialise profile metadata search job
    data = [
        {"url": f"https://www.tiktok.com/@{profile}", "country": ""}
        for profile in profile_list
    ]
    response = requests.post(
        "https://api.brightdata.com/datasets/v3/trigger",
        headers={
            "Authorization": f"Bearer {BRIGHTDATA_API}",
            "Content-Type": "application/json",
        },
        params={
            "dataset_id": "gd_l1villgoiiidt09ci",
            "include_errors": "true",
        },
        json=data,
    )
    snapshot_id = response.json().get("snapshot_id")

    # Retrieve keyword search results
    response_json = {"status": "running"}
    while isinstance(response_json, dict) and response_json.get("status") == "running":
        response = requests.get(
            f"https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}",
            headers={
                "Authorization": f"Bearer {BRIGHTDATA_API}",
            },
            params={
                "format": "json",
            },
        )
        response_json = response.json()
        time.sleep(WAIT_TIME_BETWEEN_RETRIEVAL_REQUESTS)

    profile_metadata_search_results = pd.DataFrame(response_json)
    if "warning_code" in profile_metadata_search_results.columns:
        profile_metadata_search_results = profile_metadata_search_results[
            profile_metadata_search_results["warning_code"] != "dead_page"
        ].reset_index(drop=True)
    if "error_code" in profile_metadata_search_results.columns:
        profile_metadata_search_results = profile_metadata_search_results[
            profile_metadata_search_results["error_code"] != "crawl_failed"
        ].reset_index(drop=True)

    profile_metadata_search_results.to_csv(
        os.path.join(base_dir, "../data", project_name, output_file_path), index=False
    )

    return profile_metadata_search_results


def filter_tiktok_profiles(
    project_name: str,
    profile_metadata_file: str,
    video_file: str,
    verified_profile_pool: str,
) -> tuple:
    """
    Filters TikTok profiles and associated video data based on specified criteria.

    Args:
        project_name (str): Name of the project.
        profile_metadata_file (str): Path to the CSV file containing profile metadata.
        video_file (str): Path to the CSV file containing video data.
        verified_profile_pool (str): Path to the CSV file containing verified profiles.

    Returns:
        tuple: A tuple containing two DataFrames:
            - filtered_profiles: DataFrame of profiles that meet the filtering criteria.
            - filtered_videos: DataFrame of videos associated with the filtered profiles.
    """
    profile_metadata = pd.read_csv(
        os.path.join(base_dir, "../data", project_name, profile_metadata_file)
    )
    verified_profile_pool = pd.read_csv(
        os.path.join(base_dir, "../data", project_name, verified_profile_pool)
    )
    video_data = pd.read_csv(
        os.path.join(base_dir, "../data", project_name, video_file)
    )

    # Filter profiles based on criteria
    verified_profiles = verified_profile_pool["account_id"].tolist()
    filtered_profiles = profile_metadata[
        (profile_metadata["followers"] >= MIN_FOLLOWER_COUNT)  # Minimum followers
        & (profile_metadata["videos_count"] >= MIN_VIDEO_COUNT)  # Minimum videos posted
        & ~(
            profile_metadata["account_id"].isin(verified_profiles)
        )  # Remove profiles that have been verified
    ].reset_index(drop=True)
    filtered_profiles.to_csv(profile_metadata_file, index=False)

    # Filter video files based on profiles that meet filtering criteria
    filtered_profile_list = filtered_profiles["account_id"].tolist()
    filtered_videos = video_data[
        video_data["account_id"].isin(filtered_profile_list)
    ].reset_index(drop=True)
    filtered_videos.to_csv(video_file, index=False)

    return filtered_profiles, filtered_videos


def update_verified_profile_pool(
    project_name: str, input_file_path: str, verified_profile_pool: str
) -> None:
    """
    Updates the verified profile pool by adding new financial influencers
    identified from the interviewed profiles.

    Args:
        project_name (str): The name of the project, used to locate the data directory.
        input_file_path (str): The relative path to the CSV file containing interviewed profiles.
        verified_profile_pool (str): The relative path to the CSV file containing the verified profile pool.

    Returns:
        None: Updates the verified profile pool file in place.
    """
    interviewed_profiles = pd.read_csv(
        os.path.join(base_dir, "../data", project_name, input_file_path)
    )
    verified_profiles = pd.read_csv(
        os.path.join(base_dir, "../data", project_name, verified_profile_pool)
    )

    # Filter out financial influencers
    finfluencer_profiles = interviewed_profiles[
        interviewed_profiles["Is this a finfluencer? - category"].str.contains(
            "Yes", na=False
        )
    ]

    # Add new financial influencers to the verified profile pool
    if not finfluencer_profiles.empty:
        verified_profiles = pd.concat(
            [
                verified_profiles,
                pd.DataFrame(
                    {
                        "account_id": finfluencer_profiles["account_id"].tolist(),
                        "inclusion_date": PIPELINE_EXECUTION_DATE,
                    }
                ),
            ],
            ignore_index=True,
        )

        # Save updated verified profile pool
        verified_profiles.to_csv(
            os.path.join(base_dir, "../data", project_name, verified_profile_pool),
            index=False,
        )
    else:
        pass


if __name__ == "__main__":
    # Step 1: Perform search using predefined list of search terms
    print("Perform keyword search using predefined list of search terms...")
    perform_tiktok_keyword_search(
        project_name=PROJECT_NAME_TIKTOK,
        search_terms=SEARCH_TERMS_TIKTOK,
        output_file_path=KEYWORD_SEARCH_FILE_TIKTOK,
    )

    # Step 2: Extract profile metadata for search results
    print("Perform profile metadata search for keyword search results...")
    perform_tiktok_profile_metadata_search(
        project_name=PROJECT_NAME_TIKTOK,
        input_file_path=KEYWORD_SEARCH_FILE_TIKTOK,
        output_file_path=PROFILE_METADATA_SEARCH_FILE_TIKTOK,
    )

    # Step 3: Filter profiles that do not meet filtering criteria
    print(
        "Filter TikTok profiles based on follower count, video count, and verified finfluencer list..."
    )
    filter_tiktok_profiles(
        profile_metadata_file=PROFILE_METADATA_SEARCH_FILE_TIKTOK,
        video_file=KEYWORD_SEARCH_FILE_TIKTOK,
        verified_profile_pool=FINFLUENCER_POOL_FILE_TIKTOK,
    )

    # Step 4: Perform video transcription of new videos
    print("Perform video transcription of new videos...")
    perform_video_transcription(
        project_name=PROJECT_NAME_TIKTOK,
        video_file=KEYWORD_SEARCH_FILE_TIKTOK,
    )

    # # Step 5: Conduct onboarding interview to identify financial influencers and add to influencer pool
    print("Perform onboarding interview to identify financial influencers...")
    perform_tiktok_onboarding_interview(
        project_name=PROJECT_NAME_TIKTOK,
        profile_metadata_file=PROFILE_METADATA_SEARCH_FILE_TIKTOK,
        video_file=KEYWORD_SEARCH_FILE_TIKTOK,
        output_file=ONBOARDING_RESULTS_FILE_TIKTOK,
    )
    update_verified_profile_pool(
        input_file_path=ONBOARDING_RESULTS_FILE_TIKTOK,
        verified_profile_pool=FINFLUENCER_POOL_FILE_TIKTOK,
    )

    # # Step 6: Perform profile search of identified financial influencers (profile metadata and posts) during search period
    perform_tiktok_profile_metadata_search(
        project_name=PROJECT_NAME_TIKTOK,
        input_file_path=FINFLUENCER_POOL_FILE_TIKTOK,
        output_file_path=FINFLUENCER_PROFILE_METADATA_SEARCH_FILE_TIKTOK,
    )
    perform_tiktok_profile_search(
        project_name=PROJECT_NAME_TIKTOK,
        input_file_path=FINFLUENCER_POOL_FILE_TIKTOK,
        output_file_path=FINFLUENCER_PROFILE_SEARCH_FILE_TIKTOK,
        start_date=PROFILE_SEARCH_START_DATE,
        end_date=PROFILE_SEARCH_END_DATE,
    )

    # Step 7: Generate expert reflections
    generate_expert_reflections(
        project_name=PROJECT_NAME_TIKTOK,
        role="portfolio_manager",
        profile_metadata_file=FINFLUENCER_PROFILE_METADATA_SEARCH_FILE_TIKTOK,
        video_file=FINFLUENCER_PROFILE_SEARCH_FILE_TIKTOK,
        output_file=FINFLUENCER_EXPERT_REFLECTION_FILE_TIKTOK,
    )
    generate_expert_reflections(
        project_name=PROJECT_NAME_TIKTOK,
        role="investment_advisor",
        profile_metadata_file=FINFLUENCER_EXPERT_REFLECTION_FILE_TIKTOK,
        video_file=FINFLUENCER_PROFILE_SEARCH_FILE_TIKTOK,
        output_file=FINFLUENCER_EXPERT_REFLECTION_FILE_TIKTOK,
    )
    generate_expert_reflections(
        project_name=PROJECT_NAME_TIKTOK,
        role="financial_analyst",
        profile_metadata_file=FINFLUENCER_EXPERT_REFLECTION_FILE_TIKTOK,
        video_file=FINFLUENCER_PROFILE_SEARCH_FILE_TIKTOK,
        output_file=FINFLUENCER_EXPERT_REFLECTION_FILE_TIKTOK,
    )
    generate_expert_reflections(
        project_name=PROJECT_NAME_TIKTOK,
        role="economist",
        profile_metadata_file=FINFLUENCER_EXPERT_REFLECTION_FILE_TIKTOK,
        video_file=FINFLUENCER_PROFILE_SEARCH_FILE_TIKTOK,
        output_file=FINFLUENCER_EXPERT_REFLECTION_FILE_TIKTOK,
    )

    # Step 8: Extract stock mentions from financial influencers' past video transcripts
    extract_stock_mentions(
        project_name=PROJECT_NAME_TIKTOK,
        input_file=FINFLUENCER_EXPERT_REFLECTION_FILE_TIKTOK,
        output_file=FINFLUENCER_STOCK_MENTIONS_FILE_TIKTOK,
    )

    # Step 9: Conduct interview on financial markets and stock recommendations
    perform_tiktok_finfluencer_interview(
        project_name=PROJECT_NAME_TIKTOK,
        profile_metadata_file=FINFLUENCER_STOCK_MENTIONS_FILE_TIKTOK,
        video_file=FINFLUENCER_PROFILE_SEARCH_FILE_TIKTOK,
        output_file=FINFLUENCER_POST_INTERVIEW_FILE_TIKTOK,
    )

import os
import pandas as pd
import requests
import time
from tqdm import tqdm

tqdm.pandas()
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
    EXPERT_REFLECTION_FILE_TIKTOK,
    FINFLUENCER_PROFILE_METADATA_SEARCH_FILE_TIKTOK,
    FINFLUENCER_PROFILE_SEARCH_FILE_TIKTOK,
    FINFLUENCER_STOCK_MENTIONS_FILE_TIKTOK,
    FINFLUENCER_POST_INTERVIEW_FILE_TIKTOK,
    FINFLUENCER_STOCK_RECOMMENDATION_FILE_TIKTOK,
    ONBOARDING_INTERVIEW_REGEX_PATTERNS,
    FINFLUENCER_INTERVIEW_REGEX_PATTERNS,
    STOCK_RECOMMENDATION_OUTPUT_COLUMNS,
    PREDICTION_THRESHOLD_TIKTOK,
)
from ai_population.config.base_config import (
    WAIT_TIME_BETWEEN_RETRIEVAL_REQUESTS,
    BRIGHTDATA_API,
    GPT_MODEL,
)
from ai_population.src.utils import (
    extract_llm_responses,
    format_stock_mentions,
    perform_profile_interview,
    perform_video_transcription,
    update_verified_profile_pool,
    coalesce_columns_by_regex,
    extract_stock_mentions,
    format_stock_recommendations,
)
from ai_population.prompts.prompt_template import (
    tiktok_finfluencer_onboarding_system_prompt,
    tiktok_finfluencer_onboarding_user_prompt,
    tiktok_portfoliomanager_reflection_system_prompt,
    portfoliomanager_reflection_user_prompt,
    tiktok_investmentadvisor_reflection_system_prompt,
    investmentadvisor_reflection_user_prompt,
    tiktok_financialanalyst_reflection_system_prompt,
    financialanalyst_reflection_user_prompt,
    tiktok_economist_reflection_system_prompt,
    economist_reflection_user_prompt,
    tiktok_finfluencer_interview_system_prompt,
    finfluencer_interview_user_prompt,
    stock_recommendation_interview_user_prompt,
)

base_dir = os.path.dirname(os.path.abspath(__file__))


def perform_tiktok_onboarding_interview(
    project_name: str,
    execution_date: str,
    profile_metadata_file: str,
    post_file: str,
    output_file: str,
) -> None:
    """
    Performs the TikTok onboarding interview process for financial influencer identification and saves the results.

    This function executes a profile interview using specified prompt templates and processes the resulting data.
    It reads the onboarding results, extracts LLM responses, appends them to the results, and saves the updated data.

    Args:
        project_name (str): The name of the project.
        execution_date (str): The date of execution in string format.
        profile_metadata_file (str): Path to the profile metadata file.
        post_file (str): Path to the post file to be processed.
        output_file (str): Name of the output CSV file to save results.

    Returns:
        None
    """
    # Perform financial influencer identification interview
    perform_profile_interview(
        project_name=project_name,
        execution_date=execution_date,
        gpt_model=GPT_MODEL,
        profile_metadata_file=profile_metadata_file,
        post_file=post_file,
        output_file=output_file,
        system_prompt_template=tiktok_finfluencer_onboarding_system_prompt,
        user_prompt_template=tiktok_finfluencer_onboarding_user_prompt,
        llm_response_field="onboarding_llm_response",
        interview_type="tiktok_finfluencer_onboarding",
    )

    # Preprocess onboarding results
    onboarding_results = pd.read_csv(
        os.path.join(base_dir, "../data", project_name, execution_date, output_file)
    )
    extracted_responses = onboarding_results["onboarding_llm_response"].apply(
        extract_llm_responses
    )
    onboarding_results = pd.concat([onboarding_results, extracted_responses], axis=1)

    # Merge identical columns from interview response
    onboarding_results = coalesce_columns_by_regex(
        onboarding_results, ONBOARDING_INTERVIEW_REGEX_PATTERNS
    )

    # Save identified financial influencers
    onboarding_results.to_csv(
        os.path.join(base_dir, "../data", project_name, execution_date, output_file),
        index=False,
    )


def generate_expert_reflections(
    project_name: str,
    execution_date: str,
    role: str,
    profile_metadata_file: str,
    post_file: str,
    output_file: str,
) -> None:
    """
    Generates expert reflections for a given role based on TikTok posts and profile metadata.

    Depending on the specified role, selects the appropriate prompt templates and response fields,
    then invokes the profile interview process to generate and save the expert reflection.

    Args:
        project_name (str): The name of the project.
        execution_date (str): The date of execution.
        role (str): The expert role for reflection generation. Supported roles are
            "portfolio_manager", "investment_advisor", "financial_analyst", and "economist".
        profile_metadata_file (str): Path to the profile metadata file.
        post_file (str): Path to the TikTok post file.
        output_file (str): Path where the generated reflection will be saved.

    Raises:
        ValueError: If the provided role is not supported.
    """
    if role == "portfolio_manager":
        system_prompt_template = tiktok_portfoliomanager_reflection_system_prompt
        user_prompt_template = portfoliomanager_reflection_user_prompt
        llm_response_field = (
            "tiktok_finfluencer_expert_reflection_portfoliomanager_response"
        )
        interview_type = "tiktok_finfluencer_expert_reflection_portfoliomanager"

    elif role == "investment_advisor":
        system_prompt_template = tiktok_investmentadvisor_reflection_system_prompt
        user_prompt_template = investmentadvisor_reflection_user_prompt
        llm_response_field = (
            "tiktok_finfluencer_expert_reflection_investmentadvisor_response"
        )
        interview_type = "tiktok_finfluencer_expert_reflection_investmentadvisor"

    elif role == "financial_analyst":
        system_prompt_template = tiktok_financialanalyst_reflection_system_prompt
        user_prompt_template = financialanalyst_reflection_user_prompt
        llm_response_field = (
            "tiktok_finfluencer_expert_reflection_financialanalyst_response"
        )
        interview_type = "tiktok_finfluencer_expert_reflection_financialanalyst"

    elif role == "economist":
        system_prompt_template = tiktok_economist_reflection_system_prompt
        user_prompt_template = economist_reflection_user_prompt
        llm_response_field = "tiktok_finfluencer_expert_reflection_economist_response"
        interview_type = "tiktok_finfluencer_expert_reflection_economist"

    else:
        raise ValueError(f"Role {role} is not supported.")

    perform_profile_interview(
        project_name=project_name,
        execution_date=execution_date,
        gpt_model=GPT_MODEL,
        profile_metadata_file=profile_metadata_file,
        post_file=post_file,
        output_file=output_file,
        system_prompt_template=system_prompt_template,
        user_prompt_template=user_prompt_template,
        llm_response_field=llm_response_field,
        interview_type=interview_type,
    )


def perform_tiktok_finfluencer_interview(
    project_name: str,
    execution_date: str,
    profile_metadata_file: str,
    post_file: str,
    output_file: str,
) -> None:
    """
    Conducts a TikTok finfluencer interview workflow, processes the results, and saves the formatted output.

    This function performs the following steps:
    1. Runs a profile interview for a TikTok finfluencer using specified prompt templates and parameters.
    2. Loads the interview results from a CSV file.
    3. Extracts and processes LLM responses from the interview results.
    4. Merges identical columns in the results based on predefined regex patterns.
    5. Saves the processed and formatted interview results back to the CSV file.

    Args:
        project_name (str): Name of the project directory.
        execution_date (str): Date of execution, used for organizing output files.
        profile_metadata_file (str): Path to the file containing profile metadata.
        post_file (str): Path to the file containing post data.
        output_file (str): Name of the output CSV file to save results.

    Returns:
        None
    """
    perform_profile_interview(
        project_name=project_name,
        execution_date=execution_date,
        gpt_model=GPT_MODEL,
        profile_metadata_file=profile_metadata_file,
        post_file=post_file,
        output_file=output_file,
        system_prompt_template=tiktok_finfluencer_interview_system_prompt,
        user_prompt_template=finfluencer_interview_user_prompt,
        llm_response_field="tiktok_finfluencer_interview",
        interview_type="tiktok_finfluencer_interview",
    )

    # Preprocess post interview results
    post_interview_results = pd.read_csv(
        os.path.join(base_dir, "../data", project_name, execution_date, output_file)
    )
    extracted_responses = post_interview_results["tiktok_finfluencer_interview"].apply(
        extract_llm_responses
    )
    post_interview_results = pd.concat(
        [post_interview_results, extracted_responses], axis=1
    )
    # Merge identical columns from interview response
    post_interview_results = coalesce_columns_by_regex(
        post_interview_results, FINFLUENCER_INTERVIEW_REGEX_PATTERNS
    )

    # Save formatted interview results
    post_interview_results.to_csv(
        os.path.join(base_dir, "../data", project_name, execution_date, output_file),
        index=False,
    )


def perform_tiktok_stock_recommendation_interview(
    project_name: str,
    execution_date: str,
    profile_metadata_file: str,
    post_file: str,
    finfluencer_pool: str,
    output_file: str,
) -> None:
    """
    Performs the TikTok stock recommendation interview process for a given project and execution date.

    This function processes influencer profile metadata and stock mention data, formats and merges relevant information,
    removes duplicate stock recommendations, and saves the formatted data. It then conducts an interview process using
    a language model to verify stock recommendations, sorts and filters the results, and saves the final verified recommendations.

    Args:
        project_name (str): Name of the project directory.
        execution_date (str): Date of execution, used for organizing data.
        profile_metadata_file (str): Filename of the influencer profile metadata CSV.
        post_file (str): Filename of the post data CSV.
        finfluencer_pool (str): Filename of the finfluencer pool CSV containing influence and credibility scores.
        output_file (str): Filename for saving the processed and verified stock recommendations.

    Returns:
        None
    """
    finfluencer_pool = pd.read_csv(
        os.path.join(base_dir, "../data", project_name, finfluencer_pool)
    )
    profile_metadata = pd.read_csv(
        os.path.join(
            base_dir, "../data", project_name, execution_date, profile_metadata_file
        )
    )

    # Prepare stock mention dataset for interview
    combined_stock_mentions = pd.DataFrame()
    for i in range(len(profile_metadata)):
        if (
            pd.isnull(profile_metadata.loc[i, "stock_mentions"])
            or not profile_metadata.loc[i, "stock_mentions"]
        ):
            continue  # No stock mentions

        profile_stock_mentions = format_stock_mentions(
            profile_metadata.loc[i, "stock_mentions"]
        )
        profile_stock_mentions["account_id"] = profile_metadata.loc[i, "account_id"]
        profile_stock_mentions = pd.merge(
            left=profile_stock_mentions,
            right=profile_metadata,
            how="left",
            on="account_id",
        )
        profile_stock_mentions["url"] = profile_metadata.loc[i, "url"]
        profile_stock_mentions["followers"] = profile_metadata.loc[i, "followers"]
        profile_stock_mentions["influence"] = finfluencer_pool[
            finfluencer_pool["account_id"] == profile_metadata.loc[i, "account_id"]
        ]["influence"].values[0]
        profile_stock_mentions["credibility"] = finfluencer_pool[
            finfluencer_pool["account_id"] == profile_metadata.loc[i, "account_id"]
        ]["credibility"].values[0]
        combined_stock_mentions = pd.concat(
            [combined_stock_mentions, profile_stock_mentions], ignore_index=True
        )

    # Remove duplicated stocks recommendations
    combined_stock_mentions = combined_stock_mentions.drop_duplicates().reset_index(
        drop=True
    )

    # Save formatted stock mentions for interview process
    combined_stock_mentions.to_csv(
        os.path.join(base_dir, "../data", project_name, execution_date, output_file),
        index=False,
    )

    # Perform interview for stock recommendations
    perform_profile_interview(
        project_name=project_name,
        execution_date=execution_date,
        gpt_model=GPT_MODEL,
        profile_metadata_file=output_file,
        post_file=post_file,
        output_file=output_file,
        system_prompt_template=tiktok_finfluencer_interview_system_prompt,
        user_prompt_template=stock_recommendation_interview_user_prompt,
        llm_response_field="tiktok_finfluencer_stock_recommendation",
        interview_type="tiktok_finfluencer_stock_recommendation",
    )

    stock_recommendations = pd.read_csv(
        os.path.join(base_dir, "../data", project_name, execution_date, output_file)
    )

    # Extract stock recommendation responses
    extracted_responses = stock_recommendations[
        "tiktok_finfluencer_stock_recommendation"
    ].apply(format_stock_recommendations)
    stock_recommendations = pd.concat(
        [stock_recommendations, extracted_responses], axis=1
    )

    # Sort by profile and mention date (descending order within each profile)
    stock_recommendations["mention_date"] = pd.to_datetime(
        stock_recommendations["mention_date"]
    )
    stock_recommendations = stock_recommendations.sort_values(
        by=["account_id", "mention_date"], ascending=[True, False]
    ).reset_index(drop=True)

    # Retain verified stock recommendations
    valid_stock_recommendations = stock_recommendations[
        stock_recommendations["mentioned_by_finfluencer"].isin(["Yes", "No"])
    ].reset_index(drop=True)

    # Save verified stock recommendations
    valid_stock_recommendations[STOCK_RECOMMENDATION_OUTPUT_COLUMNS].to_csv(
        os.path.join(base_dir, "../data", project_name, execution_date, output_file),
        index=False,
    )


def perform_tiktok_keyword_search(
    project_name: str,
    execution_date: str,
    search_terms: list,
    output_file_path: str,
) -> pd.DataFrame:
    """
    Perform a TikTok keyword search using the Bright Data API and save the results to a CSV file.

    Args:
        project_name (str): The name of the project. A subfolder with this name will be created
            within the data folder to store the output file.
        execute_date (str): The date of the pipeline execution, used to create a unique directory name.
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
    os.makedirs(
        os.path.join(base_dir, "../data", project_name, execution_date), exist_ok=True
    )

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
        time.sleep(WAIT_TIME_BETWEEN_RETRIEVAL_REQUESTS)
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
        os.path.join(
            base_dir, "../data", project_name, execution_date, output_file_path
        ),
        index=False,
    )

    return keyword_search_results


def perform_tiktok_profile_search(
    project_name: str,
    execution_date: str,
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
        execute_date (str): The date of the pipeline execution, used to create a unique directory name.
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
    os.makedirs(
        os.path.join(base_dir, "../data", project_name, execution_date), exist_ok=True
    )

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
        time.sleep(WAIT_TIME_BETWEEN_RETRIEVAL_REQUESTS)
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
        os.path.join(
            base_dir, "../data", project_name, execution_date, output_file_path
        ),
        index=False,
    )

    return profile_search_results


def perform_tiktok_profile_metadata_search(
    project_name: str,
    execution_date: str,
    input_file_path: str,
    output_file_path: str = "",
) -> pd.DataFrame:
    """
    Perform a TikTok profile metadata search using Bright Data API and save the results to a CSV file.

    Args:
        project_name (str): Name of the project. Used to create a subfolder within the data directory.
        execute_date (str): The date of the pipeline execution, used to create a unique directory name.
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
    os.makedirs(
        os.path.join(base_dir, "../data", project_name, execution_date), exist_ok=True
    )

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
        time.sleep(WAIT_TIME_BETWEEN_RETRIEVAL_REQUESTS)
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
        os.path.join(
            base_dir, "../data", project_name, execution_date, output_file_path
        ),
        index=False,
    )

    return profile_metadata_search_results


def filter_tiktok_profiles(
    project_name: str,
    execution_date: str,
    profile_metadata_file: str,
    post_file: str,
    verified_profile_pool: str,
) -> tuple:
    """
    Filters TikTok profiles and associated posts based on specified criteria.

    Args:
        project_name (str): Name of the project.
        execute_date (str): The date of the pipeline execution, used to create a unique directory name.
        profile_metadata_file (str): Path to the CSV file containing profile metadata.
        post_file (str): Path to the CSV file containing post data.
        verified_profile_pool (str): Path to the CSV file containing verified profiles.

    Returns:
        tuple: A tuple containing two DataFrames:
            - filtered_profiles: DataFrame of profiles that meet the filtering criteria.
            - filtered_posts: DataFrame of posts associated with the filtered profiles.
    """
    profile_metadata = pd.read_csv(
        os.path.join(
            base_dir, "../data", project_name, execution_date, profile_metadata_file
        )
    )
    post_data = pd.read_csv(
        os.path.join(base_dir, "../data", project_name, execution_date, post_file)
    )
    verified_profile_pool = pd.read_csv(
        os.path.join(base_dir, "../data", project_name, verified_profile_pool)
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
    filtered_profiles.to_csv(
        os.path.join(
            base_dir, "../data", project_name, execution_date, profile_metadata_file
        ),
        index=False,
    )

    # Filter posts based on profiles that meet filtering criteria
    filtered_profile_list = filtered_profiles["account_id"].tolist()
    filtered_posts = post_data[
        post_data["account_id"].isin(filtered_profile_list)
    ].reset_index(drop=True)
    filtered_posts.to_csv(
        os.path.join(base_dir, "../data", project_name, execution_date, post_file),
        index=False,
    )

    return filtered_profiles, filtered_posts


if __name__ == "__main__":
    # Step 1: Perform search using predefined list of search terms
    print("Perform keyword search using predefined list of search terms...")
    perform_tiktok_keyword_search(
        project_name=PROJECT_NAME_TIKTOK,
        execution_date=PIPELINE_EXECUTION_DATE,
        search_terms=SEARCH_TERMS_TIKTOK,
        output_file_path=KEYWORD_SEARCH_FILE_TIKTOK,
    )

    # Step 2: Extract profile metadata for search results
    print("Perform profile metadata search for keyword search results...")
    perform_tiktok_profile_metadata_search(
        project_name=PROJECT_NAME_TIKTOK,
        execution_date=PIPELINE_EXECUTION_DATE,
        input_file_path=os.path.join(
            PIPELINE_EXECUTION_DATE, KEYWORD_SEARCH_FILE_TIKTOK
        ),
        output_file_path=PROFILE_METADATA_SEARCH_FILE_TIKTOK,
    )

    # Step 3: Filter profiles that do not meet filtering criteria
    print(
        "Filter TikTok profiles based on follower count, video count, and verified finfluencer list and perform video transcription..."
    )
    filter_tiktok_profiles(
        project_name=PROJECT_NAME_TIKTOK,
        execution_date=PIPELINE_EXECUTION_DATE,
        profile_metadata_file=PROFILE_METADATA_SEARCH_FILE_TIKTOK,
        post_file=KEYWORD_SEARCH_FILE_TIKTOK,
        verified_profile_pool=FINFLUENCER_POOL_FILE_TIKTOK,
    )
    perform_video_transcription(
        project_name=PROJECT_NAME_TIKTOK,
        execution_date=PIPELINE_EXECUTION_DATE,
        video_file=KEYWORD_SEARCH_FILE_TIKTOK,
    )

    # Step 4: Generate expert reflections
    print("Generate expert reflections of potential influencers...")
    generate_expert_reflections(
        project_name=PROJECT_NAME_TIKTOK,
        execution_date=PIPELINE_EXECUTION_DATE,
        role="portfolio_manager",
        profile_metadata_file=PROFILE_METADATA_SEARCH_FILE_TIKTOK,
        post_file=KEYWORD_SEARCH_FILE_TIKTOK,
        output_file=EXPERT_REFLECTION_FILE_TIKTOK,
    )
    generate_expert_reflections(
        project_name=PROJECT_NAME_TIKTOK,
        execution_date=PIPELINE_EXECUTION_DATE,
        role="investment_advisor",
        profile_metadata_file=EXPERT_REFLECTION_FILE_TIKTOK,
        post_file=KEYWORD_SEARCH_FILE_TIKTOK,
        output_file=EXPERT_REFLECTION_FILE_TIKTOK,
    )
    generate_expert_reflections(
        project_name=PROJECT_NAME_TIKTOK,
        execution_date=PIPELINE_EXECUTION_DATE,
        role="financial_analyst",
        profile_metadata_file=EXPERT_REFLECTION_FILE_TIKTOK,
        post_file=KEYWORD_SEARCH_FILE_TIKTOK,
        output_file=EXPERT_REFLECTION_FILE_TIKTOK,
    )
    generate_expert_reflections(
        project_name=PROJECT_NAME_TIKTOK,
        execution_date=PIPELINE_EXECUTION_DATE,
        role="economist",
        profile_metadata_file=EXPERT_REFLECTION_FILE_TIKTOK,
        post_file=KEYWORD_SEARCH_FILE_TIKTOK,
        output_file=EXPERT_REFLECTION_FILE_TIKTOK,
    )

    # Step 5: Conduct onboarding interview to identify financial influencers and add to influencer pool
    print("Perform onboarding interview to identify financial influencers...")
    perform_tiktok_onboarding_interview(
        project_name=PROJECT_NAME_TIKTOK,
        execution_date=PIPELINE_EXECUTION_DATE,
        profile_metadata_file=EXPERT_REFLECTION_FILE_TIKTOK,
        post_file=KEYWORD_SEARCH_FILE_TIKTOK,
        output_file=ONBOARDING_RESULTS_FILE_TIKTOK,
    )
    extract_stock_mentions(
        project_name=PROJECT_NAME_TIKTOK,
        execution_date=PIPELINE_EXECUTION_DATE,
        profile_metadata_file=ONBOARDING_RESULTS_FILE_TIKTOK,
        post_file=KEYWORD_SEARCH_FILE_TIKTOK,
        output_file=ONBOARDING_RESULTS_FILE_TIKTOK,
        interview_type="tiktok_stock_mention",
    )
    update_verified_profile_pool(
        project_name=PROJECT_NAME_TIKTOK,
        execution_date=PIPELINE_EXECUTION_DATE,
        input_file_path=ONBOARDING_RESULTS_FILE_TIKTOK,
        verified_profile_pool=FINFLUENCER_POOL_FILE_TIKTOK,
        prediction_threshold=PREDICTION_THRESHOLD_TIKTOK,
    )

    # Step 6: Perform profile search of identified financial influencers (profile metadata and posts) during search period
    print(
        "Perform profile search of identified financial influencers (profile metadata and recent posts) during search period..."
    )
    perform_tiktok_profile_metadata_search(
        project_name=PROJECT_NAME_TIKTOK,
        execution_date=PIPELINE_EXECUTION_DATE,
        input_file_path=FINFLUENCER_POOL_FILE_TIKTOK,
        output_file_path=FINFLUENCER_PROFILE_METADATA_SEARCH_FILE_TIKTOK,
    )
    perform_tiktok_profile_search(
        project_name=PROJECT_NAME_TIKTOK,
        execution_date=PIPELINE_EXECUTION_DATE,
        input_file_path=FINFLUENCER_POOL_FILE_TIKTOK,
        output_file_path=FINFLUENCER_PROFILE_SEARCH_FILE_TIKTOK,
        start_date=PROFILE_SEARCH_START_DATE,
        end_date=PROFILE_SEARCH_END_DATE,
    )
    perform_video_transcription(
        project_name=PROJECT_NAME_TIKTOK,
        execution_date=PIPELINE_EXECUTION_DATE,
        video_file=FINFLUENCER_PROFILE_SEARCH_FILE_TIKTOK,
    )
    extract_stock_mentions(
        project_name=PROJECT_NAME_TIKTOK,
        execution_date=PIPELINE_EXECUTION_DATE,
        profile_metadata_file=FINFLUENCER_PROFILE_METADATA_SEARCH_FILE_TIKTOK,
        post_file=FINFLUENCER_PROFILE_SEARCH_FILE_TIKTOK,
        output_file=FINFLUENCER_STOCK_MENTIONS_FILE_TIKTOK,
        interview_type="tiktok_stock_mention",
    )

    # Step 7: Conduct interview on financial markets
    print("Conduct digital interview on financial markets...")
    perform_tiktok_finfluencer_interview(
        project_name=PROJECT_NAME_TIKTOK,
        execution_date=PIPELINE_EXECUTION_DATE,
        profile_metadata_file=FINFLUENCER_STOCK_MENTIONS_FILE_TIKTOK,
        post_file=FINFLUENCER_PROFILE_SEARCH_FILE_TIKTOK,
        output_file=FINFLUENCER_POST_INTERVIEW_FILE_TIKTOK,
    )

    # Step 8: Conduct interview on stock recommendations
    print("Conduct digital interview on stock recommendations...")
    perform_tiktok_stock_recommendation_interview(
        project_name=PROJECT_NAME_TIKTOK,
        execution_date=PIPELINE_EXECUTION_DATE,
        profile_metadata_file=FINFLUENCER_STOCK_MENTIONS_FILE_TIKTOK,
        post_file=FINFLUENCER_PROFILE_SEARCH_FILE_TIKTOK,
        finfluencer_pool=FINFLUENCER_POOL_FILE_TIKTOK,
        output_file=FINFLUENCER_STOCK_RECOMMENDATION_FILE_TIKTOK,
    )

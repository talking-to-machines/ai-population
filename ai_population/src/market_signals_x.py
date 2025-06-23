import os
import pandas as pd
import re
import string
import requests
import ast
from requests.auth import HTTPBasicAuth
from tqdm import tqdm
from datetime import datetime, timezone

tqdm.pandas()
from ai_population.config.market_signals_config import (
    PIPELINE_EXECUTION_DATE,
    MIN_FOLLOWER_COUNT,
    NUM_RESULTS_PER_PROFILE,
    MIN_POSTS_COUNT,
    NUM_POST_PER_KEYWORD,
    PROFILE_SEARCH_START_DATE,
    PROFILE_SEARCH_END_DATE,
    PROJECT_NAME_X,
    SEARCH_TERMS_X,
    FINFLUENCER_POOL_FILE_X,
    KEYWORD_SEARCH_FILE_X,
    PROFILE_METADATA_SEARCH_FILE_X,
    ONBOARDING_RESULTS_FILE_X,
    FINFLUENCER_PROFILE_METADATA_SEARCH_FILE_X,
    FINFLUENCER_PROFILE_SEARCH_FILE_X,
    FINFLUENCER_EXPERT_REFLECTION_FILE_X,
    FINFLUENCER_STOCK_MENTIONS_FILE_X,
    FINFLUENCER_POST_INTERVIEW_FILE_X,
    FINFLUENCER_STOCK_RECOMMENDATION_FILE_X,
    RUSSELL_4000_STOCK_TICKER_FILE,
    ONBOARDING_INTERVIEW_REGEX_PATTERNS,
    FINFLUENCER_INTERVIEW_REGEX_PATTERNS,
    STOCK_RECOMMENDATION_OUTPUT_COLUMNS,
)

PROFILE_SEARCH_START_DATE = datetime.strptime(
    PROFILE_SEARCH_START_DATE, "%m-%d-%Y"
).strftime("%Y-%m-%d")
PROFILE_SEARCH_END_DATE = datetime.strptime(
    PROFILE_SEARCH_END_DATE, "%m-%d-%Y"
).strftime("%Y-%m-%d")

from ai_population.config.base_config import (
    X_API_USERNAME,
    X_API_PASSWORD,
    GPT_MODEL,
)
from ai_population.src.utils import (
    extract_llm_responses,
    format_stock_mentions,
    perform_profile_interview,
    update_verified_profile_pool,
    coalesce_columns_by_regex,
    extract_stock_mentions,
    format_stock_recommendations,
)
from ai_population.prompts.prompt_template import (
    x_finfluencer_onboarding_system_prompt,
    x_finfluencer_onboarding_user_prompt,
    x_portfoliomanager_reflection_system_prompt,
    portfoliomanager_reflection_user_prompt,
    x_investmentadvisor_reflection_system_prompt,
    investmentadvisor_reflection_user_prompt,
    x_financialanalyst_reflection_system_prompt,
    financialanalyst_reflection_user_prompt,
    x_economist_reflection_system_prompt,
    economist_reflection_user_prompt,
    x_finfluencer_interview_system_prompt,
    finfluencer_interview_user_prompt,
    stock_recommendation_interview_user_prompt,
)

base_dir = os.path.dirname(os.path.abspath(__file__))


def perform_x_onboarding_interview(
    project_name: str,
    execution_date: str,
    profile_metadata_file: str,
    post_file: str,
    output_file: str,
) -> None:
    """
    Conducts an onboarding interview for financial influencers on platform X, processes the results, and saves the output.

    Args:
        project_name (str): Name of the project for which the onboarding interview is conducted.
        execution_date (str): Date of execution in string format (e.g., 'YYYY-MM-DD').
        profile_metadata_file (str): Path to the CSV file containing profile metadata.
        post_file (str): Path to the post file associated with the interview.
        output_file (str): Name of the output CSV file to save the processed results.

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
        system_prompt_template=x_finfluencer_onboarding_system_prompt,
        user_prompt_template=x_finfluencer_onboarding_user_prompt,
        llm_response_field="onboarding_llm_response",
        interview_type="x_finfluencer_onboarding",
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
    Generates expert reflections for a given project and role by selecting appropriate prompt templates and invoking the profile interview process.

    Args:
        project_name (str): The name of the project for which reflections are being generated.
        execution_date (str): The date of execution in string format.
        role (str): The expert role, must be one of "portfolio_manager", "investment_advisor", "financial_analyst", or "economist".
        profile_metadata_file (str): Path to the profile metadata file.
        post_file (str): Path to the post file associated with the expert.
        output_file (str): Path where the generated reflection output will be saved.

    Raises:
        ValueError: If the provided role is not supported.
    """
    if role == "portfolio_manager":
        system_prompt_template = x_portfoliomanager_reflection_system_prompt
        user_prompt_template = portfoliomanager_reflection_user_prompt
        llm_response_field = "x_finfluencer_expert_reflection_portfoliomanager_response"
        interview_type = "x_finfluencer_expert_reflection_portfoliomanager"

    elif role == "investment_advisor":
        system_prompt_template = x_investmentadvisor_reflection_system_prompt
        user_prompt_template = investmentadvisor_reflection_user_prompt
        llm_response_field = (
            "x_finfluencer_expert_reflection_investmentadvisor_response"
        )
        interview_type = "x_finfluencer_expert_reflection_investmentadvisor"

    elif role == "financial_analyst":
        system_prompt_template = x_financialanalyst_reflection_system_prompt
        user_prompt_template = financialanalyst_reflection_user_prompt
        llm_response_field = "x_finfluencer_expert_reflection_financialanalyst_response"
        interview_type = "x_finfluencer_expert_reflection_financialanalyst"

    elif role == "economist":
        system_prompt_template = x_economist_reflection_system_prompt
        user_prompt_template = economist_reflection_user_prompt
        llm_response_field = "x_finfluencer_expert_reflection_economist_response"
        interview_type = "x_finfluencer_expert_reflection_economist"

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


def perform_x_finfluencer_interview(
    project_name: str,
    execution_date: str,
    profile_metadata_file: str,
    post_file: str,
    output_file: str,
) -> None:
    """
    Conducts an interview process for X (Twitter) finfluencers, processes the results, and saves the formatted output.

    This function performs the following steps:
    1. Runs a profile interview using the provided project and execution details, metadata, and post files.
    2. Loads the interview results from a CSV file.
    3. Extracts and processes the LLM responses from the interview results.
    4. Merges identical columns in the results based on predefined regex patterns.
    5. Saves the formatted interview results back to the output CSV file.

    Args:
        project_name (str): Name of the project directory.
        execution_date (str): Date of execution, used for organizing output files.
        profile_metadata_file (str): Path to the profile metadata CSV file.
        post_file (str): Path to the file containing posts to be used in the interview.
        output_file (str): Name of the output CSV file to save the interview results.

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
        system_prompt_template=x_finfluencer_interview_system_prompt,
        user_prompt_template=finfluencer_interview_user_prompt,
        llm_response_field="x_finfluencer_interview",
        interview_type="x_finfluencer_interview",
    )

    # Preprocess post interview results
    post_interview_results = pd.read_csv(
        os.path.join(base_dir, "../data", project_name, execution_date, output_file)
    )
    extracted_responses = post_interview_results["x_finfluencer_interview"].apply(
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


def perform_x_stock_recommendation_interview(
    project_name: str,
    execution_date: str,
    profile_metadata_file: str,
    post_file: str,
    finfluencer_pool: str,
    output_file: str,
) -> None:
    """
    Performs an interview process to generate and verify stock recommendations from finfluencer profiles on X (formerly Twitter).

    This function processes profile metadata and finfluencer pool data to extract, format, and verify stock mentions. It saves the formatted data, performs an interview process using a language model, and outputs verified stock recommendations.

    Args:
        project_name (str): Name of the project directory.
        execution_date (str): Date of execution, used for organizing data files.
        profile_metadata_file (str): Filename for the profile metadata CSV.
        post_file (str): Filename for the posts CSV.
        finfluencer_pool (str): Filename for the finfluencer pool CSV.
        output_file (str): Filename for saving the output CSV.

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
        profile_stock_mentions["url"] = (
            "https://x.com/" + profile_metadata.loc[i, "account_id"]
        )
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
        system_prompt_template=x_finfluencer_interview_system_prompt,
        user_prompt_template=stock_recommendation_interview_user_prompt,
        llm_response_field="x_finfluencer_stock_recommendation",
        interview_type="x_finfluencer_stock_recommendation",
    )

    stock_recommendations = pd.read_csv(
        os.path.join(base_dir, "../data", project_name, execution_date, output_file)
    )

    # Extract stock recommendation responses
    extracted_responses = stock_recommendations[
        "x_finfluencer_stock_recommendation"
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


def extract_hashtags(entity_dict: dict) -> str:
    """
    Extracts unique hashtags from a string representation of a dictionary.

    Args:
        entity_dict (dict): A dictionary that may contain a "hashtags" key.
                          The "hashtags" key should map to a list of dictionaries, each with a "text" key.

    Returns:
        str: A comma-separated string of unique hashtag texts if present, otherwise an empty string.
    """
    try:
        if "hashtags" in entity_dict:
            hashtags = list(
                set([hashtag["text"] for hashtag in entity_dict["hashtags"]])
            )
            return ", ".join(hashtags)
        else:
            return ""
    except:
        return ""


def extract_tagged_users(entity_dict: dict) -> str:
    """
    Extracts and returns a comma-separated string of unique user names mentioned in the given entity string.

    Args:
        entity_dict (dict): A dictionary containing entity information,
                          expected to include a "user_mentions" key with a list of user mention dictionaries.

    Returns:
        str: A comma-separated string of unique user names if "user_mentions" exists, otherwise an empty string.
    """
    try:
        if "user_mentions" in entity_dict:
            user_mentions = list(
                set(
                    [
                        user_mention["name"]
                        for user_mention in entity_dict["user_mentions"]
                    ]
                )
            )
            return ", ".join(user_mentions)
        else:
            return ""
    except:
        return ""


def perform_x_keyword_search(
    project_name: str,
    execution_date: str,
    search_terms: list,
    output_file_path: str,
) -> pd.DataFrame:
    """
    Performs a keyword search using the X (formerly Twitter) API for a given list of search terms, processes the results, and saves them to a CSV file.

    Args:
        project_name (str): Name of the project, used to organize output data into subfolders.
        execution_date (str): Date of execution, used to further organize output data.
        search_terms (list): List of keywords or search terms to query.
        output_file_path (str): Name of the CSV file to save the search results.

    Returns:
        pd.DataFrame: DataFrame containing the search results, including extracted account IDs, hashtags, and tagged users.
    """

    def batched(iterable, n):
        """Yield successive n-sized batches from iterable."""
        for i in range(0, len(iterable), n):
            yield iterable[i : i + n]

    # Create the project subfolder within the data folder if it does not exist
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(base_dir, "../data"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "../data", project_name), exist_ok=True)
    os.makedirs(
        os.path.join(base_dir, "../data", project_name, execution_date), exist_ok=True
    )

    # Perform keyword search in batches of 5 (due to limitations of API call)
    all_search_results = []
    for batch_terms in batched(search_terms, 5):
        response = requests.get(
            "https://abundance.it.com/get_tweets_by_search_term",
            params={
                "search_term": batch_terms,
                "or_operator": 0,
                "max_tweets": NUM_POST_PER_KEYWORD * len(batch_terms),
            },
            auth=HTTPBasicAuth(X_API_USERNAME, X_API_PASSWORD),
        )
        all_search_results += response.json()

    keyword_search_results = pd.DataFrame(all_search_results)
    keyword_search_results = keyword_search_results.drop_duplicates(
        subset="id"
    ).reset_index(drop=True)
    keyword_search_results["account_id"] = keyword_search_results["author"].apply(
        lambda x: x.get("userName")
    )
    keyword_search_results["hashtags"] = keyword_search_results["entities"].apply(
        extract_hashtags
    )
    keyword_search_results["tagged_users"] = keyword_search_results["entities"].apply(
        extract_tagged_users
    )
    keyword_search_results.to_csv(
        os.path.join(
            base_dir, "../data", project_name, execution_date, output_file_path
        ),
        index=False,
    )

    return keyword_search_results


def perform_x_profile_search(
    project_name: str,
    execution_date: str,
    input_file_path: str,
    output_file_path: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Performs a profile search for a list of account IDs, retrieves tweets for each profile, and saves the results to a CSV file.

    Args:
        project_name (str): Name of the project, used to organize data directories.
        execution_date (str): Date of execution, used to create a subdirectory for output.
        input_file_path (str): Path to the CSV file containing account IDs (relative to the project data directory).
        output_file_path (str): Name of the output CSV file to save the search results.
        start_date (str): Start date for the search.
        end_date (str): End date for the search.

    Returns:
        pd.DataFrame: DataFrame containing the search results for all profiles.
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

    # Peform profile search
    response_list = []
    for profile in tqdm(profile_list):
        response = requests.get(
            "https://abundance.it.com/get_tweets",
            params={
                "user": profile,
                "max_tweets_per_user": NUM_RESULTS_PER_PROFILE,
            },
            auth=HTTPBasicAuth(X_API_USERNAME, X_API_PASSWORD),
        )
        response_list += response.json()[0]

    profile_search_results = pd.DataFrame(response_list)
    profile_search_results["account_id"] = profile_search_results["author"].apply(
        lambda x: x.get("userName")
    )
    profile_search_results["hashtags"] = profile_search_results["entities"].apply(
        extract_hashtags
    )
    profile_search_results["tagged_users"] = profile_search_results["entities"].apply(
        extract_tagged_users
    )

    # Filter posts that happen before start_date
    profile_search_results["createdAt"] = pd.to_datetime(
        profile_search_results["createdAt"], format="%a %b %d %H:%M:%S %z %Y"
    )
    filtered_profile_search_results = profile_search_results[
        profile_search_results["createdAt"]
        >= datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    ].reset_index(drop=True)

    filtered_profile_search_results.to_csv(
        os.path.join(
            base_dir, "../data", project_name, execution_date, output_file_path
        ),
        index=False,
    )

    return profile_search_results


def perform_x_profile_metadata_search(
    project_name: str,
    execution_date: str,
    input_file_path: str,
    output_file_path: str = "",
) -> pd.DataFrame:
    """
    Performs a profile metadata search for a list of account IDs from an input CSV file and saves the results to an output CSV file.

    Args:
        project_name (str): Name of the project, used to organize data directories.
        execution_date (str): Date of execution, used to organize data directories.
        input_file_path (str): Path to the input CSV file containing 'account_id' column.
        output_file_path (str, optional): Path to the output CSV file where results will be saved. Defaults to "".

    Returns:
        pd.DataFrame: DataFrame containing the profile metadata search results.
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

    # Perform profile metadata search
    response_list = []
    for profile in tqdm(profile_list):
        response = requests.get(
            "https://abundance.it.com/get_user_info",
            params={
                "user": profile,
            },
            auth=HTTPBasicAuth(X_API_USERNAME, X_API_PASSWORD),
        )
        response_list += response.json()

    profile_metadata_search_results = pd.DataFrame(response_list)
    profile_metadata_search_results.rename(
        columns={"userName": "account_id"}, inplace=True
    )
    profile_metadata_search_results.to_csv(
        os.path.join(
            base_dir, "../data", project_name, execution_date, output_file_path
        ),
        index=False,
    )

    return profile_metadata_search_results


def filter_x_profiles(
    project_name: str,
    execution_date: str,
    profile_metadata_file: str,
    post_file: str,
    verified_profile_pool: str,
) -> tuple:
    """
    Filters profile and post data based on specified criteria and updates the corresponding CSV files.

    Args:
        project_name (str): Name of the project directory.
        execution_date (str): Date string specifying the execution context.
        profile_metadata_file (str): Filename of the profile metadata CSV.
        post_file (str): Filename of the post data CSV.
        verified_profile_pool (str): Filename of the CSV containing verified profile IDs.

    Returns:
        tuple: A tuple containing:
            - filtered_profiles (pd.DataFrame): DataFrame of profiles that meet the filtering criteria.
            - filtered_posts (pd.DataFrame): DataFrame of posts corresponding to the filtered profiles.
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
        & (
            profile_metadata["statusesCount"] >= MIN_POSTS_COUNT
        )  # Minimum number of posts
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

    # Filter posts files based on profiles that meet filtering criteria
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
    perform_x_keyword_search(
        project_name=PROJECT_NAME_X,
        execution_date=PIPELINE_EXECUTION_DATE,
        search_terms=SEARCH_TERMS_X,
        output_file_path=KEYWORD_SEARCH_FILE_X,
    )

    # Step 2: Extract profile metadata for search results
    print("Perform profile metadata search for keyword search results...")
    perform_x_profile_metadata_search(
        project_name=PROJECT_NAME_X,
        execution_date=PIPELINE_EXECUTION_DATE,
        input_file_path=os.path.join(PIPELINE_EXECUTION_DATE, KEYWORD_SEARCH_FILE_X),
        output_file_path=PROFILE_METADATA_SEARCH_FILE_X,
    )

    # Step 3: Filter profiles that do not meet filtering criteria
    print(
        "Filter X profiles based on follower count, post count, and verified finfluencer list..."
    )
    filter_x_profiles(
        project_name=PROJECT_NAME_X,
        execution_date=PIPELINE_EXECUTION_DATE,
        profile_metadata_file=PROFILE_METADATA_SEARCH_FILE_X,
        post_file=KEYWORD_SEARCH_FILE_X,
        verified_profile_pool=FINFLUENCER_POOL_FILE_X,
    )

    # Step 4: Conduct onboarding interview to identify financial influencers and add to influencer pool
    print("Perform onboarding interview to identify financial influencers...")
    perform_x_onboarding_interview(
        project_name=PROJECT_NAME_X,
        execution_date=PIPELINE_EXECUTION_DATE,
        profile_metadata_file=PROFILE_METADATA_SEARCH_FILE_X,
        post_file=KEYWORD_SEARCH_FILE_X,
        output_file=ONBOARDING_RESULTS_FILE_X,
    )
    extract_stock_mentions(
        project_name=PROJECT_NAME_X,
        execution_date=PIPELINE_EXECUTION_DATE,
        input_file=ONBOARDING_RESULTS_FILE_X,
        output_file=ONBOARDING_RESULTS_FILE_X,
    )
    update_verified_profile_pool(
        project_name=PROJECT_NAME_X,
        execution_date=PIPELINE_EXECUTION_DATE,
        input_file_path=ONBOARDING_RESULTS_FILE_X,
        verified_profile_pool=FINFLUENCER_POOL_FILE_X,
    )

    # Step 5: Perform profile search of identified financial influencers (profile metadata and posts)
    print("Perform profile search of identified financial influencers...")
    perform_x_profile_metadata_search(
        project_name=PROJECT_NAME_X,
        execution_date=PIPELINE_EXECUTION_DATE,
        input_file_path=FINFLUENCER_POOL_FILE_X,
        output_file_path=FINFLUENCER_PROFILE_METADATA_SEARCH_FILE_X,
    )
    perform_x_profile_search(
        project_name=PROJECT_NAME_X,
        execution_date=PIPELINE_EXECUTION_DATE,
        input_file_path=FINFLUENCER_POOL_FILE_X,
        output_file_path=FINFLUENCER_PROFILE_SEARCH_FILE_X,
        start_date=PROFILE_SEARCH_START_DATE,
        end_date=PROFILE_SEARCH_END_DATE,
    )

    # Step 6: Generate expert reflections
    print("Generate expert reflections of financial influencers...")
    generate_expert_reflections(
        project_name=PROJECT_NAME_X,
        execution_date=PIPELINE_EXECUTION_DATE,
        role="portfolio_manager",
        profile_metadata_file=FINFLUENCER_PROFILE_METADATA_SEARCH_FILE_X,
        post_file=FINFLUENCER_PROFILE_SEARCH_FILE_X,
        output_file=FINFLUENCER_EXPERT_REFLECTION_FILE_X,
    )
    generate_expert_reflections(
        project_name=PROJECT_NAME_X,
        execution_date=PIPELINE_EXECUTION_DATE,
        role="investment_advisor",
        profile_metadata_file=FINFLUENCER_EXPERT_REFLECTION_FILE_X,
        post_file=FINFLUENCER_PROFILE_SEARCH_FILE_X,
        output_file=FINFLUENCER_EXPERT_REFLECTION_FILE_X,
    )
    generate_expert_reflections(
        project_name=PROJECT_NAME_X,
        execution_date=PIPELINE_EXECUTION_DATE,
        role="financial_analyst",
        profile_metadata_file=FINFLUENCER_EXPERT_REFLECTION_FILE_X,
        post_file=FINFLUENCER_PROFILE_SEARCH_FILE_X,
        output_file=FINFLUENCER_EXPERT_REFLECTION_FILE_X,
    )
    generate_expert_reflections(
        project_name=PROJECT_NAME_X,
        execution_date=PIPELINE_EXECUTION_DATE,
        role="economist",
        profile_metadata_file=FINFLUENCER_EXPERT_REFLECTION_FILE_X,
        post_file=FINFLUENCER_PROFILE_SEARCH_FILE_X,
        output_file=FINFLUENCER_EXPERT_REFLECTION_FILE_X,
    )

    # Step 7: Extract stock mentions from financial influencers' past posts
    print("Extract stock recommendations...")
    extract_stock_mentions(
        project_name=PROJECT_NAME_X,
        execution_date=PIPELINE_EXECUTION_DATE,
        input_file=FINFLUENCER_EXPERT_REFLECTION_FILE_X,
        output_file=FINFLUENCER_STOCK_MENTIONS_FILE_X,
    )

    # Step 8: Conduct interview on financial markets
    print("Conduct digital interview on financial markets...")
    perform_x_finfluencer_interview(
        project_name=PROJECT_NAME_X,
        execution_date=PIPELINE_EXECUTION_DATE,
        profile_metadata_file=FINFLUENCER_STOCK_MENTIONS_FILE_X,
        post_file=FINFLUENCER_PROFILE_SEARCH_FILE_X,
        output_file=FINFLUENCER_POST_INTERVIEW_FILE_X,
    )

    # Step 9: Conduct interview on stock recommendations
    print("Conduct digital interview on stock recommendations...")
    perform_x_stock_recommendation_interview(
        project_name=PROJECT_NAME_X,
        execution_date=PIPELINE_EXECUTION_DATE,
        profile_metadata_file=FINFLUENCER_STOCK_MENTIONS_FILE_X,
        post_file=FINFLUENCER_PROFILE_SEARCH_FILE_X,
        finfluencer_pool=FINFLUENCER_POOL_FILE_X,
        output_file=FINFLUENCER_STOCK_RECOMMENDATION_FILE_X,
    )

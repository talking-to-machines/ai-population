import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime

tqdm.pandas()
from ai_population.config.market_signals_config import (
    PIPELINE_EXECUTION_DATE,
    NUM_POSTS_PER_PROFILE,
    PROFILE_SEARCH_START_DATE,
    PROFILE_SEARCH_END_DATE,
    PROJECT_NAME_X,
    FINFLUENCER_POOL_FILE_X,
    FINFLUENCER_PROFILE_METADATA_SEARCH_FILE_X,
    FINFLUENCER_PROFILE_SEARCH_FILE_X,
    FINFLUENCER_STOCK_MENTIONS_FILE_X,
    FINFLUENCER_POST_INTERVIEW_FILE_X,
    FINFLUENCER_STOCK_RECOMMENDATION_FILE_X,
    FINFLUENCER_INTERVIEW_REGEX_PATTERNS,
    STOCK_RECOMMENDATION_OUTPUT_COLUMNS,
)

PROFILE_SEARCH_START_DATE = datetime.strptime(
    PROFILE_SEARCH_START_DATE, "%m-%d-%Y"
).strftime("%Y-%m-%d")
PROFILE_SEARCH_END_DATE = datetime.strptime(
    PROFILE_SEARCH_END_DATE, "%m-%d-%Y"
).strftime("%Y-%m-%d")

from ai_population.config.base_config import GPT_MODEL
from ai_population.src.utils import (
    extract_llm_responses,
    format_stock_mentions,
    perform_profile_interview,
    coalesce_columns_by_regex,
    extract_stock_mentions,
    format_stock_recommendations,
    perform_x_profile_metadata_search,
    perform_x_profile_search,
)
from ai_population.prompts.prompt_template import (
    x_finfluencer_interview_system_prompt,
    finfluencer_interview_user_prompt,
    stock_recommendation_interview_user_prompt,
)

base_dir = os.path.dirname(os.path.abspath(__file__))
LOCAL_PROFILE_METADATA_FILE = os.path.join(
    base_dir,
    "../data/market-signals-x/august-pilot/x_finfluencer_profile_metadata.csv",
)
LOCAL_PROFILE_POST_FILE = os.path.join(
    base_dir,
    "../data/market-signals-x/august-pilot/x_finfluencer_profile_search.csv",
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

    # Include LLM model information
    post_interview_results["model"] = GPT_MODEL

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

    # Include LLM model information
    valid_stock_recommendations["model"] = GPT_MODEL

    # Save verified stock recommendations
    valid_stock_recommendations[STOCK_RECOMMENDATION_OUTPUT_COLUMNS].to_csv(
        os.path.join(base_dir, "../data", project_name, execution_date, output_file),
        index=False,
    )


if __name__ == "__main__":
    # Step 1: Perform profile search of identified financial influencers (profile metadata and posts)
    print(
        "Step 1: Perform profile search of identified financial influencers (profile metadata and recent posts) during the search period."
    )
    perform_x_profile_metadata_search(
        project_name=PROJECT_NAME_X,
        execution_date=PIPELINE_EXECUTION_DATE,
        input_file=FINFLUENCER_POOL_FILE_X,
        output_file=FINFLUENCER_PROFILE_METADATA_SEARCH_FILE_X,
        local_file=LOCAL_PROFILE_METADATA_FILE,
    )
    perform_x_profile_search(
        project_name=PROJECT_NAME_X,
        execution_date=PIPELINE_EXECUTION_DATE,
        input_file=FINFLUENCER_POOL_FILE_X,
        output_file=FINFLUENCER_PROFILE_SEARCH_FILE_X,
        start_date=PROFILE_SEARCH_START_DATE,
        end_date=PROFILE_SEARCH_END_DATE,
        num_posts_per_profile=NUM_POSTS_PER_PROFILE,
        local_file=LOCAL_PROFILE_POST_FILE,
    )
    extract_stock_mentions(
        project_name=PROJECT_NAME_X,
        execution_date=PIPELINE_EXECUTION_DATE,
        profile_metadata_file=FINFLUENCER_PROFILE_METADATA_SEARCH_FILE_X,
        post_file=FINFLUENCER_PROFILE_SEARCH_FILE_X,
        output_file=FINFLUENCER_STOCK_MENTIONS_FILE_X,
        interview_type="x_stock_mention",
    )

    # Step 2: Conduct interview on financial markets
    print("Step 2: Conduct digital interview on financial markets.")
    perform_x_finfluencer_interview(
        project_name=PROJECT_NAME_X,
        execution_date=PIPELINE_EXECUTION_DATE,
        profile_metadata_file=FINFLUENCER_STOCK_MENTIONS_FILE_X,
        post_file=FINFLUENCER_PROFILE_SEARCH_FILE_X,
        output_file=FINFLUENCER_POST_INTERVIEW_FILE_X,
    )

    # Step 3: Conduct interview on stock recommendations
    print("Step 3: Conduct digital interview on stock recommendations.")
    perform_x_stock_recommendation_interview(
        project_name=PROJECT_NAME_X,
        execution_date=PIPELINE_EXECUTION_DATE,
        profile_metadata_file=FINFLUENCER_STOCK_MENTIONS_FILE_X,
        post_file=FINFLUENCER_PROFILE_SEARCH_FILE_X,
        finfluencer_pool=FINFLUENCER_POOL_FILE_X,
        output_file=FINFLUENCER_STOCK_RECOMMENDATION_FILE_X,
    )

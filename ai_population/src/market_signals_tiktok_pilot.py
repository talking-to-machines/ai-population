import os
import pandas as pd
from tqdm import tqdm

tqdm.pandas()
from ai_population.config.market_signals_config import (
    PIPELINE_EXECUTION_DATE,
    NUM_POSTS_PER_PROFILE,
    PROFILE_SEARCH_START_DATE,
    PROFILE_SEARCH_END_DATE,
    PROJECT_NAME_TIKTOK,
    FINFLUENCER_POOL_FILE_TIKTOK,
    FINFLUENCER_PROFILE_METADATA_SEARCH_FILE_TIKTOK,
    FINFLUENCER_PROFILE_SEARCH_FILE_TIKTOK,
    FINFLUENCER_STOCK_MENTIONS_FILE_TIKTOK,
    FINFLUENCER_POST_INTERVIEW_FILE_TIKTOK,
    FINFLUENCER_STOCK_RECOMMENDATION_FILE_TIKTOK,
    FINFLUENCER_INTERVIEW_REGEX_PATTERNS,
    STOCK_RECOMMENDATION_OUTPUT_COLUMNS,
)
from ai_population.config.base_config import GPT_MODEL
from ai_population.src.utils import (
    extract_llm_responses,
    format_stock_mentions,
    perform_profile_interview,
    perform_video_transcription,
    coalesce_columns_by_regex,
    extract_stock_mentions,
    format_stock_recommendations,
    perform_tiktok_profile_metadata_search,
    perform_tiktok_profile_search,
)
from ai_population.prompts.prompt_template import (
    tiktok_finfluencer_interview_system_prompt,
    finfluencer_interview_user_prompt,
    stock_recommendation_interview_user_prompt,
)

base_dir = os.path.dirname(os.path.abspath(__file__))
LOCAL_PROFILE_METADATA_FILE = os.path.join(
    base_dir,
    "../data/market-signals-tiktok/august-pilot/tiktok_finfluencer_profile_metadata.csv",
)
LOCAL_PROFILE_POST_FILE = os.path.join(
    base_dir,
    "../data/market-signals-tiktok/august-pilot/tiktok_finfluencer_profile_search.csv",
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
        model_name=GPT_MODEL,
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

    # Include LLM model information
    post_interview_results["model"] = GPT_MODEL

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
        model_name=GPT_MODEL,
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

    # Include LLM model information
    valid_stock_recommendations["model"] = GPT_MODEL

    # Save verified stock recommendations
    valid_stock_recommendations[STOCK_RECOMMENDATION_OUTPUT_COLUMNS].to_csv(
        os.path.join(base_dir, "../data", project_name, execution_date, output_file),
        index=False,
    )


if __name__ == "__main__":
    # Step 1: Perform profile search of identified financial influencers (profile metadata and posts) during search period
    print(
        "Step 1: Perform profile search of identified financial influencers (profile metadata and recent posts) during search period."
    )
    perform_tiktok_profile_metadata_search(
        project_name=PROJECT_NAME_TIKTOK,
        execution_date=PIPELINE_EXECUTION_DATE,
        input_file=FINFLUENCER_POOL_FILE_TIKTOK,
        output_file=FINFLUENCER_PROFILE_METADATA_SEARCH_FILE_TIKTOK,
        local_file=LOCAL_PROFILE_METADATA_FILE,
    )
    perform_tiktok_profile_search(
        project_name=PROJECT_NAME_TIKTOK,
        execution_date=PIPELINE_EXECUTION_DATE,
        input_file=FINFLUENCER_POOL_FILE_TIKTOK,
        output_file=FINFLUENCER_PROFILE_SEARCH_FILE_TIKTOK,
        start_date=PROFILE_SEARCH_START_DATE,
        end_date=PROFILE_SEARCH_END_DATE,
        num_posts_per_profile=NUM_POSTS_PER_PROFILE,
        local_file=LOCAL_PROFILE_POST_FILE,
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

    # Step 2: Conduct interview on financial markets
    print("Step 2: Conduct digital interview on financial markets.")
    perform_tiktok_finfluencer_interview(
        project_name=PROJECT_NAME_TIKTOK,
        execution_date=PIPELINE_EXECUTION_DATE,
        profile_metadata_file=FINFLUENCER_STOCK_MENTIONS_FILE_TIKTOK,
        post_file=FINFLUENCER_PROFILE_SEARCH_FILE_TIKTOK,
        output_file=FINFLUENCER_POST_INTERVIEW_FILE_TIKTOK,
    )

    # Step 3: Conduct interview on stock recommendations
    print("Step 3: Conduct digital interview on stock recommendations.")
    perform_tiktok_stock_recommendation_interview(
        project_name=PROJECT_NAME_TIKTOK,
        execution_date=PIPELINE_EXECUTION_DATE,
        profile_metadata_file=FINFLUENCER_STOCK_MENTIONS_FILE_TIKTOK,
        post_file=FINFLUENCER_PROFILE_SEARCH_FILE_TIKTOK,
        finfluencer_pool=FINFLUENCER_POOL_FILE_TIKTOK,
        output_file=FINFLUENCER_STOCK_RECOMMENDATION_FILE_TIKTOK,
    )

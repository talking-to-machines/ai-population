import argparse
import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

tqdm.pandas()
from ai_population.config.market_signals_config import (
    PIPELINE_EXECUTION_DATE,
    MIN_FOLLOWER_COUNT,
    MIN_VIDEO_COUNT,
    NUM_POSTS_PER_KEYWORD,
    NUM_POSTS_PER_PROFILE,
    LATEST_K_POSTS_PER_PROFILE,
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
    FINFLUENCER_HISTORICAL_PROFILE_SEARCH_FILE_TIKTOK,
    FINFLUENCER_STOCK_MENTIONS_FILE_TIKTOK,
    FINFLUENCER_POST_INTERVIEW_FILE_TIKTOK,
    FINFLUENCER_STOCK_RECOMMENDATION_FILE_TIKTOK,
    FINFLUENCER_DAILY_STOCK_PICK_FILE_TIKTOK,
    ONBOARDING_INTERVIEW_REGEX_PATTERNS,
    FINFLUENCER_INTERVIEW_REGEX_PATTERNS,
    FINFLUENCER_DAILY_STOCK_PICK_REGEX_PATTERNS,
    STOCK_RECOMMENDATION_OUTPUT_COLUMNS,
    PREDICTION_THRESHOLD_TIKTOK,
    FILTER_ORIGINAL_PROFILES_TIKTOK,
    ORIGINAL_PROFILES_TIKTOK,
    DAILY_STOCK_PICK_PROFILES_TIKTOK,
)
from ai_population.config.base_config import GPT_MODEL
from ai_population.src.utils import (
    extract_llm_responses,
    format_stock_mentions,
    perform_profile_interview,
    perform_video_transcription,
    update_verified_profile_pool,
    coalesce_columns_by_regex,
    extract_stock_mentions,
    format_stock_recommendations,
    perform_tiktok_keyword_search,
    perform_tiktok_profile_metadata_search,
    perform_tiktok_profile_search,
)
from ai_population.prompts.prompt_template import (
    tiktok_finfluencer_onboarding_system_prompt,
    tiktok_finfluencer_onboarding_user_prompt,
    tiktok_investmentadvisor_reflection_system_prompt,
    investmentadvisor_reflection_user_prompt,
    tiktok_finfluencer_interview_system_prompt,
    finfluencer_interview_user_prompt,
    stock_recommendation_interview_user_prompt,
    daily_stock_pick_user_prompts,
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
        model_name=GPT_MODEL,
        profile_metadata_file=profile_metadata_file,
        post_file=post_file,
        output_file=output_file,
        system_prompt_template=tiktok_finfluencer_onboarding_system_prompt,
        user_prompt_template=tiktok_finfluencer_onboarding_user_prompt,
        llm_response_field="onboarding_llm_response",
        interview_type="tiktok_finfluencer_onboarding",
        enable_web_search=True,
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
            "investment_advisor".
        profile_metadata_file (str): Path to the profile metadata file.
        post_file (str): Path to the TikTok post file.
        output_file (str): Path where the generated reflection will be saved.

    Raises:
        ValueError: If the provided role is not supported.
    """
    if role == "investment_advisor":
        system_prompt_template = tiktok_investmentadvisor_reflection_system_prompt
        user_prompt_template = investmentadvisor_reflection_user_prompt
        llm_response_field = (
            "tiktok_finfluencer_expert_reflection_investmentadvisor_response"
        )
        interview_type = "tiktok_finfluencer_expert_reflection_investmentadvisor"

    else:
        raise ValueError(f"Role {role} is not supported.")

    perform_profile_interview(
        project_name=project_name,
        execution_date=execution_date,
        model_name=GPT_MODEL,
        profile_metadata_file=profile_metadata_file,
        post_file=post_file,
        output_file=output_file,
        system_prompt_template=system_prompt_template,
        user_prompt_template=user_prompt_template,
        llm_response_field=llm_response_field,
        interview_type=interview_type,
        enable_web_search=True,
    )


def perform_tiktok_finfluencer_interview(
    project_name: str,
    execution_date: str,
    profile_metadata_file: str,
    post_file: str,
    output_file: str,
    filter_original_profiles: bool = False,
    model_name: str = GPT_MODEL,
    provider: str = None,
    enable_web_search: bool = True,
    use_row_query: bool = False,
) -> None:
    """
    Conducts an interview process for TikTok finfluencer profiles using a language model, processes the results, and saves the formatted output.

    Args:
        project_name (str): Name of the project directory.
        execution_date (str): Date of execution, used for organizing output files.
        profile_metadata_file (str): Path to the CSV file containing profile metadata.
        post_file (str): Path to the CSV file containing post data.
        output_file (str): Name of the output CSV file to save interview results.
        filter_original_profiles (bool, optional): If True, filters results to include only original TikTok profiles. Defaults to False.
        model_name (str, optional): Model id used for the interview. Defaults to GPT_MODEL.
        provider (str, optional): Force provider routing (openai | anthropic/claude | xai/grok). Defaults to auto-detection from the model id.
        enable_web_search (bool, optional): Enable provider-native web search during the interview. Defaults to True.
        use_row_query (bool, optional): Force per-row real-time API calls instead of the provider batch API. Defaults to False.

    Returns:
        None

    Side Effects:
        - Reads and writes CSV files to disk.
        - Processes and merges interview results.
        - Optionally filters and saves results for original profiles only.
    """
    perform_profile_interview(
        project_name=project_name,
        execution_date=execution_date,
        model_name=model_name,
        profile_metadata_file=profile_metadata_file,
        post_file=post_file,
        output_file=output_file,
        system_prompt_template=tiktok_finfluencer_interview_system_prompt,
        user_prompt_template=finfluencer_interview_user_prompt,
        llm_response_field="tiktok_finfluencer_interview",
        interview_type="tiktok_finfluencer_interview",
        enable_web_search=enable_web_search,
        use_row_query=use_row_query,
        response_timestamp_col="finfluencer_interview_datetime",
        latest_k_posts=LATEST_K_POSTS_PER_PROFILE,
        batch_timeout_seconds=7200,
        provider=provider,
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
    post_interview_results["model"] = model_name

    # Include timestamp information for when the interview was conducted
    if "finfluencer_interview_datetime" not in post_interview_results.columns:
        post_interview_results["finfluencer_interview_datetime"] = (
            pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        )

    # Save formatted interview results
    if filter_original_profiles:
        filtered_post_interview_results = post_interview_results[
            post_interview_results["account_id"].isin(ORIGINAL_PROFILES_TIKTOK)
        ].reset_index(drop=True)
        filtered_post_interview_results.to_csv(
            os.path.join(
                base_dir, "../data", project_name, execution_date, output_file
            ),
            index=False,
        )

    post_interview_results.to_csv(
        os.path.join(
            base_dir,
            "../data",
            project_name,
            execution_date,
            output_file[:-4] + "_full.csv",
        ),
        index=False,
    )


def perform_tiktok_stock_recommendation_interview(
    project_name: str,
    execution_date: str,
    profile_metadata_file: str,
    post_file: str,
    finfluencer_pool: str,
    output_file: str,
    filter_original_profiles: bool = False,
    model_name: str = GPT_MODEL,
    provider: str = None,
    enable_web_search: bool = True,
    use_row_query: bool = False,
) -> None:
    """
    Performs a TikTok stock recommendation interview process by preparing, formatting, and verifying stock mention data from TikTok finfluencer profiles.

    This function processes profile metadata and finfluencer pool data to extract stock mentions, merges relevant information, and saves the formatted data. It then conducts an interview process using a language model to verify stock recommendations, extracts and formats the responses, sorts and filters the results, and saves the final verified stock recommendations.

    Args:
        project_name (str): Name of the project directory.
        execution_date (str): Date of execution, used for organizing data files.
        profile_metadata_file (str): Filename of the profile metadata CSV.
        post_file (str): Filename of the post data CSV.
        finfluencer_pool (str): Filename of the finfluencer pool CSV.
        output_file (str): Filename for saving the output CSV.
        filter_original_profiles (bool, optional): If True, only retain recommendations from original profiles. Defaults to False.
        model_name (str, optional): Model id used for the interview. Defaults to GPT_MODEL.
        provider (str, optional): Force provider routing (openai | anthropic/claude | xai/grok). Defaults to auto-detection from the model id.
        enable_web_search (bool, optional): Enable provider-native web search during the interview. Defaults to True.
        use_row_query (bool, optional): Force per-row real-time API calls instead of the provider batch API. Defaults to False.

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
        model_name=model_name,
        profile_metadata_file=output_file,
        post_file=post_file,
        output_file=output_file,
        system_prompt_template=tiktok_finfluencer_interview_system_prompt,
        user_prompt_template=stock_recommendation_interview_user_prompt,
        llm_response_field="tiktok_finfluencer_stock_recommendation",
        interview_type="tiktok_finfluencer_stock_recommendation",
        enable_web_search=enable_web_search,
        use_row_query=use_row_query,
        response_timestamp_col="stock_recommendation_interview_datetime",
        batch_timeout_seconds=7200,
        provider=provider,
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
    valid_stock_recommendations["model"] = model_name

    # Include timestamp information for when the interview was conducted
    if (
        "stock_recommendation_interview_datetime"
        not in valid_stock_recommendations.columns
    ):
        valid_stock_recommendations["stock_recommendation_interview_datetime"] = (
            pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        )

    # Save verified stock recommendations
    if filter_original_profiles:
        filtered_stock_recommendations = valid_stock_recommendations[
            valid_stock_recommendations["account_id"].isin(ORIGINAL_PROFILES_TIKTOK)
        ].reset_index(drop=True)
        filtered_stock_recommendations[STOCK_RECOMMENDATION_OUTPUT_COLUMNS].to_csv(
            os.path.join(
                base_dir, "../data", project_name, execution_date, output_file
            ),
            index=False,
        )

    valid_stock_recommendations[STOCK_RECOMMENDATION_OUTPUT_COLUMNS].to_csv(
        os.path.join(
            base_dir,
            "../data",
            project_name,
            execution_date,
            output_file[:-4] + "_full.csv",
        ),
        index=False,
    )


def perform_tiktok_daily_stock_pick_interview(
    project_name: str,
    execution_date: str,
    profile_metadata_file: str,
    post_file: str,
    output_file: str,
    filter_original_profiles: bool = False,
    model_name: str = GPT_MODEL,
    provider: str = None,
    enable_web_search: bool = True,
    use_row_query: bool = False,
) -> None:
    """
    Conducts the TikTok daily stock pick interview across the sampled finfluencer profiles.

    The interview is chunked into multiple user prompts (one per entry in
    ``daily_stock_pick_user_prompts``); each chunk is run independently, cached to its own
    output file, then merged back together into a single result.

    Args:
        project_name (str): Name of the project directory.
        execution_date (str): Date of execution, used for organizing output files.
        profile_metadata_file (str): Path to the CSV file containing profile metadata.
        post_file (str): Path to the CSV file containing post data.
        output_file (str): Name of the output CSV file to save interview results.
        filter_original_profiles (bool, optional): If True, filters results to only include original profiles. Defaults to False.
        model_name (str, optional): Model id used for the interview. Defaults to GPT_MODEL.
        provider (str, optional): Force provider routing (openai | anthropic/claude | xai/grok). Defaults to auto-detection from the model id.
        enable_web_search (bool, optional): Enable provider-native web search during the interview. Defaults to True.
        use_row_query (bool, optional): Force per-row real-time API calls instead of the provider batch API. Defaults to False.

    Returns:
        None
    """
    profile_metadata = pd.read_csv(
        os.path.join(
            base_dir, "../data", project_name, execution_date, profile_metadata_file
        )
    )
    sampled_profile_metadata = profile_metadata[
        profile_metadata["account_id"].isin(
            DAILY_STOCK_PICK_PROFILES_TIKTOK + ORIGINAL_PROFILES_TIKTOK
        )
    ].reset_index(drop=True)
    sampled_profile_metadata.to_csv(
        os.path.join(
            base_dir,
            "../data",
            project_name,
            execution_date,
            f"tiktok_finfluencer_sampled_profiles_{execution_date}.csv",
        ),
        index=False,
    )

    def run_daily_stock_pick_interview(idx_prompt):
        idx, user_prompt = idx_prompt
        chunk_output_file = output_file[:-4] + f"_{idx+1}.csv"
        chunk_output_path = os.path.join(
            base_dir, "../data", project_name, execution_date, chunk_output_file
        )
        if os.path.exists(chunk_output_path):
            print(f"Skipping idx={idx+1}: {chunk_output_path} already exists.")
            return
        perform_profile_interview(
            project_name=project_name,
            execution_date=execution_date,
            model_name=model_name,
            profile_metadata_file=f"tiktok_finfluencer_sampled_profiles_{execution_date}.csv",
            post_file=post_file,
            output_file=chunk_output_file,
            system_prompt_template=tiktok_finfluencer_interview_system_prompt,
            user_prompt_template=user_prompt,
            llm_response_field="tiktok_finfluencer_daily_stock_pick",
            interview_type=f"tiktok_finfluencer_daily_stock_pick_{idx+1}",
            enable_web_search=enable_web_search,
            use_row_query=use_row_query,
            response_timestamp_col="daily_stock_pick_interview_datetime",
            latest_k_posts=LATEST_K_POSTS_PER_PROFILE,
            batch_timeout_seconds=4800,
            provider=provider,
        )

    with ThreadPoolExecutor(max_workers=3) as executor:
        list(
            executor.map(
                run_daily_stock_pick_interview, enumerate(daily_stock_pick_user_prompts)
            )
        )

    # Preprocess daily stock pick results
    extracted_responses_list = []
    for idx in tqdm(range(len(daily_stock_pick_user_prompts))):
        daily_stock_pick_chunk = pd.read_csv(
            os.path.join(
                base_dir,
                "../data",
                project_name,
                execution_date,
                output_file[:-4] + f"_{idx+1}.csv",
            )
        )
        extracted_responses = daily_stock_pick_chunk[
            "tiktok_finfluencer_daily_stock_pick"
        ].apply(extract_llm_responses)
        extracted_responses[f"tiktok_finfluencer_daily_stock_pick_{idx+1}"] = (
            daily_stock_pick_chunk["tiktok_finfluencer_daily_stock_pick"]
        )
        extracted_responses_list.append(extracted_responses)

    daily_stock_pick_results = pd.concat(
        [daily_stock_pick_chunk.drop(columns=["tiktok_finfluencer_daily_stock_pick"])]
        + extracted_responses_list,
        axis=1,
    )
    # Merge identical columns from interview response
    daily_stock_pick_results = coalesce_columns_by_regex(
        daily_stock_pick_results, FINFLUENCER_DAILY_STOCK_PICK_REGEX_PATTERNS
    )

    # Include LLM model information
    daily_stock_pick_results["model"] = model_name

    # Include timestamp information for when the interview was conducted
    if "daily_stock_pick_interview_datetime" not in daily_stock_pick_results.columns:
        daily_stock_pick_results["daily_stock_pick_interview_datetime"] = (
            pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        )

    # Save formatted interview results
    if filter_original_profiles:
        filtered_daily_stock_pick_results = daily_stock_pick_results[
            daily_stock_pick_results["account_id"].isin(ORIGINAL_PROFILES_TIKTOK)
        ].reset_index(drop=True)
        filtered_daily_stock_pick_results.to_csv(
            os.path.join(
                base_dir, "../data", project_name, execution_date, output_file
            ),
            index=False,
        )

    daily_stock_pick_results.to_csv(
        os.path.join(
            base_dir,
            "../data",
            project_name,
            execution_date,
            output_file[:-4] + "_full.csv",
        ),
        index=False,
    )


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
    profile_metadata["followers"] = (
        pd.to_numeric(profile_metadata["followers"], errors="coerce")
        .fillna(0.0)
        .astype(float)
    )
    profile_metadata["videos_count"] = (
        pd.to_numeric(profile_metadata["videos_count"], errors="coerce")
        .fillna(0.0)
        .astype(float)
    )
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
    parser = argparse.ArgumentParser(
        description="Run the TikTok market signals interview pipeline.",
    )
    parser.add_argument(
        "--model",
        dest="model_name",
        type=str,
        default=GPT_MODEL,
        help=(
            "Model id used for every interview step. Use an OpenAI model id "
            "(e.g. gpt-5.1-2025-11-13), an Anthropic Claude model id "
            "(e.g. claude-opus-4-7), or an xAI Grok model id "
            "(e.g. grok-4-fast-non-reasoning). The provider is auto-detected "
            "from the model prefix unless --provider is set."
        ),
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "anthropic", "claude", "xai", "grok"],
        default=None,
        help=(
            "Force the provider routing (openai | anthropic/claude | xai/grok). "
            "Defaults to auto-detection from --model."
        ),
    )
    parser.add_argument(
        "--enable-web-search",
        dest="enable_web_search",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable provider-native web search (OpenAI web_search tool, "
            "Anthropic web_search_20250305 tool, xAI Live Search). "
            "Use --no-enable-web-search to disable."
        ),
    )
    parser.add_argument(
        "--use-row-query",
        dest="use_row_query",
        action="store_true",
        default=False,
        help=(
            "Force per-row real-time API calls instead of the provider batch API. "
            "Batch is used by default; this flag is useful for ad-hoc runs or "
            "providers without a batch endpoint."
        ),
    )
    args = parser.parse_args()
    model_name = args.model_name
    provider = args.provider
    enable_web_search = args.enable_web_search
    use_row_query = args.use_row_query

    # Step 1: Perform search using predefined list of search terms
    print("1. Perform keyword search using predefined list of search terms...")
    perform_tiktok_keyword_search(
        project_name=PROJECT_NAME_TIKTOK,
        execution_date=PIPELINE_EXECUTION_DATE,
        search_terms=SEARCH_TERMS_TIKTOK,
        output_file=KEYWORD_SEARCH_FILE_TIKTOK,
        num_posts_per_keyword=NUM_POSTS_PER_KEYWORD,
    )

    # Step 2: Extract profile metadata for search results
    print("2. Perform profile metadata search for keyword search results...")
    perform_tiktok_profile_metadata_search(
        project_name=PROJECT_NAME_TIKTOK,
        execution_date=PIPELINE_EXECUTION_DATE,
        input_file=os.path.join(PIPELINE_EXECUTION_DATE, KEYWORD_SEARCH_FILE_TIKTOK),
        output_file=PROFILE_METADATA_SEARCH_FILE_TIKTOK,
    )

    # Step 3: Filter profiles that do not meet filtering criteria
    print(
        "3. Filter TikTok profiles based on follower count, video count, and verified finfluencer list and perform video transcription..."
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
    print("4. Generate expert reflections of potential influencers...")
    generate_expert_reflections(
        project_name=PROJECT_NAME_TIKTOK,
        execution_date=PIPELINE_EXECUTION_DATE,
        role="investment_advisor",
        profile_metadata_file=PROFILE_METADATA_SEARCH_FILE_TIKTOK,
        post_file=KEYWORD_SEARCH_FILE_TIKTOK,
        output_file=EXPERT_REFLECTION_FILE_TIKTOK,
    )

    # Step 5: Conduct onboarding interview to identify financial influencers and add to influencer pool
    print("5. Perform onboarding interview to identify financial influencers...")
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
        input_file=ONBOARDING_RESULTS_FILE_TIKTOK,
        verified_profile_pool=FINFLUENCER_POOL_FILE_TIKTOK,
        prediction_threshold=PREDICTION_THRESHOLD_TIKTOK,
    )

    # Step 6: Perform profile search of identified financial influencers (profile metadata and posts) during search period
    print(
        "6. Perform profile search of identified financial influencers (profile metadata and recent posts) during search period..."
    )
    perform_tiktok_profile_metadata_search(
        project_name=PROJECT_NAME_TIKTOK,
        execution_date=PIPELINE_EXECUTION_DATE,
        input_file=FINFLUENCER_POOL_FILE_TIKTOK,
        output_file=FINFLUENCER_PROFILE_METADATA_SEARCH_FILE_TIKTOK,
    )
    perform_tiktok_profile_search(
        project_name=PROJECT_NAME_TIKTOK,
        execution_date=PIPELINE_EXECUTION_DATE,
        input_file=FINFLUENCER_POOL_FILE_TIKTOK,
        output_file=FINFLUENCER_PROFILE_SEARCH_FILE_TIKTOK,
        start_date=PROFILE_SEARCH_START_DATE,
        end_date=PROFILE_SEARCH_END_DATE,
        num_posts_per_profile=NUM_POSTS_PER_PROFILE,
    )
    perform_video_transcription(
        project_name=PROJECT_NAME_TIKTOK,
        execution_date=PIPELINE_EXECUTION_DATE,
        video_file=FINFLUENCER_PROFILE_SEARCH_FILE_TIKTOK,
        historical_post_file=FINFLUENCER_HISTORICAL_PROFILE_SEARCH_FILE_TIKTOK,
    )
    extract_stock_mentions(
        project_name=PROJECT_NAME_TIKTOK,
        execution_date=PIPELINE_EXECUTION_DATE,
        profile_metadata_file=FINFLUENCER_PROFILE_METADATA_SEARCH_FILE_TIKTOK,
        post_file=FINFLUENCER_PROFILE_SEARCH_FILE_TIKTOK,
        output_file=FINFLUENCER_STOCK_MENTIONS_FILE_TIKTOK,
        interview_type="tiktok_stock_mention",
    )

    # Steps 7 & 8: Run finfluencer interview and stock recommendations interview in parallel
    print(
        "7+8. Run finfluencer interview and stock recommendations interview in parallel..."
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        step7 = executor.submit(
            perform_tiktok_finfluencer_interview,
            project_name=PROJECT_NAME_TIKTOK,
            execution_date=PIPELINE_EXECUTION_DATE,
            profile_metadata_file=FINFLUENCER_STOCK_MENTIONS_FILE_TIKTOK,
            post_file=FINFLUENCER_HISTORICAL_PROFILE_SEARCH_FILE_TIKTOK,
            output_file=FINFLUENCER_POST_INTERVIEW_FILE_TIKTOK,
            filter_original_profiles=FILTER_ORIGINAL_PROFILES_TIKTOK,
            model_name=model_name,
            provider=provider,
            enable_web_search=enable_web_search,
            use_row_query=use_row_query,
        )
        step8 = executor.submit(
            perform_tiktok_stock_recommendation_interview,
            project_name=PROJECT_NAME_TIKTOK,
            execution_date=PIPELINE_EXECUTION_DATE,
            profile_metadata_file=FINFLUENCER_STOCK_MENTIONS_FILE_TIKTOK,
            post_file=FINFLUENCER_PROFILE_SEARCH_FILE_TIKTOK,
            finfluencer_pool=FINFLUENCER_POOL_FILE_TIKTOK,
            output_file=FINFLUENCER_STOCK_RECOMMENDATION_FILE_TIKTOK,
            filter_original_profiles=FILTER_ORIGINAL_PROFILES_TIKTOK,
            model_name=model_name,
            provider=provider,
            enable_web_search=enable_web_search,
            use_row_query=use_row_query,
        )
        # Surface exceptions from either future
        step7.result()
        step8.result()

    # Step 9: Conduct daily stock pick interview
    print("9. Conduct daily stock pick interview...")
    perform_tiktok_daily_stock_pick_interview(
        project_name=PROJECT_NAME_TIKTOK,
        execution_date=PIPELINE_EXECUTION_DATE,
        profile_metadata_file=FINFLUENCER_STOCK_MENTIONS_FILE_TIKTOK,
        post_file=FINFLUENCER_HISTORICAL_PROFILE_SEARCH_FILE_TIKTOK,
        output_file=FINFLUENCER_DAILY_STOCK_PICK_FILE_TIKTOK,
        filter_original_profiles=FILTER_ORIGINAL_PROFILES_TIKTOK,
        model_name=model_name,
        provider=provider,
        enable_web_search=enable_web_search,
        use_row_query=use_row_query,
    )

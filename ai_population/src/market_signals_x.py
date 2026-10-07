import argparse
import asyncio
import os
import sys
import traceback
import pandas as pd
import json
from tqdm import tqdm
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

tqdm.pandas()
from ai_population.config.market_signals_config import (
    PIPELINE_EXECUTION_DATE,
    MIN_FOLLOWER_COUNT,
    NUM_POSTS_PER_PROFILE,
    MIN_POSTS_COUNT,
    NUM_POSTS_PER_KEYWORD,
    PROFILE_SEARCH_START_DATE,
    PROFILE_SEARCH_END_DATE,
    PROJECT_NAME_X,
    SEARCH_TERMS_X,
    FINFLUENCER_POOL_FILE_X,
    KEYWORD_SEARCH_FILE_X,
    PROFILE_METADATA_SEARCH_FILE_X,
    ONBOARDING_RESULTS_FILE_X,
    EXPERT_REFLECTION_FILE_X,
    FINFLUENCER_PROFILE_METADATA_SEARCH_FILE_X,
    FINFLUENCER_PROFILE_SEARCH_FILE_X,
    FINFLUENCER_STOCK_MENTIONS_FILE_X,
    FINFLUENCER_POST_INTERVIEW_FILE_X,
    FINFLUENCER_STOCK_RECOMMENDATION_FILE_X,
    ONBOARDING_INTERVIEW_REGEX_PATTERNS,
    FINFLUENCER_INTERVIEW_REGEX_PATTERNS,
    FINFLUENCER_DAILY_STOCK_PICK_REGEX_PATTERNS,
    STOCK_RECOMMENDATION_OUTPUT_COLUMNS,
    PREDICTION_THRESHOLD_X,
    FILTER_ORIGINAL_PROFILES_X,
    ORIGINAL_PROFILES_X,
    DAILY_STOCK_PICK_PROFILES_X,
    FINFLUENCER_DAILY_STOCK_PICK_FILE_X,
    FINFLUENCER_HISTORICAL_PROFILE_SEARCH_FILE_X,
    LATEST_K_POSTS_PER_PROFILE,
    FINFLUENCER_PREDICTION_MARKET_FILE_X,
    PREDICTION_MARKET_INTERVIEW_REGEX_PATTERNS,
    POLYMARKET_EVENTS,
    GDP_MANUAL_OVERRIDE,
    BASELINE_MODULES_X,
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
    update_verified_profile_pool,
    coalesce_columns_by_regex,
    extract_stock_mentions,
    format_stock_recommendations,
    perform_x_keyword_search,
    perform_x_profile_metadata_search,
    perform_x_profile_search,
    fetch_daily_snapshot,
)
from ai_population.prompts.prompt_template import (
    x_finfluencer_onboarding_system_prompt,
    x_finfluencer_onboarding_user_prompt,
    x_investmentadvisor_reflection_system_prompt,
    investmentadvisor_reflection_user_prompt,
    x_finfluencer_interview_system_prompt,
    finfluencer_interview_user_prompt,
    prediction_market_interview_user_prompt_preamble,
    prediction_market_question_block_template,
    prediction_market_interview_user_prompt_suffix,
    stock_recommendation_interview_user_prompt,
    daily_stock_pick_user_prompts,
)
from ai_population.src.baseline_arms import (
    ARM_RESPONSE_FIELD,
    ARM_SYSTEM_PROMPT_COL,
    ARM_TIMESTAMP_COL,
    EMPTY_POST_FILE,
    MODULES as BASELINE_MODULES,
    STATUS_CODES,
    BaselineArmsError,
    arm_clock,
    arm_input_file,
    arm_interview_type,
    arm_output_file,
    baseline_execution_date,
    build_arm_plan,
    collect_arm_calls,
    consolidate_module_results,
    extract_production_search_traces,
    log_arm_failure,
    module_chunks,
    print_plan_summary,
    qc_and_log,
    remove_arm_batch_inputs,
    write_arm_batch_inputs,
    write_arm_call_records,
    write_arm_plan_files,
    yes_cell_schedule,
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
        model_name=GPT_MODEL,
        profile_metadata_file=profile_metadata_file,
        post_file=post_file,
        output_file=output_file,
        system_prompt_template=x_finfluencer_onboarding_system_prompt,
        user_prompt_template=x_finfluencer_onboarding_user_prompt,
        llm_response_field="onboarding_llm_response",
        interview_type="x_finfluencer_onboarding",
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
    Generates expert reflections for a given project and role by selecting appropriate prompt templates and invoking the profile interview process.

    Args:
        project_name (str): The name of the project for which reflections are being generated.
        execution_date (str): The date of execution in string format.
        role (str): The expert role, must be one of the following roles: "investment_advisor".
        profile_metadata_file (str): Path to the profile metadata file.
        post_file (str): Path to the post file associated with the expert.
        output_file (str): Path where the generated reflection output will be saved.

    Raises:
        ValueError: If the provided role is not supported.
    """
    if role == "investment_advisor":
        system_prompt_template = x_investmentadvisor_reflection_system_prompt
        user_prompt_template = investmentadvisor_reflection_user_prompt
        llm_response_field = (
            "x_finfluencer_expert_reflection_investmentadvisor_response"
        )
        interview_type = "x_finfluencer_expert_reflection_investmentadvisor"

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


def perform_x_finfluencer_interview(
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
    Conducts an interview process for X (Twitter) finfluencer profiles, processes the results, and saves the formatted output.

    This function performs the following steps:
    1. Runs the profile interview using the specified GPT model and prompt templates.
    2. Loads the interview results from a CSV file.
    3. Extracts and processes LLM responses from the interview results.
    4. Merges columns with identical information based on predefined regex patterns.
    5. Optionally filters the results to include only original profiles.
    6. Saves both the filtered and full interview results to CSV files.

    Args:
        project_name (str): Name of the project directory.
        execution_date (str): Date of execution, used for organizing output files.
        profile_metadata_file (str): Path to the file containing profile metadata.
        post_file (str): Path to the file containing post data.
        output_file (str): Name of the output CSV file for interview results.
        filter_original_profiles (bool, optional): If True, filters results to only include original profiles. Defaults to False.

    Returns:
        None
    """
    perform_profile_interview(
        project_name=project_name,
        execution_date=execution_date,
        model_name=model_name,
        profile_metadata_file=profile_metadata_file,
        post_file=post_file,
        output_file=output_file,
        system_prompt_template=x_finfluencer_interview_system_prompt,
        user_prompt_template=finfluencer_interview_user_prompt,
        llm_response_field="x_finfluencer_interview",
        interview_type="x_finfluencer_interview",
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
    post_interview_results["model"] = model_name

    # Include timestamp information for when the interview was conducted
    if "finfluencer_interview_datetime" not in post_interview_results.columns:
        post_interview_results["finfluencer_interview_datetime"] = (
            pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        )

    # # Format past conversation
    # post_interview_results["history"] = post_interview_results.apply(
    #     lambda row: json.dumps(
    #         [
    #             {
    #                 "role": "user",
    #                 "content": finfluencer_interview_user_prompt,
    #             },
    #             {
    #                 "role": "assistant",
    #                 "content": row[
    #                     "x_finfluencer_interview"
    #                 ],
    #             },
    #         ],
    #         ensure_ascii=False,
    #         separators=(",", ":"),
    #     ),
    #     axis=1,
    # )

    # Save formatted interview results
    if filter_original_profiles:
        filtered_post_interview_results = post_interview_results[
            post_interview_results["account_id"].isin(ORIGINAL_PROFILES_X)
        ].reset_index(drop=True)
        filtered_post_interview_results.to_csv(
            os.path.join(
                base_dir, "../data", project_name, execution_date, output_file
            ),
            index=False,
        )

    output_file_dir = os.path.join(
        base_dir,
        "../data",
        project_name,
        execution_date,
        output_file[:-4] + "_full.csv",
    )
    post_interview_results.to_csv(
        output_file_dir,
        index=False,
    )

    # # Conduct prediction market interview
    # perform_x_prediction_market_interview(
    #     project_name=project_name,
    #     execution_date=execution_date,
    #     profile_metadata_file=output_file_dir,
    #     post_file=post_file,
    #     output_file=FINFLUENCER_PREDICTION_MARKET_FILE_X,
    #     filter_original_profiles=filter_original_profiles,
    # )


def perform_x_prediction_market_interview(
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
    snapshot = fetch_daily_snapshot(
        events=POLYMARKET_EVENTS,
        gdp_override=GDP_MANUAL_OVERRIDE,
    )
    macro = {
        row["variable"]: row["value"]
        for _, row in snapshot.iterrows()
        if row["contract_id"] is None or pd.isna(row["contract_id"])
    }

    blocks = []
    for event in POLYMARKET_EVENTS:
        mid = str(event["market_id"])
        contract_rows = snapshot[snapshot["contract_id"].astype(str) == mid]
        poly_p_row = contract_rows[contract_rows["variable"] == "polymarket_p"]["value"]
        vol_row = contract_rows[contract_rows["variable"] == "contract_volume"]["value"]
        blocks.append(
            prediction_market_question_block_template.format(
                question_text=event["question_text"],
                resolution_text=event["resolution_text"].format(**macro),
                contract_id=mid,
                fed_rate=macro["fed_rate"],
                cpi_yoy=macro["cpi_yoy"],
                unemp_rate=macro["unemp_rate"],
                gdp_growth=macro["gdp_growth"],
                polymarket_p=poly_p_row.iloc[0] if len(poly_p_row) else "NA",
                contract_volume=vol_row.iloc[0] if len(vol_row) else "NA",
            )
        )

    rendered_prompt = (
        prediction_market_interview_user_prompt_preamble
        + "\n\n"
        + "\n\n".join(blocks)
        + "\n\n"
        + prediction_market_interview_user_prompt_suffix
    )

    perform_profile_interview(
        project_name=project_name,
        execution_date=execution_date,
        model_name=model_name,
        profile_metadata_file=profile_metadata_file,
        post_file=post_file,
        output_file=output_file,
        system_prompt_template=x_finfluencer_interview_system_prompt,
        user_prompt_template=rendered_prompt,
        llm_response_field="x_prediction_market_interview",
        interview_type="x_prediction_market_interview",
        enable_web_search=enable_web_search,
        use_row_query=use_row_query,
        response_timestamp_col="prediction_market_interview_datetime",
        latest_k_posts=LATEST_K_POSTS_PER_PROFILE,
        history_field="history",
        batch_timeout_seconds=7200,
        provider=provider,
    )

    # Preprocess post interview results
    post_interview_results = pd.read_csv(
        os.path.join(base_dir, "../data", project_name, execution_date, output_file)
    )
    extracted_responses = post_interview_results["x_prediction_market_interview"].apply(
        extract_llm_responses
    )
    post_interview_results = pd.concat(
        [post_interview_results, extracted_responses], axis=1
    )
    # Merge identical columns from interview response
    post_interview_results = coalesce_columns_by_regex(
        post_interview_results, PREDICTION_MARKET_INTERVIEW_REGEX_PATTERNS
    )

    # Include LLM model information
    post_interview_results["model"] = model_name

    # Include timestamp information for when the interview was conducted
    if "prediction_market_interview_datetime" not in post_interview_results.columns:
        post_interview_results["prediction_market_interview_datetime"] = (
            pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        )

    # Save formatted interview results
    if filter_original_profiles:
        filtered_post_interview_results = post_interview_results[
            post_interview_results["account_id"].isin(ORIGINAL_PROFILES_X)
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


def perform_x_stock_recommendation_interview(
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
    Performs an interview process to extract and verify stock recommendations from X (formerly Twitter) finfluencers.

    This function processes profile metadata and finfluencer pool data to prepare a dataset of stock mentions,
    formats and enriches the data, and then uses an LLM-based interview process to extract stock recommendations.
    It further verifies and filters the recommendations, saving both the full and filtered results.

    Args:
        project_name (str): Name of the project directory.
        execution_date (str): Date of execution, used for organizing data files.
        profile_metadata_file (str): Filename for the profile metadata CSV.
        post_file (str): Filename for the posts CSV.
        finfluencer_pool (str): Filename for the finfluencer pool CSV.
        output_file (str): Filename for saving the output CSV.
        filter_original_profiles (bool, optional): Whether to filter results to only original profiles. Defaults to False.

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

    # If no stock mentions were found across all profiles, write empty output
    # files with the expected headers and skip the interview step entirely.
    if combined_stock_mentions.empty:
        print(
            "No stock mentions found across all profiles; writing empty stock "
            "recommendation output and skipping the interview."
        )
        empty_output = pd.DataFrame(columns=STOCK_RECOMMENDATION_OUTPUT_COLUMNS)
        empty_output.to_csv(
            os.path.join(
                base_dir, "../data", project_name, execution_date, output_file
            ),
            index=False,
        )
        empty_output.to_csv(
            os.path.join(
                base_dir,
                "../data",
                project_name,
                execution_date,
                output_file[:-4] + "_full.csv",
            ),
            index=False,
        )
        return

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
        system_prompt_template=x_finfluencer_interview_system_prompt,
        user_prompt_template=stock_recommendation_interview_user_prompt,
        llm_response_field="x_finfluencer_stock_recommendation",
        interview_type="x_finfluencer_stock_recommendation",
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
            valid_stock_recommendations["account_id"].isin(ORIGINAL_PROFILES_X)
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


def perform_x_daily_stock_pick_interview(
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

    profile_metadata = pd.read_csv(
        os.path.join(
            base_dir, "../data", project_name, execution_date, profile_metadata_file
        )
    )
    sampled_profile_metadata = profile_metadata[
        profile_metadata["account_id"].isin(
            DAILY_STOCK_PICK_PROFILES_X + ORIGINAL_PROFILES_X
        )
    ].reset_index(drop=True)
    sampled_profile_metadata.to_csv(
        os.path.join(
            base_dir,
            "../data",
            project_name,
            execution_date,
            f"x_finfluencer_sampled_profiles_{execution_date}.csv",
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
            profile_metadata_file=f"x_finfluencer_sampled_profiles_{execution_date}.csv",
            post_file=post_file,
            output_file=chunk_output_file,
            system_prompt_template=x_finfluencer_interview_system_prompt,
            user_prompt_template=user_prompt,
            llm_response_field="x_finfluencer_daily_stock_pick",
            interview_type=f"x_finfluencer_daily_stock_pick_{idx+1}",
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
            "x_finfluencer_daily_stock_pick"
        ].apply(extract_llm_responses)
        extracted_responses[f"x_finfluencer_daily_stock_pick_{idx+1}"] = (
            daily_stock_pick_chunk["x_finfluencer_daily_stock_pick"]
        )
        extracted_responses_list.append(extracted_responses)

    daily_stock_pick_results = pd.concat(
        [daily_stock_pick_chunk.drop(columns=["x_finfluencer_daily_stock_pick"])]
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
            daily_stock_pick_results["account_id"].isin(ORIGINAL_PROFILES_X)
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


def perform_x_baseline_arms(
    project_name: str,
    execution_date: str,
    model_name: str = GPT_MODEL,
    provider: str = None,
    enable_web_search: bool = True,
    use_row_query: bool = False,
    cells: list = None,
    dry_run: bool = False,
    ignore_arm_clock: bool = False,
) -> int:
    """
    Runs the baseline arms (bare / generic / twin setups with web search on and off) on
    the production drop of execution_date and saves the outputs to the
    `{execution_date}-baseline` folder beside the drop.

    The search-on cells run first and follow the arm clock: they are skipped if the arms
    start after 09:00 ET on the day after the drop, fall back to row mode at 08:00 ET and
    start no call after 09:15 ET. The search-off cells run afterwards, 3 batches at a time.
    The batch input files are deleted unless the run fails.

    Args:
        project_name (str): Name of the project directory.
        execution_date (str): Drop date (DD-MM-YYYY) whose production prompts are re-run.
        model_name (str, optional): Production model id. Defaults to GPT_MODEL.
        provider (str, optional): Provider override. Defaults to None.
        enable_web_search (bool, optional): Production's web search setting; must be True. Defaults to True.
        use_row_query (bool, optional): Query row by row instead of using the batch API. Defaults to False.
        cells (list, optional): Subset of BASELINE_ARMS_CELLS_X keys to run. Defaults to None (all enabled cells).
        dry_run (bool, optional): Only write the manifest, input CSVs and batch input files. Defaults to False.
        ignore_arm_clock (bool, optional): Testing only. Run the search-on cells regardless of the arm clock;
            their timing is logged as "override". Defaults to False.

    Returns:
        int: Exit status, 0 OK / 1 WARN / 2 FAIL.

    Raises:
        BaselineArmsError: If web search is off or the arm plan fails its checks.
    """
    run_start = datetime.now(timezone.utc)
    if not enable_web_search:
        raise BaselineArmsError(
            "Production web search is off; the search-on arms could not mirror production."
        )

    plan = build_arm_plan(project_name, execution_date, model_name, provider, cells)
    out_dir = write_arm_plan_files(plan, project_name)
    print_plan_summary(plan)
    if dry_run:
        written = write_arm_batch_inputs(plan, project_name)
        print(
            f"Dry run: wrote the prompt manifest, input CSVs and {len(written)} batch "
            f"input files to {out_dir}. No API calls were made."
        )
        return 0

    baseline_date = baseline_execution_date(execution_date)
    clock = arm_clock(execution_date)
    task_errors = []

    def run_arm_task(task):
        """
        Runs one arm interview (a module, condition and chunk) unless its output exists.

        Args:
            task (tuple): (module, condition, chunk, schedule), where schedule holds
                use_row_query, batch_timeout_seconds and row_deadline_utc.
        """
        module, condition, chunk, schedule = task
        output_file = arm_output_file(module, condition, chunk, execution_date)
        output_path = os.path.join(
            base_dir, "../data", project_name, baseline_date, output_file
        )
        if os.path.exists(output_path):
            print(f"Skipping {output_file}: already exists.")
            return
        battery = plan["batteries"][module][chunk - 1]
        perform_profile_interview(
            project_name=project_name,
            execution_date=baseline_date,
            model_name=model_name,
            profile_metadata_file=arm_input_file(module, condition, execution_date),
            post_file=EMPTY_POST_FILE,
            output_file=output_file,
            system_prompt_template="",
            user_prompt_template=battery["template"],
            llm_response_field=ARM_RESPONSE_FIELD,
            interview_type=arm_interview_type(module, condition, chunk),
            enable_web_search=condition == "yes",
            use_row_query=schedule["use_row_query"],
            response_timestamp_col=ARM_TIMESTAMP_COL,
            batch_timeout_seconds=schedule["batch_timeout_seconds"],
            provider=provider,
            system_prompt_column=ARM_SYSTEM_PROMPT_COL,
            responses_api_without_tools=condition == "no",
            capture_raw_response=True,
            row_deadline_utc=schedule["row_deadline_utc"],
        )

    def run_arm_tasks(tasks, max_workers):
        """
        Runs arm tasks in parallel and records failed tasks in task_errors.

        Args:
            tasks (list): Tasks for run_arm_task.
            max_workers (int): Number of tasks run at once.
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_arm_task, task): task for task in tasks}
            for future in as_completed(futures):
                module, condition, chunk, _ = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Arm task {module}/{condition}/chunk {chunk} failed: {e}")
                    task_errors.append(
                        {
                            "module": module,
                            "condition": condition,
                            "chunk": chunk,
                            "error": str(e),
                        }
                    )

    def production_schedule(module):
        """
        Gets the schedule production uses for a module, without a deadline.

        Args:
            module (str): "post_interview" or "daily_stock_pick".

        Returns:
            dict: use_row_query, batch_timeout_seconds and row_deadline_utc.
        """
        return {
            "use_row_query": use_row_query,
            "batch_timeout_seconds": BASELINE_MODULES_X[module][
                "batch_timeout_seconds"
            ],
            "row_deadline_utc": None,
        }

    def arm_tasks(condition, schedule_for):
        """
        Lists the tasks of one condition, one per module and chunk that has arm rows.

        Args:
            condition (str): "yes" or "no".
            schedule_for (Callable[[str], dict]): Returns the schedule of a module.

        Returns:
            list: Tasks for run_arm_task.
        """
        planned = set(zip(plan["rows"]["module"], plan["rows"]["condition"]))
        return [
            (module, condition, chunk, schedule_for(module))
            for module in BASELINE_MODULES
            if (module, condition) in planned
            for chunk in module_chunks(module)
        ]

    yes_skipped = False
    if ignore_arm_clock:
        yes_tasks = arm_tasks("yes", production_schedule)
    else:
        schedule = yes_cell_schedule(run_start, clock, use_row_query)
        yes_skipped = schedule["skip"]
        yes_tasks = [] if yes_skipped else arm_tasks("yes", lambda module: schedule)
        if yes_skipped:
            print(
                f"Arms started after 09:00 ET on the day after {execution_date}: "
                f"skipping the search-on cells."
            )
    if yes_tasks:
        # Submit all search-on batches at once; limit concurrency in row mode
        run_arm_tasks(
            yes_tasks, 3 if yes_tasks[0][3]["use_row_query"] else len(yes_tasks)
        )

    run_arm_tasks(arm_tasks("no", production_schedule), 3)

    calls = collect_arm_calls(plan, project_name, clock)
    results = {}
    if not calls.empty:
        write_arm_call_records(calls, project_name, execution_date)
        for module in BASELINE_MODULES:
            results[module] = consolidate_module_results(
                plan, calls, module, project_name
            )
    production_traces, missing_traces = extract_production_search_traces(
        plan, project_name
    )
    status = qc_and_log(
        plan=plan,
        calls=calls,
        results=results,
        production_traces=production_traces,
        missing_production_traces=missing_traces,
        project_name=project_name,
        run_start=run_start,
        yes_skipped=yes_skipped,
        ignore_arm_clock=ignore_arm_clock,
        task_errors=task_errors,
    )

    if status != STATUS_CODES["FAIL"]:
        remove_arm_batch_inputs(project_name, execution_date)
    return status


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
    profile_metadata["followers"] = (
        pd.to_numeric(profile_metadata["followers"], errors="coerce")
        .fillna(0.0)
        .astype(float)
    )
    profile_metadata["statusesCount"] = (
        pd.to_numeric(profile_metadata["statusesCount"], errors="coerce")
        .fillna(0.0)
        .astype(float)
    )
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
    parser = argparse.ArgumentParser(
        description="Run the X (Twitter) market signals interview pipeline.",
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
    parser.add_argument(
        "--baseline-arms",
        dest="baseline_arms",
        action="store_true",
        default=False,
        help=(
            "After step 9, run the baseline arms on the day's drop. "
            "Exit code is the arms status: 0 OK / 1 WARN / 2 FAIL."
        ),
    )
    parser.add_argument(
        "--arms-only",
        dest="arms_only",
        action="store_true",
        default=False,
        help="Skip steps 6-9 and run the baseline arms on an existing drop (requires --date).",
    )
    parser.add_argument(
        "--date",
        dest="arms_date",
        type=str,
        default=None,
        help="Drop date (DD-MM-YYYY) for --arms-only.",
    )
    parser.add_argument(
        "--arms-dry-run",
        dest="arms_dry_run",
        action="store_true",
        default=False,
        help=(
            "With --arms-only: write the prompt manifest, input CSVs and batch "
            "input files without making any API call."
        ),
    )
    parser.add_argument(
        "--arm-cells",
        dest="arm_cells",
        type=str,
        default=None,
        help=(
            "Comma-separated subset of arm cells to run (bare_yes, bare_no, "
            "generic_yes, generic_no, twin_no, twin_replicate). Defaults to all "
            "cells enabled in BASELINE_ARMS_CELLS_X."
        ),
    )
    parser.add_argument(
        "--ignore-arm-clock",
        dest="ignore_arm_clock",
        action="store_true",
        default=False,
        help=(
            "Testing only: run the search-on arm cells regardless of the 09:00 ET "
            "arm clock. Their timing is logged as 'override', never as clean."
        ),
    )
    args = parser.parse_args()
    model_name = args.model_name
    provider = args.provider
    enable_web_search = args.enable_web_search
    use_row_query = args.use_row_query
    run_arms = args.baseline_arms or args.arms_only
    if args.arms_only and not args.arms_date:
        parser.error("--arms-only requires --date DD-MM-YYYY.")
    if args.arms_date and not args.arms_only:
        parser.error(
            "--date is only used with --arms-only; production runs on PIPELINE_EXECUTION_DATE."
        )
    if args.arms_dry_run and not args.arms_only:
        parser.error("--arms-dry-run requires --arms-only.")
    if (args.arm_cells or args.ignore_arm_clock) and not run_arms:
        parser.error(
            "--arm-cells and --ignore-arm-clock require --baseline-arms or --arms-only."
        )
    if args.arms_date:
        try:
            datetime.strptime(args.arms_date, "%d-%m-%Y")
        except ValueError:
            parser.error(f"--date {args.arms_date!r} is not in DD-MM-YYYY format.")

    # # Step 1: Perform search using predefined list of search terms
    # print("1. Perform keyword search using predefined list of search terms...")
    # perform_x_keyword_search(
    #     project_name=PROJECT_NAME_X,
    #     execution_date=PIPELINE_EXECUTION_DATE,
    #     search_terms=SEARCH_TERMS_X,
    #     output_file=KEYWORD_SEARCH_FILE_X,
    #     num_posts_per_keyword=NUM_POSTS_PER_KEYWORD,
    # )

    # # Step 2: Extract profile metadata for search results
    # print("2. Perform profile metadata search for keyword search results...")
    # perform_x_profile_metadata_search(
    #     project_name=PROJECT_NAME_X,
    #     execution_date=PIPELINE_EXECUTION_DATE,
    #     input_file=os.path.join(PIPELINE_EXECUTION_DATE, KEYWORD_SEARCH_FILE_X),
    #     output_file=PROFILE_METADATA_SEARCH_FILE_X,
    # )

    # # Step 3: Filter profiles that do not meet filtering criteria
    # print(
    #     "3. Filter X profiles based on follower count, post count, and verified finfluencer list..."
    # )
    # filter_x_profiles(
    #     project_name=PROJECT_NAME_X,
    #     execution_date=PIPELINE_EXECUTION_DATE,
    #     profile_metadata_file=PROFILE_METADATA_SEARCH_FILE_X,
    #     post_file=KEYWORD_SEARCH_FILE_X,
    #     verified_profile_pool=FINFLUENCER_POOL_FILE_X,
    # )

    # # Step 4: Generate expert reflections
    # print("4. Generate expert reflections of potential influencers...")
    # generate_expert_reflections(
    #     project_name=PROJECT_NAME_X,
    #     execution_date=PIPELINE_EXECUTION_DATE,
    #     role="investment_advisor",
    #     profile_metadata_file=PROFILE_METADATA_SEARCH_FILE_X,
    #     post_file=KEYWORD_SEARCH_FILE_X,
    #     output_file=EXPERT_REFLECTION_FILE_X,
    # )

    # # Step 5: Conduct onboarding interview to identify financial influencers and add to influencer pool
    # print("5. Perform onboarding interview to identify financial influencers...")
    # perform_x_onboarding_interview(
    #     project_name=PROJECT_NAME_X,
    #     execution_date=PIPELINE_EXECUTION_DATE,
    #     profile_metadata_file=EXPERT_REFLECTION_FILE_X,
    #     post_file=KEYWORD_SEARCH_FILE_X,
    #     output_file=ONBOARDING_RESULTS_FILE_X,
    # )
    # extract_stock_mentions(
    #     project_name=PROJECT_NAME_X,
    #     execution_date=PIPELINE_EXECUTION_DATE,
    #     profile_metadata_file=ONBOARDING_RESULTS_FILE_X,
    #     post_file=KEYWORD_SEARCH_FILE_X,
    #     output_file=ONBOARDING_RESULTS_FILE_X,
    #     interview_type="x_stock_mention",
    # )
    # update_verified_profile_pool(
    #     project_name=PROJECT_NAME_X,
    #     execution_date=PIPELINE_EXECUTION_DATE,
    #     input_file=ONBOARDING_RESULTS_FILE_X,
    #     verified_profile_pool=FINFLUENCER_POOL_FILE_X,
    #     prediction_threshold=PREDICTION_THRESHOLD_X,
    # )

    if not args.arms_only:
        # Step 6: Perform profile search of identified financial influencers (profile metadata and posts)
        print(
            "6. Perform profile search of identified financial influencers (profile metadata and recent posts) during the search period..."
        )
        perform_x_profile_metadata_search(
            project_name=PROJECT_NAME_X,
            execution_date=PIPELINE_EXECUTION_DATE,
            input_file=FINFLUENCER_POOL_FILE_X,
            output_file=FINFLUENCER_PROFILE_METADATA_SEARCH_FILE_X,
            cache_name="x_finfluencer_profile_metadata",
        )
        perform_x_profile_search(
            project_name=PROJECT_NAME_X,
            execution_date=PIPELINE_EXECUTION_DATE,
            input_file=FINFLUENCER_POOL_FILE_X,
            output_file=FINFLUENCER_PROFILE_SEARCH_FILE_X,
            start_date=PROFILE_SEARCH_START_DATE,
            end_date=PROFILE_SEARCH_END_DATE,
            num_posts_per_profile=NUM_POSTS_PER_PROFILE,
            historical_post_file=FINFLUENCER_HISTORICAL_PROFILE_SEARCH_FILE_X,
        )

        extract_stock_mentions(
            project_name=PROJECT_NAME_X,
            execution_date=PIPELINE_EXECUTION_DATE,
            profile_metadata_file=FINFLUENCER_PROFILE_METADATA_SEARCH_FILE_X,
            post_file=FINFLUENCER_PROFILE_SEARCH_FILE_X,
            output_file=FINFLUENCER_STOCK_MENTIONS_FILE_X,
            interview_type="x_stock_mention",
        )

        # Steps 7 & 8: Run finfluencer interview and stock recommendations interview in parallel
        print(
            "7+8. Run finfluencer interview and stock recommendations interview in parallel..."
        )
        with ThreadPoolExecutor(max_workers=2) as executor:
            step7 = executor.submit(
                perform_x_finfluencer_interview,
                project_name=PROJECT_NAME_X,
                execution_date=PIPELINE_EXECUTION_DATE,
                profile_metadata_file=FINFLUENCER_STOCK_MENTIONS_FILE_X,
                post_file=FINFLUENCER_HISTORICAL_PROFILE_SEARCH_FILE_X,
                output_file=FINFLUENCER_POST_INTERVIEW_FILE_X,
                filter_original_profiles=FILTER_ORIGINAL_PROFILES_X,
                model_name=model_name,
                provider=provider,
                enable_web_search=enable_web_search,
                use_row_query=use_row_query,
            )
            step8 = executor.submit(
                perform_x_stock_recommendation_interview,
                project_name=PROJECT_NAME_X,
                execution_date=PIPELINE_EXECUTION_DATE,
                profile_metadata_file=FINFLUENCER_STOCK_MENTIONS_FILE_X,
                post_file=FINFLUENCER_PROFILE_SEARCH_FILE_X,
                finfluencer_pool=FINFLUENCER_POOL_FILE_X,
                output_file=FINFLUENCER_STOCK_RECOMMENDATION_FILE_X,
                filter_original_profiles=FILTER_ORIGINAL_PROFILES_X,
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
        perform_x_daily_stock_pick_interview(
            project_name=PROJECT_NAME_X,
            execution_date=PIPELINE_EXECUTION_DATE,
            profile_metadata_file=FINFLUENCER_STOCK_MENTIONS_FILE_X,
            post_file=FINFLUENCER_HISTORICAL_PROFILE_SEARCH_FILE_X,
            output_file=FINFLUENCER_DAILY_STOCK_PICK_FILE_X,
            filter_original_profiles=FILTER_ORIGINAL_PROFILES_X,
            model_name=model_name,
            provider=provider,
            enable_web_search=enable_web_search,
            use_row_query=use_row_query,
        )

    # Step 10: Run baseline arms on the day's drop
    if run_arms:
        arms_date = args.arms_date or PIPELINE_EXECUTION_DATE
        print(f"10. Run baseline arms on the {arms_date} drop...")
        try:
            arms_status = perform_x_baseline_arms(
                project_name=PROJECT_NAME_X,
                execution_date=arms_date,
                model_name=model_name,
                provider=provider,
                enable_web_search=enable_web_search,
                use_row_query=use_row_query,
                cells=args.arm_cells.split(",") if args.arm_cells else None,
                dry_run=args.arms_dry_run,
                ignore_arm_clock=args.ignore_arm_clock,
            )
        except Exception as e:
            if not isinstance(e, BaselineArmsError):
                traceback.print_exc()
            print(f"BASELINE {arms_date}: FAIL {e}")
            if not args.arms_dry_run:
                log_arm_failure(PROJECT_NAME_X, arms_date, str(e))
            arms_status = 2
        sys.exit(arms_status)

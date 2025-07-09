import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime

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
    STOCK_RECOMMENDATION_OUTPUT_COLUMNS,
    PREDICTION_THRESHOLD_X,
)

PROFILE_SEARCH_START_DATE = datetime.strptime(
    PROFILE_SEARCH_START_DATE, "%m-%d-%Y"
).strftime("%Y-%m-%d")
PROFILE_SEARCH_END_DATE = datetime.strptime(
    PROFILE_SEARCH_END_DATE, "%m-%d-%Y"
).strftime("%Y-%m-%d")

from ai_population.config.base_config import (
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
    perform_x_keyword_search,
    perform_x_profile_metadata_search,
    perform_x_profile_search,
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
        output_file=KEYWORD_SEARCH_FILE_X,
        num_posts_per_keyword=NUM_POSTS_PER_KEYWORD,
    )

    # Step 2: Extract profile metadata for search results
    print("Perform profile metadata search for keyword search results...")
    perform_x_profile_metadata_search(
        project_name=PROJECT_NAME_X,
        execution_date=PIPELINE_EXECUTION_DATE,
        input_file=os.path.join(PIPELINE_EXECUTION_DATE, KEYWORD_SEARCH_FILE_X),
        output_file=PROFILE_METADATA_SEARCH_FILE_X,
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

    # Step 4: Generate expert reflections
    print("Generate expert reflections of potential influencers...")
    generate_expert_reflections(
        project_name=PROJECT_NAME_X,
        execution_date=PIPELINE_EXECUTION_DATE,
        role="portfolio_manager",
        profile_metadata_file=PROFILE_METADATA_SEARCH_FILE_X,
        post_file=KEYWORD_SEARCH_FILE_X,
        output_file=EXPERT_REFLECTION_FILE_X,
    )
    generate_expert_reflections(
        project_name=PROJECT_NAME_X,
        execution_date=PIPELINE_EXECUTION_DATE,
        role="investment_advisor",
        profile_metadata_file=EXPERT_REFLECTION_FILE_X,
        post_file=KEYWORD_SEARCH_FILE_X,
        output_file=EXPERT_REFLECTION_FILE_X,
    )
    generate_expert_reflections(
        project_name=PROJECT_NAME_X,
        execution_date=PIPELINE_EXECUTION_DATE,
        role="financial_analyst",
        profile_metadata_file=EXPERT_REFLECTION_FILE_X,
        post_file=KEYWORD_SEARCH_FILE_X,
        output_file=EXPERT_REFLECTION_FILE_X,
    )
    generate_expert_reflections(
        project_name=PROJECT_NAME_X,
        execution_date=PIPELINE_EXECUTION_DATE,
        role="economist",
        profile_metadata_file=EXPERT_REFLECTION_FILE_X,
        post_file=KEYWORD_SEARCH_FILE_X,
        output_file=EXPERT_REFLECTION_FILE_X,
    )

    # Step 5: Conduct onboarding interview to identify financial influencers and add to influencer pool
    print("Perform onboarding interview to identify financial influencers...")
    perform_x_onboarding_interview(
        project_name=PROJECT_NAME_X,
        execution_date=PIPELINE_EXECUTION_DATE,
        profile_metadata_file=EXPERT_REFLECTION_FILE_X,
        post_file=KEYWORD_SEARCH_FILE_X,
        output_file=ONBOARDING_RESULTS_FILE_X,
    )
    extract_stock_mentions(
        project_name=PROJECT_NAME_X,
        execution_date=PIPELINE_EXECUTION_DATE,
        profile_metadata_file=ONBOARDING_RESULTS_FILE_X,
        post_file=KEYWORD_SEARCH_FILE_X,
        output_file=ONBOARDING_RESULTS_FILE_X,
        interview_type="x_stock_mention",
    )
    update_verified_profile_pool(
        project_name=PROJECT_NAME_X,
        execution_date=PIPELINE_EXECUTION_DATE,
        input_file=ONBOARDING_RESULTS_FILE_X,
        verified_profile_pool=FINFLUENCER_POOL_FILE_X,
        prediction_threshold=PREDICTION_THRESHOLD_X,
    )

    # Step 6: Perform profile search of identified financial influencers (profile metadata and posts)
    print(
        "Perform profile search of identified financial influencers (profile metadata and recent posts) during the search period..."
    )
    perform_x_profile_metadata_search(
        project_name=PROJECT_NAME_X,
        execution_date=PIPELINE_EXECUTION_DATE,
        input_file=FINFLUENCER_POOL_FILE_X,
        output_file=FINFLUENCER_PROFILE_METADATA_SEARCH_FILE_X,
    )
    perform_x_profile_search(
        project_name=PROJECT_NAME_X,
        execution_date=PIPELINE_EXECUTION_DATE,
        input_file=FINFLUENCER_POOL_FILE_X,
        output_file=FINFLUENCER_PROFILE_SEARCH_FILE_X,
        start_date=PROFILE_SEARCH_START_DATE,
        end_date=PROFILE_SEARCH_END_DATE,
        num_posts_per_profile=NUM_POSTS_PER_PROFILE,
    )
    extract_stock_mentions(
        project_name=PROJECT_NAME_X,
        execution_date=PIPELINE_EXECUTION_DATE,
        profile_metadata_file=FINFLUENCER_PROFILE_METADATA_SEARCH_FILE_X,
        post_file=FINFLUENCER_PROFILE_SEARCH_FILE_X,
        output_file=FINFLUENCER_STOCK_MENTIONS_FILE_X,
        interview_type="x_stock_mention",
    )

    # Step 7: Conduct interview on financial markets
    print("Conduct digital interview on financial markets...")
    perform_x_finfluencer_interview(
        project_name=PROJECT_NAME_X,
        execution_date=PIPELINE_EXECUTION_DATE,
        profile_metadata_file=FINFLUENCER_STOCK_MENTIONS_FILE_X,
        post_file=FINFLUENCER_PROFILE_SEARCH_FILE_X,
        output_file=FINFLUENCER_POST_INTERVIEW_FILE_X,
    )

    # Step 8: Conduct interview on stock recommendations
    print("Conduct digital interview on stock recommendations...")
    perform_x_stock_recommendation_interview(
        project_name=PROJECT_NAME_X,
        execution_date=PIPELINE_EXECUTION_DATE,
        profile_metadata_file=FINFLUENCER_STOCK_MENTIONS_FILE_X,
        post_file=FINFLUENCER_PROFILE_SEARCH_FILE_X,
        finfluencer_pool=FINFLUENCER_POOL_FILE_X,
        output_file=FINFLUENCER_STOCK_RECOMMENDATION_FILE_X,
    )

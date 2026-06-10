import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime

tqdm.pandas()
from ai_population.config.market_signals_config import (
    NUM_POSTS_PER_PROFILE,
    PROJECT_NAME_X,
    ONBOARDING_INTERVIEW_REGEX_PATTERNS,
)

# Unique configurations for market signals project - X New Users
PIPELINE_EXECUTION_DATE = "03-11-2025_newusers"
FINFLUENCER_POOL_FILE_X = "x_verified_finfluencer_profiles.csv"
FINFLUENCER_PROFILE_METADATA_SEARCH_FILE_X = (
    f"x_finfluencer_profile_metadata_{PIPELINE_EXECUTION_DATE}.csv"
)
FINFLUENCER_PROFILE_SEARCH_FILE_X = (
    f"x_finfluencer_profile_search_{PIPELINE_EXECUTION_DATE}.csv"
)
EXPERT_REFLECTION_FILE_X = f"x_expert_reflection_{PIPELINE_EXECUTION_DATE}.csv"
ONBOARDING_RESULTS_FILE_X = f"x_onboarding_results_{PIPELINE_EXECUTION_DATE}.csv"
PREDICTION_THRESHOLD_X = 0
PROFILE_SEARCH_START_DATE = "01-01-2025"
PROFILE_SEARCH_END_DATE = "11-03-2025"
PROFILE_SEARCH_START_DATE = datetime.strptime(
    PROFILE_SEARCH_START_DATE, "%m-%d-%Y"
).strftime("%Y-%m-%d")
PROFILE_SEARCH_END_DATE = datetime.strptime(
    PROFILE_SEARCH_END_DATE, "%m-%d-%Y"
).strftime("%Y-%m-%d")
NEW_USER_INPUT_FILE = "x_new_users_3Nov2025.csv"

from ai_population.config.base_config import GPT_MODEL
from ai_population.src.utils import (
    extract_llm_responses,
    perform_profile_interview,
    update_verified_profile_pool,
    coalesce_columns_by_regex,
    perform_x_profile_metadata_search,
    perform_x_profile_search,
)
from ai_population.prompts.prompt_template import (
    x_finfluencer_onboarding_system_prompt,
    x_finfluencer_onboarding_user_prompt,
    x_investmentadvisor_reflection_system_prompt,
    investmentadvisor_reflection_user_prompt,
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
    )


if __name__ == "__main__":
    print("1. Extract profile metadata and recent posts from new users...")
    perform_x_profile_metadata_search(
        project_name=PROJECT_NAME_X,
        execution_date=PIPELINE_EXECUTION_DATE,
        input_file=NEW_USER_INPUT_FILE,
        output_file=FINFLUENCER_PROFILE_METADATA_SEARCH_FILE_X,
    )
    perform_x_profile_search(
        project_name=PROJECT_NAME_X,
        execution_date=PIPELINE_EXECUTION_DATE,
        input_file=NEW_USER_INPUT_FILE,
        output_file=FINFLUENCER_PROFILE_SEARCH_FILE_X,
        start_date=PROFILE_SEARCH_START_DATE,
        end_date=PROFILE_SEARCH_END_DATE,
        num_posts_per_profile=NUM_POSTS_PER_PROFILE,
    )

    print("2. Generate expert reflections of new financial influencers...")
    generate_expert_reflections(
        project_name=PROJECT_NAME_X,
        execution_date=PIPELINE_EXECUTION_DATE,
        role="investment_advisor",
        profile_metadata_file=FINFLUENCER_PROFILE_METADATA_SEARCH_FILE_X,
        post_file=FINFLUENCER_PROFILE_SEARCH_FILE_X,
        output_file=EXPERT_REFLECTION_FILE_X,
    )

    print("3. Perform onboarding interview for new financial influencers...")
    perform_x_onboarding_interview(
        project_name=PROJECT_NAME_X,
        execution_date=PIPELINE_EXECUTION_DATE,
        profile_metadata_file=EXPERT_REFLECTION_FILE_X,
        post_file=FINFLUENCER_PROFILE_SEARCH_FILE_X,
        output_file=ONBOARDING_RESULTS_FILE_X,
    )

    print("4. Update verified pool of financial influencers...")
    update_verified_profile_pool(
        project_name=PROJECT_NAME_X,
        execution_date=PIPELINE_EXECUTION_DATE,
        input_file=ONBOARDING_RESULTS_FILE_X,
        verified_profile_pool=FINFLUENCER_POOL_FILE_X,
        prediction_threshold=PREDICTION_THRESHOLD_X,
        filter_by_stock_recommendation=False,
    )

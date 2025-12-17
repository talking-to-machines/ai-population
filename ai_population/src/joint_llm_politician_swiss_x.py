import os, json
import pandas as pd
from tqdm import tqdm
from datetime import datetime

tqdm.pandas()
from ai_population.config.base_config import GPT_MODEL
from ai_population.config.joint_llm_swiss_config import (
    PROJECT_NAME_X,
    PIPELINE_EXECUTION_DATE,
    PROFILE_SEARCH_START_DATE,
    PROFILE_SEARCH_END_DATE,
    NUM_POSTS_PER_PROFILE,
    POLITICIAN_POOL_FILE_X,
    POLITICIAN_PROFILE_METADATA_SEARCH_FILE_X,
    POLITICIAN_PROFILE_SEARCH_FILE_X,
    LOCAL_POLITICIAN_PROFILE_METADATA_FILE,
    LOCAL_POLITICIAN_PROFILE_POST_FILE,
    POLITICIAN_POST_DEMOGRAPHIC_INTERVIEW_FILE_X,
    POLITICIAN_POST_VOTING_INTERVIEW_FILE_X,
    DEMOGRAPHIC_INTERVIEW_REGEX_PATTERNS,
    VOTING_PREFERENCE_INTERVIEW_REGEX_PATTERNS,
)
from ai_population.src.utils import (
    perform_x_profile_search,
    perform_x_profile_metadata_search,
    perform_profile_interview,
    extract_llm_responses,
    coalesce_columns_by_regex,
)
from ai_population.prompts.prompt_template import (
    x_jointllm_demographic_interview_system_prompt,
    x_jointllm_demographic_interview_user_prompt,
    x_jointllm_voting_interview_user_prompt,
)

base_dir = os.path.dirname(os.path.abspath(__file__))

LOCAL_POLITICIAN_PROFILE_METADATA_FILE_FULL_PATH = os.path.join(
    base_dir,
    "../data",
    PROJECT_NAME_X,
    PIPELINE_EXECUTION_DATE,
    LOCAL_POLITICIAN_PROFILE_METADATA_FILE,
)
LOCAL_POLITICIAN_PROFILE_POST_FILE_FULL_PATH = os.path.join(
    base_dir,
    "../data",
    PROJECT_NAME_X,
    PIPELINE_EXECUTION_DATE,
    LOCAL_POLITICIAN_PROFILE_POST_FILE,
)


PROFILE_SEARCH_START_DATE = datetime.strptime(
    PROFILE_SEARCH_START_DATE, "%m-%d-%Y"
).strftime("%Y-%m-%d")
PROFILE_SEARCH_END_DATE = datetime.strptime(
    PROFILE_SEARCH_END_DATE, "%m-%d-%Y"
).strftime("%Y-%m-%d")


def conduct_demographic_interview(
    project_name: str,
    execution_date: str,
    profile_metadata_file: str,
    post_file: str,
    output_file: str,
) -> None:
    perform_profile_interview(
        project_name=project_name,
        execution_date=execution_date,
        gpt_model=GPT_MODEL,
        profile_metadata_file=profile_metadata_file,
        post_file=post_file,
        output_file=output_file,
        system_prompt_template=x_jointllm_demographic_interview_system_prompt,
        user_prompt_template=x_jointllm_demographic_interview_user_prompt,
        llm_response_field="x_jointllm_demographic_interview_llm_response",
        interview_type="x_jointllm_demographic_interview",
    )

    # Preprocess post interview results
    post_interview_results = pd.read_csv(
        os.path.join(base_dir, "../data", project_name, execution_date, output_file)
    )
    extracted_responses = post_interview_results[
        "x_jointllm_demographic_interview_llm_response"
    ].apply(extract_llm_responses)
    post_interview_results = pd.concat(
        [post_interview_results, extracted_responses], axis=1
    )
    # Merge identical columns from interview response
    post_interview_results = coalesce_columns_by_regex(
        post_interview_results, DEMOGRAPHIC_INTERVIEW_REGEX_PATTERNS
    )

    # Format past conversation
    post_interview_results["history"] = post_interview_results.apply(
        lambda row: json.dumps(
            [
                {
                    "role": "user",
                    "content": x_jointllm_demographic_interview_user_prompt,
                },
                {
                    "role": "assistant",
                    "content": row["x_jointllm_demographic_interview_llm_response"],
                },
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        axis=1,
    )

    # Save formatted interview results
    post_interview_results.to_csv(
        os.path.join(base_dir, "../data", project_name, execution_date, output_file),
        index=False,
    )


def conduct_voting_preference_interview(
    project_name: str,
    execution_date: str,
    profile_metadata_file: str,
    post_file: str,
    output_file: str,
) -> None:
    perform_profile_interview(
        project_name=project_name,
        execution_date=execution_date,
        gpt_model=GPT_MODEL,
        profile_metadata_file=profile_metadata_file,
        post_file=post_file,
        output_file=output_file,
        system_prompt_template=x_jointllm_demographic_interview_system_prompt,
        user_prompt_template=x_jointllm_voting_interview_user_prompt,
        llm_response_field="x_jointllm_voting_interview_llm_response",
        interview_type="x_jointllm_voting_interview",
        history_field="history",
    )

    # Preprocess post interview results
    post_interview_results = pd.read_csv(
        os.path.join(base_dir, "../data", project_name, execution_date, output_file)
    )
    extracted_responses = post_interview_results[
        "x_jointllm_voting_interview_llm_response"
    ].apply(extract_llm_responses)
    post_interview_results = pd.concat(
        [post_interview_results, extracted_responses], axis=1
    )
    # Merge identical columns from interview response
    post_interview_results = coalesce_columns_by_regex(
        post_interview_results, VOTING_PREFERENCE_INTERVIEW_REGEX_PATTERNS
    )

    # Include LLM model information
    post_interview_results["model"] = GPT_MODEL

    # Save formatted interview results
    post_interview_results.to_csv(
        os.path.join(base_dir, "../data", project_name, execution_date, output_file),
        index=False,
    )


if __name__ == "__main__":
    # Step 1: Perform profile search of identified politicians (profile metadata and posts) during search period
    print(
        "1. Perform profile search of identified politicians (profile metadata and recent posts) during search period"
    )
    if os.path.exists(LOCAL_POLITICIAN_PROFILE_METADATA_FILE_FULL_PATH):
        perform_x_profile_metadata_search(
            project_name=PROJECT_NAME_X,
            execution_date=PIPELINE_EXECUTION_DATE,
            input_file=POLITICIAN_POOL_FILE_X,
            output_file=POLITICIAN_PROFILE_METADATA_SEARCH_FILE_X,
            local_file=LOCAL_POLITICIAN_PROFILE_METADATA_FILE_FULL_PATH,
        )
    else:
        perform_x_profile_metadata_search(
            project_name=PROJECT_NAME_X,
            execution_date=PIPELINE_EXECUTION_DATE,
            input_file=POLITICIAN_POOL_FILE_X,
            output_file=LOCAL_POLITICIAN_PROFILE_METADATA_FILE,
        )

    if os.path.exists(LOCAL_POLITICIAN_PROFILE_POST_FILE_FULL_PATH):
        perform_x_profile_search(
            project_name=PROJECT_NAME_X,
            execution_date=PIPELINE_EXECUTION_DATE,
            input_file=POLITICIAN_POOL_FILE_X,
            output_file=POLITICIAN_PROFILE_SEARCH_FILE_X,
            start_date=PROFILE_SEARCH_START_DATE,
            end_date=PROFILE_SEARCH_END_DATE,
            num_posts_per_profile=NUM_POSTS_PER_PROFILE,
            local_file=LOCAL_POLITICIAN_PROFILE_POST_FILE_FULL_PATH,
        )
    else:
        perform_x_profile_search(
            project_name=PROJECT_NAME_X,
            execution_date=PIPELINE_EXECUTION_DATE,
            input_file=POLITICIAN_POOL_FILE_X,
            output_file=LOCAL_POLITICIAN_PROFILE_POST_FILE,
            start_date=PROFILE_SEARCH_START_DATE,
            end_date=PROFILE_SEARCH_END_DATE,
            num_posts_per_profile=NUM_POSTS_PER_PROFILE,
        )

    # Step 2: Perform demographic interview to infer demographic information
    print("2. Perform demographic interview to infer demographic information")
    conduct_demographic_interview(
        project_name=PROJECT_NAME_X,
        execution_date=PIPELINE_EXECUTION_DATE,
        profile_metadata_file=POLITICIAN_PROFILE_METADATA_SEARCH_FILE_X,
        post_file=POLITICIAN_PROFILE_SEARCH_FILE_X,
        output_file=POLITICIAN_POST_DEMOGRAPHIC_INTERVIEW_FILE_X,
    )

    # Step 3: Perform voting interview to infer digital election polling preferences
    print("3. Perform voting interview to infer digital election polling preferences")
    conduct_voting_preference_interview(
        project_name=PROJECT_NAME_X,
        execution_date=PIPELINE_EXECUTION_DATE,
        profile_metadata_file=POLITICIAN_POST_DEMOGRAPHIC_INTERVIEW_FILE_X,
        post_file=POLITICIAN_PROFILE_SEARCH_FILE_X,
        output_file=POLITICIAN_POST_VOTING_INTERVIEW_FILE_X,
    )

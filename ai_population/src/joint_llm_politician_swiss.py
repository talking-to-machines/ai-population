import os, json, argparse
import pandas as pd
from tqdm import tqdm
from datetime import datetime

tqdm.pandas()
from ai_population.config.base_config import GPT_MODEL
from ai_population.config.joint_llm_swiss_config import (
    PROJECT_NAME,
    POLITICIAN_PIPELINE,
    PROFILE_SEARCH_START_DATE,
    PROFILE_SEARCH_END_DATE,
    NUM_POSTS_PER_PROFILE,
    POLITICIAN_POOL_FILE_X,
    POLITICIAN_POOL_FILE_TIKTOK,
    POLITICIAN_PROFILE_METADATA_SEARCH_FILE_X,
    POLITICIAN_PROFILE_SEARCH_FILE_X,
    POLITICIAN_PROFILE_METADATA_SEARCH_FILE_TIKTOK,
    POLITICIAN_PROFILE_SEARCH_FILE_TIKTOK,
    LOCAL_POLITICIAN_PROFILE_METADATA_FILE_X,
    LOCAL_POLITICIAN_PROFILE_POST_FILE_X,
    LOCAL_POLITICIAN_PROFILE_METADATA_FILE_TIKTOK,
    LOCAL_POLITICIAN_PROFILE_POST_FILE_TIKTOK,
    POLITICIAN_POST_DEMOGRAPHIC_INTERVIEW_FILE,
    POLITICIAN_POST_DIGITAL_POLLING_INTERVIEW_FILE,
    POLITICIAN_DEMOGRAPHIC_INTERVIEW_REGEX_PATTERNS,
    DIGITAL_POLLING_REGEX_PATTERNS,
)
from ai_population.src.utils import (
    perform_x_profile_search,
    perform_x_profile_metadata_search,
    perform_tiktok_profile_search,
    perform_tiktok_profile_metadata_search,
    perform_video_transcription,
    perform_profile_interview_x_tiktok,
    extract_llm_responses,
    coalesce_columns_by_regex,
)
from ai_population.prompts.prompt_template import (
    jointllm_politician_system_prompt,
    jointllm_politician_demographic_interview_user_prompt,
    jointllm_politician_digital_polling_user_prompt,
)

base_dir = os.path.dirname(os.path.abspath(__file__))

LOCAL_POLITICIAN_PROFILE_METADATA_FILE_X_FULL_PATH = os.path.join(
    base_dir,
    "../data",
    PROJECT_NAME,
    POLITICIAN_PIPELINE,
    LOCAL_POLITICIAN_PROFILE_METADATA_FILE_X,
)
LOCAL_POLITICIAN_PROFILE_POST_FILE_X_FULL_PATH = os.path.join(
    base_dir,
    "../data",
    PROJECT_NAME,
    POLITICIAN_PIPELINE,
    LOCAL_POLITICIAN_PROFILE_POST_FILE_X,
)
LOCAL_POLITICIAN_PROFILE_METADATA_FILE_TIKTOK_FULL_PATH = os.path.join(
    base_dir,
    "../data",
    PROJECT_NAME,
    POLITICIAN_PIPELINE,
    LOCAL_POLITICIAN_PROFILE_METADATA_FILE_TIKTOK,
)
LOCAL_POLITICIAN_PROFILE_POST_FILE_TIKTOK_FULL_PATH = os.path.join(
    base_dir,
    "../data",
    PROJECT_NAME,
    POLITICIAN_PIPELINE,
    LOCAL_POLITICIAN_PROFILE_POST_FILE_TIKTOK,
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
    x_profile_metadata_file: str,
    x_post_file: str,
    tiktok_profile_metadata_file: str,
    tiktok_post_file: str,
    output_file: str,
    model_name: str = GPT_MODEL,
    together_ai_endpoint: str = None,
    grok_endpoint: str = None,
    batch_timeout_seconds: int = 7200,
) -> None:
    perform_profile_interview_x_tiktok(
        project_name=project_name,
        execution_date=execution_date,
        model_name=model_name,
        x_profile_metadata_file=x_profile_metadata_file,
        x_post_file=x_post_file,
        tiktok_profile_metadata_file=tiktok_profile_metadata_file,
        tiktok_post_file=tiktok_post_file,
        output_file=output_file,
        system_prompt_template=jointllm_politician_system_prompt,
        user_prompt_template=jointllm_politician_demographic_interview_user_prompt,
        llm_response_field="jointllm_politician_demographic_interview_llm_response",
        interview_type="jointllm_politician_demographic_interview",
        batch_timeout_seconds=batch_timeout_seconds,
        together_ai_endpoint=together_ai_endpoint,
        grok_endpoint=grok_endpoint,
    )

    # Preprocess post interview results
    post_interview_results = pd.read_csv(
        os.path.join(base_dir, "../data", project_name, execution_date, output_file)
    )
    extracted_responses = post_interview_results[
        "jointllm_politician_demographic_interview_llm_response"
    ].apply(extract_llm_responses)
    post_interview_results = pd.concat(
        [post_interview_results, extracted_responses], axis=1
    )
    # Merge identical columns from interview response
    post_interview_results = coalesce_columns_by_regex(
        post_interview_results, POLITICIAN_DEMOGRAPHIC_INTERVIEW_REGEX_PATTERNS
    )

    # Format past conversation
    post_interview_results["history"] = post_interview_results.apply(
        lambda row: json.dumps(
            [
                {
                    "role": "user",
                    "content": jointllm_politician_demographic_interview_user_prompt,
                },
                {
                    "role": "assistant",
                    "content": row[
                        "jointllm_politician_demographic_interview_llm_response"
                    ],
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


def conduct_digital_polling(
    project_name: str,
    execution_date: str,
    x_profile_metadata_file: str,
    x_post_file: str,
    tiktok_profile_metadata_file: str,
    tiktok_post_file: str,
    output_file: str,
    model_name: str = GPT_MODEL,
    together_ai_endpoint: str = None,
    grok_endpoint: str = None,
    batch_timeout_seconds: int = 7200,
) -> None:
    perform_profile_interview_x_tiktok(
        project_name=project_name,
        execution_date=execution_date,
        model_name=model_name,
        x_profile_metadata_file=x_profile_metadata_file,
        x_post_file=x_post_file,
        tiktok_profile_metadata_file=tiktok_profile_metadata_file,
        tiktok_post_file=tiktok_post_file,
        output_file=output_file,
        system_prompt_template=jointllm_politician_system_prompt,
        user_prompt_template=jointllm_politician_digital_polling_user_prompt,
        llm_response_field="jointllm_politician_digital_polling_llm_response",
        interview_type="jointllm_politician_digital_polling_interview",
        history_field="history",
        batch_timeout_seconds=batch_timeout_seconds,
        together_ai_endpoint=together_ai_endpoint,
        grok_endpoint=grok_endpoint,
    )

    # Preprocess post interview results
    post_interview_results = pd.read_csv(
        os.path.join(base_dir, "../data", project_name, execution_date, output_file)
    )
    extracted_responses = post_interview_results[
        "jointllm_politician_digital_polling_llm_response"
    ].apply(extract_llm_responses)
    post_interview_results = pd.concat(
        [post_interview_results, extracted_responses], axis=1
    )
    # Merge identical columns from interview response
    post_interview_results = coalesce_columns_by_regex(
        post_interview_results, DIGITAL_POLLING_REGEX_PATTERNS
    )

    # Include LLM model information
    post_interview_results["model"] = model_name

    # Save formatted interview results
    post_interview_results.to_csv(
        os.path.join(base_dir, "../data", project_name, execution_date, output_file),
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Swiss politician interview pipeline."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=GPT_MODEL,
        help="Model name to use for the interview. Pass an OpenAI model id (default) "
        "or a Together AI model id when --together-ai-endpoint is also set.",
    )
    parser.add_argument(
        "--together-ai-endpoint",
        type=str,
        default=None,
        help="Together AI base URL (serverless or dedicated endpoint). When set, "
        "the interview is routed through Together AI instead of OpenAI.",
    )
    parser.add_argument(
        "--grok-endpoint",
        type=str,
        default=None,
        help="xAI (Grok) base URL, typically https://api.x.ai/v1. When set, "
        "the interview is routed through xAI instead of OpenAI. "
        "Mutually exclusive with --together-ai-endpoint.",
    )
    parser.add_argument(
        "--batch-timeout-seconds",
        type=int,
        default=7200,
        help="Seconds to wait for the OpenAI batch job to complete before "
        "falling back to row-by-row API calls. Default: 7200 (2 hours).",
    )
    args = parser.parse_args()
    model_name = args.model_name
    together_ai_endpoint = args.together_ai_endpoint
    grok_endpoint = args.grok_endpoint
    batch_timeout_seconds = args.batch_timeout_seconds

    # # Step 1: Perform profile search of identified politicians with a X profile (profile metadata and posts) during search period
    # print(
    #     "1. Perform profile search of identified politicians with a X profile (profile metadata and recent posts) during search period"
    # )
    # if os.path.exists(LOCAL_POLITICIAN_PROFILE_METADATA_FILE_X_FULL_PATH):
    #     perform_x_profile_metadata_search(
    #         project_name=PROJECT_NAME,
    #         execution_date=POLITICIAN_PIPELINE,
    #         input_file=POLITICIAN_POOL_FILE_X,
    #         output_file=POLITICIAN_PROFILE_METADATA_SEARCH_FILE_X,
    #         local_file=LOCAL_POLITICIAN_PROFILE_METADATA_FILE_X_FULL_PATH,
    #     )
    # else:
    #     perform_x_profile_metadata_search(
    #         project_name=PROJECT_NAME,
    #         execution_date=POLITICIAN_PIPELINE,
    #         input_file=POLITICIAN_POOL_FILE_X,
    #         output_file=LOCAL_POLITICIAN_PROFILE_METADATA_FILE_X,
    #     )

    # if os.path.exists(LOCAL_POLITICIAN_PROFILE_POST_FILE_X_FULL_PATH):
    #     perform_x_profile_search(
    #         project_name=PROJECT_NAME,
    #         execution_date=POLITICIAN_PIPELINE,
    #         input_file=POLITICIAN_POOL_FILE_X,
    #         output_file=POLITICIAN_PROFILE_SEARCH_FILE_X,
    #         start_date=PROFILE_SEARCH_START_DATE,
    #         end_date=PROFILE_SEARCH_END_DATE,
    #         num_posts_per_profile=NUM_POSTS_PER_PROFILE,
    #         local_file=LOCAL_POLITICIAN_PROFILE_POST_FILE_X_FULL_PATH,
    #     )
    # else:
    #     perform_x_profile_search(
    #         project_name=PROJECT_NAME,
    #         execution_date=POLITICIAN_PIPELINE,
    #         input_file=POLITICIAN_POOL_FILE_X,
    #         output_file=LOCAL_POLITICIAN_PROFILE_POST_FILE_X,
    #         start_date=PROFILE_SEARCH_START_DATE,
    #         end_date=PROFILE_SEARCH_END_DATE,
    #         num_posts_per_profile=NUM_POSTS_PER_PROFILE,
    #     )

    # # Step 2: Perform profile search of identified politicians with a Tiktok profile (profile metadata and posts) during search period
    # print(
    #     "2. Perform profile search of identified politicians with a Tiktok profile (profile metadata and recent posts) during search period"
    # )
    # if os.path.exists(LOCAL_POLITICIAN_PROFILE_METADATA_FILE_TIKTOK_FULL_PATH):
    #     perform_tiktok_profile_metadata_search(
    #         project_name=PROJECT_NAME,
    #         execution_date=POLITICIAN_PIPELINE,
    #         input_file=POLITICIAN_POOL_FILE_TIKTOK,
    #         output_file=POLITICIAN_PROFILE_METADATA_SEARCH_FILE_TIKTOK,
    #         local_file=LOCAL_POLITICIAN_PROFILE_METADATA_FILE_TIKTOK_FULL_PATH,
    #     )
    # else:
    #     perform_tiktok_profile_metadata_search(
    #         project_name=PROJECT_NAME,
    #         execution_date=POLITICIAN_PIPELINE,
    #         input_file=POLITICIAN_POOL_FILE_TIKTOK,
    #         output_file=LOCAL_POLITICIAN_PROFILE_METADATA_FILE_TIKTOK,
    #     )

    # if os.path.exists(LOCAL_POLITICIAN_PROFILE_POST_FILE_TIKTOK_FULL_PATH):
    #     perform_tiktok_profile_search(
    #         project_name=PROJECT_NAME,
    #         execution_date=POLITICIAN_PIPELINE,
    #         input_file=POLITICIAN_POOL_FILE_TIKTOK,
    #         output_file=POLITICIAN_PROFILE_SEARCH_FILE_TIKTOK,
    #         start_date=PROFILE_SEARCH_START_DATE,
    #         end_date=PROFILE_SEARCH_END_DATE,
    #         num_posts_per_profile=NUM_POSTS_PER_PROFILE,
    #         local_file=LOCAL_POLITICIAN_PROFILE_POST_FILE_TIKTOK_FULL_PATH,
    #     )
    # else:
    #     perform_tiktok_profile_search(
    #         project_name=PROJECT_NAME,
    #         execution_date=POLITICIAN_PIPELINE,
    #         input_file=POLITICIAN_POOL_FILE_TIKTOK,
    #         output_file=LOCAL_POLITICIAN_PROFILE_POST_FILE_TIKTOK,
    #         start_date=PROFILE_SEARCH_START_DATE,
    #         end_date=PROFILE_SEARCH_END_DATE,
    #         num_posts_per_profile=NUM_POSTS_PER_PROFILE,
    #     )
    #     perform_video_transcription(
    #         project_name=PROJECT_NAME,
    #         execution_date=POLITICIAN_PIPELINE,
    #         video_file=LOCAL_POLITICIAN_PROFILE_POST_FILE_TIKTOK,
    #     )

    # Step 3: Perform demographic interview to infer demographic information
    print("3. Perform demographic interview to infer demographic information")
    conduct_demographic_interview(
        project_name=PROJECT_NAME,
        execution_date=POLITICIAN_PIPELINE,
        x_profile_metadata_file=POLITICIAN_PROFILE_METADATA_SEARCH_FILE_X,
        x_post_file=POLITICIAN_PROFILE_SEARCH_FILE_X,
        tiktok_profile_metadata_file=POLITICIAN_PROFILE_METADATA_SEARCH_FILE_TIKTOK,
        tiktok_post_file=POLITICIAN_PROFILE_SEARCH_FILE_TIKTOK,
        output_file=POLITICIAN_POST_DEMOGRAPHIC_INTERVIEW_FILE,
        model_name=model_name,
        together_ai_endpoint=together_ai_endpoint,
        grok_endpoint=grok_endpoint,
        batch_timeout_seconds=batch_timeout_seconds,
    )

    # Step 4: Perform digital polling to infer election polling preferences
    print("4. Perform digital polling to infer election polling preferences")
    conduct_digital_polling(
        project_name=PROJECT_NAME,
        execution_date=POLITICIAN_PIPELINE,
        x_profile_metadata_file=POLITICIAN_PROFILE_METADATA_SEARCH_FILE_X,
        x_post_file=POLITICIAN_PROFILE_SEARCH_FILE_X,
        tiktok_profile_metadata_file=POLITICIAN_PROFILE_METADATA_SEARCH_FILE_TIKTOK,
        tiktok_post_file=POLITICIAN_PROFILE_SEARCH_FILE_TIKTOK,
        output_file=POLITICIAN_POST_DIGITAL_POLLING_INTERVIEW_FILE,
        model_name=model_name,
        together_ai_endpoint=together_ai_endpoint,
        grok_endpoint=grok_endpoint,
        batch_timeout_seconds=batch_timeout_seconds,
    )

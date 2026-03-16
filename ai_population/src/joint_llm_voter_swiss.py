import os
import pandas as pd
import argparse
from tqdm import tqdm
import json

tqdm.pandas()
from ai_population.config.joint_llm_swiss_config import (
    PROJECT_NAME,
    VOTER_PIPELINE_X,
    VOTER_PIPELINE_TIKTOK,
    VOTER_KEYWORD_SEARCH_FILE_X,
    VOTER_KEYWORD_SEARCH_FILE_TIKTOK,
    VOTER_SEARCH_TERMS_X,
    VOTER_SEARCH_TERMS_TIKTOK,
    VOTER_KEYWORD_PROFILE_METADATA_FILE_X,
    VOTER_KEYWORD_PROFILE_METADATA_FILE_TIKTOK,
    VOTER_KEYWORD_PROFILE_POSTS_FILE_X,
    VOTER_KEYWORD_PROFILE_POSTS_FILE_TIKTOK,
    VOTER_ENTITY_GEOGRAPHIC_EXCLUSION_CRITERIA_FILE_X,
    VOTER_ENTITY_GEOGRAPHIC_EXCLUSION_CRITERIA_FILE_TIKTOK,
    VOTER_QUOTA_INCLUSION_CRITERIA_FILE_X,
    VOTER_QUOTA_INCLUSION_CRITERIA_FILE_TIKTOK,
    VOTER_ELIGIBLE_PROFILE_SEARCH_FILE_X,
    VOTER_ELIGIBLE_PROFILE_SEARCH_FILE_TIKTOK,
    VOTER_TARGET_STRATIFICATION_FRAME_X,
    VOTER_CURRENT_STRATIFICATION_FRAME_X,
    VOTER_TARGET_STRATIFICATION_FRAME_TIKTOK,
    VOTER_CURRENT_STRATIFICATION_FRAME_TIKTOK,
    VOTER_DIGITAL_POLLING_FILE_X,
    VOTER_DIGITAL_POLLING_FILE_TIKTOK,
    PROFILE_SEARCH_START_DATE,
    PROFILE_SEARCH_END_DATE,
    NUM_POSTS_PER_KEYWORD,
    NUM_POSTS_PER_PROFILE_FROM_KEYWORD_SEARCH,
    NUM_POSTS_PER_PROFILE,
    VOTER_ENTITY_GEOGRAPHIC_INCLUSION_REGEX_PATTERNS,
    DIGITAL_POLLING_REGEX_PATTERNS,
)
from config.base_config import GPT_MODEL

from src.utils import (
    perform_x_keyword_search,
    perform_tiktok_keyword_search,
    perform_x_profile_metadata_search,
    perform_tiktok_profile_metadata_search,
    perform_x_profile_search,
    perform_tiktok_profile_search,
    perform_video_transcription,
    extract_llm_responses,
    perform_profile_interview,
    coalesce_columns_by_regex,
)
from prompts.prompt_template import (
    x_jointllm_voter_entity_geographic_exclusion_criteria_system_prompt,
    tiktok_jointllm_voter_entity_geographic_exclusion_criteria_system_prompt,
    jointllm_voter_entity_geographic_exclusion_criteria_user_prompt,
    jointllm_voter_digital_polling_user_prompt,
)

base_dir = os.path.dirname(os.path.abspath(__file__))


def conduct_demographic_interview(
    project_name: str,
    execution_date: str,
    profile_metadata_file: str,
    post_file: str,
    output_file: str,
    system_prompt_template: str,
    user_prompt_template: str,
    interview_type: str,
) -> None:
    perform_profile_interview(
        project_name=project_name,
        execution_date=execution_date,
        gpt_model=GPT_MODEL,
        profile_metadata_file=profile_metadata_file,
        post_file=post_file,
        output_file=output_file,
        system_prompt_template=system_prompt_template,
        user_prompt_template=user_prompt_template,
        llm_response_field="entity_geographic_exclusion_llm_response",
        interview_type=interview_type,
    )

    # Preprocess post interview results
    post_interview_profile_metadata = pd.read_csv(
        os.path.join(base_dir, "../data", project_name, execution_date, output_file)
    )
    extracted_responses = post_interview_profile_metadata[
        "jointllm_voter_demographic_interview_llm_response"
    ].apply(extract_llm_responses)
    post_interview_profile_metadata = pd.concat(
        [post_interview_profile_metadata, extracted_responses], axis=1
    )

    # Merge identical columns from interview response
    post_interview_profile_metadata = coalesce_columns_by_regex(
        post_interview_profile_metadata,
        VOTER_ENTITY_GEOGRAPHIC_INCLUSION_REGEX_PATTERNS,
    )

    # Filter out profiles that are non-individuals (entity inclusion criteria)
    filtered_profile_metadata = post_interview_profile_metadata[
        post_interview_profile_metadata["ENTITY - symbol"] == "ENT1"
    ].reset_index(drop=True)

    # Filter out profiles that are not based in Canada (geographic inclusion criteria)
    filtered_profile_metadata = filtered_profile_metadata[
        filtered_profile_metadata[f"CITIZENSHIP - symbol"] == "CIT1"
    ].reset_index(drop=True)

    # Format past conversation
    filtered_profile_metadata["history"] = filtered_profile_metadata.apply(
        lambda row: json.dumps(
            [
                {
                    "role": "user",
                    "content": user_prompt_template,
                },
                {
                    "role": "assistant",
                    "content": row["jointllm_voter_demographic_interview_llm_response"],
                },
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        axis=1,
    )

    # Save profiles that meet entity and geographic inclusion criteria
    filtered_profile_metadata.to_csv(
        os.path.join(base_dir, "../data", project_name, execution_date, output_file),
        index=False,
    )


def apply_quota_inclusion_criteria(
    project_name: str,
    execution_date: str,
    input_file: str,
    output_file: str,
    target_stratification_frame: str,
    current_stratification_frame: str,
) -> bool:
    # Load target stratification frame
    target_stratification_df = pd.read_csv(
        os.path.join(base_dir, "../data", project_name, target_stratification_frame)
    )

    # Load or initialize current stratification frame
    current_strat_path = os.path.join(
        base_dir, "../data", project_name, execution_date, current_stratification_frame
    )
    if os.path.exists(current_strat_path):
        current_stratification_df = pd.read_csv(current_strat_path)
    else:
        current_stratification_df = target_stratification_df.copy()
        current_stratification_df["Count"] = 0

    # Load input file (voter profiles with demographic attributes)
    voters_df = pd.read_csv(
        os.path.join(base_dir, "../data", project_name, execution_date, input_file)
    )

    # TODO Need to rename columns to align with stratification frame column names

    # Demographic columns are all stratification frame columns except "Count"
    demographic_cols = [
        col for col in target_stratification_df.columns if col != "Count"
    ]

    # Map each voter to a cell in the stratification frame
    eligible_voters = []
    for _, voter in voters_df.iterrows():
        # Skip voter if any demographic column is missing from their data
        if not all(col in voter.index for col in demographic_cols):
            continue

        # Find the matching cell based on demographic attributes
        mask = pd.Series(True, index=target_stratification_df.index)
        for col in demographic_cols:
            mask = mask & (target_stratification_df[col].astype(str) == str(voter[col]))

        matching_indices = target_stratification_df[mask].index
        if len(matching_indices) == 0:
            continue
        if len(matching_indices) > 1:
            raise ValueError(
                f"Voter matched {len(matching_indices)} cells in the stratification frame. "
                f"Expected at most 1. Matched rows: {matching_indices.tolist()}"
            )
        cell_idx = matching_indices[0]

        # Drop voter if cell quota is already full
        if (
            current_stratification_df.loc[cell_idx, "Count"]
            >= target_stratification_df.loc[cell_idx, "Count"]
        ):
            continue

        # Assign voter to cell and increment count
        current_stratification_df.loc[cell_idx, "Count"] += 1
        eligible_voters.append(voter)

    # Save eligible voters to output file (append if file already exists)
    new_eligible_df = (
        pd.DataFrame(eligible_voters).reset_index(drop=True)
        if eligible_voters
        else pd.DataFrame(columns=voters_df.columns)
    )
    output_path = os.path.join(
        base_dir, "../data", project_name, execution_date, output_file
    )
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        eligible_df = pd.concat([existing_df, new_eligible_df], ignore_index=True)
    else:
        eligible_df = new_eligible_df
    eligible_df.to_csv(output_path, index=False)

    # Save current stratification frame for future runs
    current_stratification_df.to_csv(current_strat_path, index=False)

    # Return True if all cells match their target counts, False otherwise
    return (
        current_stratification_df["Count"] == target_stratification_df["Count"]
    ).all()


def conduct_digital_polling(
    project_name: str,
    execution_date: str,
    profile_metadata_file: str,
    post_file: str,
    output_file: str,
    system_prompt_template: str,
    user_prompt_template: str,
    interview_type: str,
) -> None:
    perform_profile_interview(
        project_name=project_name,
        execution_date=execution_date,
        gpt_model=GPT_MODEL,
        profile_metadata_file=profile_metadata_file,
        post_file=post_file,
        output_file=output_file,
        system_prompt_template=system_prompt_template,
        user_prompt_template=user_prompt_template,
        llm_response_field="jointllm_voter_digital_polling_llm_response",
        interview_type=interview_type,
        history_field="history",
    )

    # Preprocess post interview results
    post_interview_results = pd.read_csv(
        os.path.join(base_dir, "../data", project_name, execution_date, output_file)
    )
    extracted_responses = post_interview_results[
        "jointllm_voter_digital_polling_llm_response"
    ].apply(extract_llm_responses)
    post_interview_results = pd.concat(
        [post_interview_results, extracted_responses], axis=1
    )
    # Merge identical columns from interview response
    post_interview_results = coalesce_columns_by_regex(
        post_interview_results, DIGITAL_POLLING_REGEX_PATTERNS
    )

    # Include LLM model information
    post_interview_results["model"] = GPT_MODEL

    # Save formatted interview results
    post_interview_results.to_csv(
        os.path.join(base_dir, "../data", project_name, execution_date, output_file),
        index=False,
    )


def define_pipeline_constants(platform: str) -> dict:
    if platform == "x":
        constants = {
            "project_name": PROJECT_NAME,
            "pipeline_name": VOTER_PIPELINE_X,
            "search_term_list": VOTER_SEARCH_TERMS_X,
            "keyword_search_file": VOTER_KEYWORD_SEARCH_FILE_X,
            "keyword_profile_metadata_file": VOTER_KEYWORD_PROFILE_METADATA_FILE_X,
            "keyword_profile_posts_file": VOTER_KEYWORD_PROFILE_POSTS_FILE_X,
            "entity_geographic_exclusion_criteria_file": VOTER_ENTITY_GEOGRAPHIC_EXCLUSION_CRITERIA_FILE_X,
            "entity_geographic_exclusion_criteria_system_prompt": x_jointllm_voter_entity_geographic_exclusion_criteria_system_prompt,
            "entity_geographic_exclusion_criteria_user_prompt": jointllm_voter_entity_geographic_exclusion_criteria_user_prompt,
            "entity_geographic_exclusion_criteria_interview_type": "x_jointllm_voter_entity_geographic_exclusion_criteria_interview",
            "quota_inclusion_criteria_file": VOTER_QUOTA_INCLUSION_CRITERIA_FILE_X,
            "eligible_profile_posts_file": VOTER_ELIGIBLE_PROFILE_SEARCH_FILE_X,
            "target_stratification_frame": VOTER_TARGET_STRATIFICATION_FRAME_X,
            "current_stratification_frame": VOTER_CURRENT_STRATIFICATION_FRAME_X,
            "digital_polling_file": VOTER_DIGITAL_POLLING_FILE_X,
            "digital_polling_user_prompt": jointllm_voter_digital_polling_user_prompt,
            "digital_polling_interview_type": "x_jointllm_voter_digital_polling_interview",
        }
    else:  # tiktok
        constants = {
            "project_name": PROJECT_NAME,
            "pipeline_name": VOTER_PIPELINE_TIKTOK,
            "search_term_list": VOTER_SEARCH_TERMS_TIKTOK,
            "keyword_search_file": VOTER_KEYWORD_SEARCH_FILE_TIKTOK,
            "keyword_profile_metadata_file": VOTER_KEYWORD_PROFILE_METADATA_FILE_TIKTOK,
            "keyword_profile_posts_file": VOTER_KEYWORD_PROFILE_POSTS_FILE_TIKTOK,
            "entity_geographic_exclusion_criteria_file": VOTER_ENTITY_GEOGRAPHIC_EXCLUSION_CRITERIA_FILE_TIKTOK,
            "entity_geographic_exclusion_criteria_system_prompt": tiktok_jointllm_voter_entity_geographic_exclusion_criteria_system_prompt,
            "entity_geographic_exclusion_criteria_user_prompt": jointllm_voter_entity_geographic_exclusion_criteria_user_prompt,
            "entity_geographic_exclusion_criteria_interview_type": "tiktok_jointllm_voter_entity_geographic_exclusion_criteria_interview",
            "quota_inclusion_criteria_file": VOTER_QUOTA_INCLUSION_CRITERIA_FILE_TIKTOK,
            "eligible_profile_posts_file": VOTER_ELIGIBLE_PROFILE_SEARCH_FILE_TIKTOK,
            "target_stratification_frame": VOTER_TARGET_STRATIFICATION_FRAME_TIKTOK,
            "current_stratification_frame": VOTER_CURRENT_STRATIFICATION_FRAME_TIKTOK,
            "digital_polling_file": VOTER_DIGITAL_POLLING_FILE_TIKTOK,
            "digital_polling_user_prompt": jointllm_voter_digital_polling_user_prompt,
            "digital_polling_interview_type": "tiktok_jointllm_voter_digital_polling_interview",
        }
    return constants


if __name__ == "__main__":
    # Load pipeline arguments (e.g., platform to run pipeline for - X vs. TikTok)
    parser = argparse.ArgumentParser(description="Run Swiss voter polling pipeline.")
    parser.add_argument(
        "--platform",
        type=str,
        required=True,
        help="Social media platform to run the pipeline for (e.g., 'x' or 'tiktok')",
    )
    args = parser.parse_args()
    platform = args.platform

    if platform not in ["x", "tiktok"]:
        raise ValueError(
            "Invalid platform specified. Supported platforms are 'x' and 'tiktok'."
        )

    # Step 0: Define pipeline constants based on social media platform (i.e., X vs. TikTok)
    constants = define_pipeline_constants(platform=platform)

    STRATIFICATION_FRAME_NOT_FILED = True
    while STRATIFICATION_FRAME_NOT_FILED:
        # Step 1: Get Pool
        print("Step 1: Generate a subject pool of social media users")
        ## Perform key word search for social media posts
        print("Perform keyword search using predefined list of search terms...")
        if platform == "x":
            perform_x_keyword_search(
                project_name=constants["project_name"],
                execution_date=constants["pipeline_name"],
                search_terms=constants["search_term_list"],
                output_file=constants["keyword_search_file"],
                num_post_per_keyword=NUM_POSTS_PER_KEYWORD,
            )
        else:
            perform_tiktok_keyword_search(
                project_name=constants["project_name"],
                execution_date=constants["pipeline_name"],
                search_terms=constants["search_term_list"],
                output_file=constants["keyword_search_file"],
                num_post_per_keyword=NUM_POSTS_PER_KEYWORD,
            )

        ## Extract profile metadata for search results
        print(
            "Perform profile metadata search for profiles obtained via keyword search results..."
        )
        if platform == "x":
            perform_x_profile_metadata_search(
                project_name=constants["project_name"],
                execution_date=constants["pipeline_name"],
                input_file=os.path.join(
                    constants["pipeline_name"], constants["keyword_search_file"]
                ),
                output_file=constants["keyword_profile_metadata_file"],
            )
            perform_x_profile_search(
                project_name=PROJECT_NAME,
                execution_date=constants["pipeline_name"],
                input_file=constants["keyword_profile_metadata_file"],
                output_file=constants["keyword_profile_posts_file"],
                start_date=PROFILE_SEARCH_START_DATE,
                end_date=PROFILE_SEARCH_END_DATE,
                num_posts_per_profile=NUM_POSTS_PER_PROFILE_FROM_KEYWORD_SEARCH,
            )

        else:
            perform_tiktok_profile_metadata_search(
                project_name=constants["project_name"],
                execution_date=constants["pipeline_name"],
                input_file=os.path.join(
                    constants["pipeline_name"], constants["keyword_search_file"]
                ),
                output_file=constants["keyword_profile_metadata_file"],
            )
            perform_tiktok_profile_search(
                project_name=PROJECT_NAME,
                execution_date=constants["pipeline_name"],
                input_file=constants["keyword_profile_metadata_file"],
                output_file=constants["keyword_profile_posts_file"],
                start_date=PROFILE_SEARCH_START_DATE,
                end_date=PROFILE_SEARCH_END_DATE,
                num_posts_per_profile=NUM_POSTS_PER_PROFILE_FROM_KEYWORD_SEARCH,
            )
            perform_video_transcription(
                project_name=PROJECT_NAME,
                execution_date=constants["pipeline_name"],
                video_file=constants["keyword_profile_posts_file"],
            )

        # Step 2: Filter users based on exclusion criteria and map into stratification frame
        print(
            "Step 2: Filter users based on exclusion criteria and map into stratification frame"
        )
        # Apply geographic and entity filters to exclude users who are unlikely to reside in Switzerland and related to organisations (e.g., news outlets, NGOs) and bots
        print("Apply geographic and entity filters...")
        conduct_demographic_interview(
            project_name=constants["project_name"],
            execution_date=constants["pipeline_name"],
            profile_metadata_file=constants["keyword_profile_metadata_file"],
            post_file=constants["keyword_profile_posts_file"],
            output_file=constants["entity_geographic_exclusion_criteria_file"],
            system_prompt_template=constants[
                "entity_geographic_exclusion_criteria_system_prompt"
            ],
            user_prompt_template=constants[
                "entity_geographic_exclusion_criteria_user_prompt"
            ],
            interview_type=constants[
                "entity_geographic_exclusion_criteria_interview_type"
            ],
        )

        # Step 3: Apply quota inclusion criteria to identify eligible profiles for polling
        print(
            "Step 3: Apply quota inclusion criteria to identify eligible profiles for polling"
        )
        ## Apply quota inclusion criteria to identify eligible profiles for polling based on pre-defined stratification frame
        quota_inclusion_criteria_result = apply_quota_inclusion_criteria(
            project_name=constants["project_name"],
            execution_date=constants["pipeline_name"],
            input_file=constants["entity_geographic_exclusion_criteria_file"],
            output_file=constants["quota_inclusion_criteria_file"],
            target_stratification_frame=constants["target_stratification_frame"],
            current_stratification_frame=constants["current_stratification_frame"],
        )
        STRATIFICATION_FRAME_NOT_FILED = quota_inclusion_criteria_result

    # Step 4: Extract posts from eligible profiles during polling period
    print("Step 4: Extract posts from eligible profiles during polling period")
    profile_latest_videos = perform_x_profile_search(
        project_name=constants["project_name"],
        execution_date=constants["pipeline_name"],
        input_file=constants["quota_inclusion_criteria_file"],
        output_file=constants["eligible_profile_posts_file"],
        start_date=PROFILE_SEARCH_START_DATE,
        end_date=PROFILE_SEARCH_END_DATE,
        num_posts_per_profile=NUM_POSTS_PER_PROFILE,
    )

    # Perform digital election polling on eligible voters
    print("Step 5: Perform digital election polling of eligible voters")
    conduct_digital_polling(
        project_name=constants["project_name"],
        execution_date=constants["pipeline_name"],
        profile_metadata_file=constants["quota_inclusion_criteria_file"],
        post_file=constants["eligible_profile_posts_file"],
        output_file=constants["digital_polling_file"],
        system_prompt_template=constants["digital_polling_system_prompt"],
        user_prompt_template=constants["digital_polling_user_prompt"],
        interview_type=constants["digital_polling_interview_type"],
    )

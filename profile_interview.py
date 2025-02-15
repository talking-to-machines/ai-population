import os
import pandas as pd
from config import PROJECT
from utils import (
    extract_profile_id,
    extract_video_transcripts,
    construct_system_prompt,
    create_batch_file,
    batch_query,
)
from prompt_template import (
    finfluencer_identification_user_prompt,
    interview_user_prompt,
)


def perform_profile_interview(is_interview: bool) -> None:
    # Load profile and video metadata
    print("Loading profile and video metadata...")
    if is_interview:
        profile_metadata = pd.read_csv(
            f"data/{PROJECT}/profile_metadata_postinterview.csv"
        )
    else:
        profile_metadata = pd.read_csv(
            f"data/{PROJECT}/profilesearch_profile_metadata_sample.csv"
        )
    video_metadata = pd.read_csv(f"data/{PROJECT}/profilesearch_video_metadata.csv")
    video_metadata["createTimeISO"] = pd.to_datetime(video_metadata["createTimeISO"])

    # Preprocess profile and video metadata
    print("Preprocess profile and video metadata...")
    video_metadata["profile_id"] = video_metadata["authorMeta"].apply(
        extract_profile_id
    )
    video_metadata["profile_id"] = video_metadata["profile_id"].astype(str)
    profile_metadata["id"] = profile_metadata["id"].astype(str)

    # Generate system and user prompts
    print("Generate system and user prompts...")
    if is_interview:  # Finfluencer interview
        profile_metadata["interview_user_prompt"] = interview_user_prompt
        profile_metadata["interview_system_prompt"] = profile_metadata.apply(
            construct_system_prompt, args=(is_interview,), axis=1
        )

    else:  # Finfluencer identification
        profile_metadata["transcripts_combined"] = profile_metadata["id"].apply(
            extract_video_transcripts, args=(video_metadata,)
        )
        profile_metadata["identification_user_prompt"] = (
            finfluencer_identification_user_prompt
        )
        profile_metadata["identification_system_prompt"] = profile_metadata.apply(
            construct_system_prompt, args=(is_interview,), axis=1
        )

        # Generate custom ids
        profile_metadata = profile_metadata.reset_index(drop=False)
        profile_metadata.rename(columns={"index": "custom_id"}, inplace=True)

    # Create folder to contain batch files
    batch_file_dir = f"data/{PROJECT}/batch-files"
    os.makedirs(batch_file_dir, exist_ok=True)

    # Perform batch query for survey questions
    print("Prepare OpenAI batch files for batch API...")
    if is_interview:  # Finfluencer interview
        system_prompt_field = "interview_system_prompt"
        user_prompt_field = "interview_user_prompt"
        llm_response_field = "interview_llm_response"
    else:
        system_prompt_field = "identification_system_prompt"
        user_prompt_field = "identification_user_prompt"
        llm_response_field = "identification_llm_response"

    batch_file_dir = create_batch_file(
        profile_metadata,
        system_prompt_field=system_prompt_field,
        project_name=PROJECT,
        user_prompt_field=user_prompt_field,
        batch_file_name="batch_input.jsonl",
    )

    print("Perform batch query using OpenAI API...")
    llm_responses = batch_query(
        batch_input_file_dir="batch_input.jsonl",
        batch_output_file_dir="batch_output.jsonl",
        project_name=PROJECT,
    )
    llm_responses.rename(columns={"query_response": llm_response_field}, inplace=True)

    # Merge LLM response with original dataset
    print("Merge LLM response with original dataset...")
    profile_metadata["custom_id"] = profile_metadata["custom_id"].astype("int64")
    llm_responses["custom_id"] = llm_responses["custom_id"].astype("int64")
    profile_metadata_with_responses = pd.merge(
        left=profile_metadata,
        right=llm_responses,
        on="custom_id",
    )

    # Save profile metadata after analysis into CSV file
    print("Saving profile metadata with analysis...")
    profile_metadata_with_responses.to_csv(
        f"data/{PROJECT}/profile_metadata_postinterview.csv", index=False
    )


if __name__ == "__main__":
    perform_profile_interview(is_interview=False)
    perform_profile_interview(is_interview=True)

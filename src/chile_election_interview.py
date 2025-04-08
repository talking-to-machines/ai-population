import os
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm

tqdm.pandas()
from config.base_config import GPT_MODEL
from config.chile_election_config import (
    PROJECT,
    SEARCH_TERMS_FILE,
    KEYWORD_SEARCH_VIDEO_METADATA_FILE,
    # PROFILE_SEARCH_VIDEO_METADATA_FILE,
    KEYWORD_SEARCH_PROFILE_METADATA_FILE,
    # PROFILE_SEARCH_PROFILE_METADATA_FILE,
    # POLLED_PROFILES_FILE,
    TEMPORAL_INCLUSION_PERIOD,
    # PROFILE_METADATA_POST_PROFILE_PROMPT_FILE,
    # PROFILE_METADATA_POST_TEMPORAL_INCLUSION_FILE,
    # PROFILE_METADATA_POST_GEOGRAPHY_EXCLUSION_FILE,
    # PROFILE_METADATA_POST_ENTITY_GEOGRAPHIC_INCLUSION_FILE,
    # PROFILE_METADATA_POST_POLLING_FILE,
)
from src.utils import (
    # build_profile_prompt,
    extract_llm_responses,
    perform_profile_interview_shorten,
    construct_system_prompt,
    construct_user_prompt,
    extract_video_transcripts,
    calculate_profile_engagement,
)
from src.keyword_search import perform_keyword_search

# from src.profile_search import perform_profile_search
from prompts.prompt_template import profile_prompt_template

base_dir = os.path.dirname(os.path.abspath(__file__))


def apply_temporal_inclusion_criteria(
    project_name: str,
    profile_metadata_input_file: str,
    profile_metadata_output_file: str,
    polled_profiles_file: str,
) -> None:
    print("Load profile metadata...")
    profile_metadata = pd.read_csv(
        f"{base_dir}/../data/{project_name}/{profile_metadata_input_file}"
    )

    print("Exclude profiles that have been polled within the last N days...")
    polled_profiles_file_path = Path(
        f"{base_dir}/../data/{project_name}/{polled_profiles_file}"
    )
    if polled_profiles_file_path.exists():
        polled_profiles = pd.read_csv(polled_profiles_file_path)
        polled_profiles["poll_date"] = pd.to_datetime(polled_profiles["poll_date"])

        # Identify profiles that were polled within the last N days
        recently_polled_profiles = polled_profiles[
            polled_profiles["poll_date"]
            >= datetime.today().date() - timedelta(days=TEMPORAL_INCLUSION_PERIOD)
        ].reset_index(drop=True)

        # Exclude profiles that were polled within the last N days
        sampled_profile_metadata = profile_metadata[
            ~profile_metadata["profile"].isin(recently_polled_profiles["profile"])
        ]
        sampled_profile_metadata.to_csv(
            f"{base_dir}/../data/{project_name}/{profile_metadata_output_file}",
            index=False,
        )

        # Update polled profiles with profiles that will be polled in the current survey iteration
        newly_polled_profiles = sampled_profile_metadata[["profile"]]
        newly_polled_profiles["poll_date"] = datetime.today().date()
        updated_polled_profiles = pd.concat(
            [recently_polled_profiles, newly_polled_profiles], ignore_index=True
        )
        updated_polled_profiles.to_csv(
            f"{base_dir}/../data/{project_name}/{polled_profiles_file}", index=False
        )

    else:  # If no profiles have been polled yet, all existing profiles will be polled
        sampled_profile_metadata = profile_metadata
        sampled_profile_metadata.to_csv(
            f"{base_dir}/../data/{project_name}/{profile_metadata_output_file}",
            index=False,
        )

        newly_polled_profiles = sampled_profile_metadata[["profile"]]
        newly_polled_profiles["poll_date"] = datetime.today().date()
        newly_polled_profiles.to_csv(
            f"{base_dir}/../data/{project_name}/{polled_profiles_file}", index=False
        )

    return None


def apply_null_geography_exclusion_criteria(
    project_name: str,
    profile_metadata_input_file: str,
    profile_metadata_output_file: str,
) -> None:
    print("Load profile metadata...")
    sampled_profile_metadata = pd.read_csv(
        f"{base_dir}/../data/{project_name}/{profile_metadata_input_file}"
    )

    print("Exclude profiles without self-reported location information...")
    sampled_profile_metadata = sampled_profile_metadata[
        sampled_profile_metadata["region"].notnull()
    ].reset_index(drop=True)
    sampled_profile_metadata.to_csv(
        f"{base_dir}/../data/{project_name}/{profile_metadata_output_file}", index=False
    )

    return None


def apply_entity_geographic_inclusion_criteria(
    project_name: str,
    profile_metadata_input_file: str,
    profile_metadata_output_file: str,
) -> None:

    # Perform entity geographic inclusion criteria interview
    perform_profile_interview_shorten(
        project_name=project_name,
        gpt_model=GPT_MODEL,
        profile_metadata_input_file=profile_metadata_input_file,
        profile_metadata_output_file=profile_metadata_output_file,
        system_prompt_field="entity_geographic_inclusion_system_prompt",
        user_prompt_field="entity_geographic_inclusion_user_prompt",
        llm_response_field="entity_geographic_inclusion_llm_response",
        interview_type="entity_geographic_inclusion",
        batch_interview=True,
    )

    # Preprocess post interview results
    post_interview_profile_metadata = pd.read_csv(
        f"{base_dir}/../data/{project_name}/{profile_metadata_output_file}"
    )
    extracted_responses = post_interview_profile_metadata[
        "entity_geographic_inclusion_llm_response"
    ].apply(extract_llm_responses)
    post_interview_profile_metadata = pd.concat(
        [post_interview_profile_metadata, extracted_responses], axis=1
    )

    # Filter out profiles that are non-individuals (entity inclusion criteria)
    filtered_profile_metadata = post_interview_profile_metadata[
        post_interview_profile_metadata[
            "Is this an account of a real-life existing person, or of another kind of entity? - category"
        ]
        == "Person"
    ].reset_index(drop=True)

    # Filter out profiles that are not based in Canada (geographic inclusion criteria)
    filtered_profile_metadata = filtered_profile_metadata[
        filtered_profile_metadata[
            "Does the user of this TikTok account live in Canada? - category"
        ]
        == "Yes"
    ].reset_index(drop=True)

    # Save profiles that meet entity and geographic inclusion criteria
    filtered_profile_metadata.to_csv(
        f"{base_dir}/../data/{project_name}/{profile_metadata_output_file}", index=False
    )

    return None


def apply_quota_inclusion_criteria(
    profile: pd.Series,
) -> pd.Series:  # TODO to be implemented

    return None


def conduct_polling(
    project_name: str,
    profile: pd.Series,
    profile_latest_videos: pd.DataFrame,
    polling_results_file: str,
    poll_date: datetime,
) -> None:
    # Format past video transcripts
    video_transcripts_combined = extract_video_transcripts(
        profile_id=profile["id"], video_metadata=profile_latest_videos
    )

    # Construct profile prompt
    profile["profile_prompt"] = profile_prompt_template.format(
        profile_image=profile["avatar"],
        profile_name=profile["profile"],
        profile_nickname=profile["nickName"],
        verified_status=profile["verified"],
        private_account=profile["privateAccount"],
        region=profile["region"],
        tiktok_seller=profile["ttSeller"],
        profile_signature=profile["signature"],
        num_followers=profile["fans"],
        num_following=profile["following"],
        num_likes=profile["heart"],
        num_videos=profile["video"],
        num_digg=profile["digg"],
        total_likes_over_num_followers=calculate_profile_engagement(
            profile["heart"], profile["fans"]
        ),
        total_likes_over_num_videos=calculate_profile_engagement(
            profile["heart"], profile["video"]
        ),
        video_transcripts=video_transcripts_combined,
    )
    # TODO need to refer to technical paper on dependent and indepepdent features?
    # TODO include construction of background-informed, feature building prompt

    # Construct system prompt
    profile["system_prompt"] = construct_system_prompt(
        profile, interview_type="polling"
    )

    # Construct user prompt
    profile["user_prompt"] = construct_user_prompt(profile, interview_type="polling")

    # Perform polling interview
    interview_response = ""  # TODO to be implemented

    # Preprocess post interview responses
    extracted_interview_responses = extract_llm_responses(
        interview_response, substring_exclusion_list=[]
    )
    profile_with_interview_responses = pd.concat(
        [profile, extracted_interview_responses], ignore_index=True
    )
    profile_with_interview_responses["poll_date"] = poll_date

    # Save the formatted polling interview responses
    past_polling_results = pd.read_csv(
        f"{base_dir}/../data/{project_name}/{polling_results_file}"
    )
    updated_polling_results = pd.concat(
        [past_polling_results, profile_with_interview_responses.to_frame().T],
        ignore_index=True,
    )
    updated_polling_results.to_csv(
        f"{base_dir}/../data/{project_name}/{polling_results_file}", index=False
    )

    return None


if __name__ == "__main__":
    poll_date = datetime.today().date()

    # Step 1: Get Pool
    print("Step 1: Get Pool")

    ## Perform key word search for TikTok videos discussing Canada elections
    print("Performing key word search...")
    perform_keyword_search(
        project_name=PROJECT,
        search_terms_file=SEARCH_TERMS_FILE,
        profile_metadata_file=KEYWORD_SEARCH_PROFILE_METADATA_FILE,
        video_metadata_file=KEYWORD_SEARCH_VIDEO_METADATA_FILE,
        perform_audio_transcription=True,
    )
    print()

    # ## Build user profile prompt
    # print("Building user profile prompt...")
    # build_profile_prompt(
    #     project_name=PROJECT,
    #     profile_metadata_input_file=KEYWORD_SEARCH_PROFILE_METADATA_FILE,
    #     profile_metadata_output_file=PROFILE_METADATA_POST_PROFILE_PROMPT_FILE,
    #     video_metadata_file=KEYWORD_SEARCH_VIDEO_METADATA_FILE,
    # )
    # print()

    # # Step 2: Poll Users
    # print("Step 2: Poll Users")

    # ## Apply temporal inclusion criteria (limit number of survey responses from a single user within a given timeframe)
    # print("Applying temporal inclusion criteria...")
    # apply_temporal_inclusion_criteria(
    #     project_name=PROJECT,
    #     profile_metadata_input_file=PROFILE_METADATA_POST_PROFILE_PROMPT_FILE,
    #     profile_metadata_output_file=PROFILE_METADATA_POST_TEMPORAL_INCLUSION_FILE,
    #     polled_profiles_file=POLLED_PROFILES_FILE,
    # )
    # print()

    # ## Apply null geography exclusion criteria (remove profiles without self-reported location information)
    # print("Applying null geography exclusion criteria...")
    # apply_null_geography_exclusion_criteria(
    #     project_name=PROJECT,
    #     profile_metadata_input_file=PROFILE_METADATA_POST_TEMPORAL_INCLUSION_FILE,
    #     profile_metadata_output_file=PROFILE_METADATA_POST_GEOGRAPHY_EXCLUSION_FILE,
    # )
    # print()

    # ## Apply entity inclusion criteria (exclude profiles that do not belong to an individual (i.e., organisations, bots, etc) and geographic inclusion criteria (filter out profiles that are unlikely to reside in Level 1 geography (i.e., Canada))
    # print("Applying entity inclusion criteria and geographic inclusion criteria...")
    # apply_entity_geographic_inclusion_criteria(
    #     project_name=PROJECT,
    #     profile_metadata_input_file=PROFILE_METADATA_POST_GEOGRAPHY_EXCLUSION_FILE,
    #     profile_metadata_output_file=PROFILE_METADATA_POST_ENTITY_GEOGRAPHIC_INCLUSION_FILE,
    # )
    # print()

    # ## Iterate through valid profile pool
    # print("Iterate through valid profile pool and store polling results...")
    # eligible_profile_pool = pd.read_csv(
    #     f"{base_dir}/../data/{PROJECT}/{PROFILE_METADATA_POST_ENTITY_GEOGRAPHIC_INCLUSION_FILE}"
    # )
    # polling_results = pd.read_csv(
    #     f"{base_dir}/../data/{PROJECT}/{POLLED_PROFILES_FILE}"
    # )
    # for i in tqdm(range(len(eligible_profile_pool))):
    #     ## Apply quota inclusion criteria
    #     eligible_profile = apply_quota_inclusion_criteria(
    #         profile=eligible_profile_pool.iloc[i]
    #     )

    #     if eligible_profile is None:  # Profile does not meet quota inclusion criteria
    #         continue

    #     ## Sample latest videos from eligible profiles
    #     profile_latest_videos = perform_profile_search(
    #         project_name=PROJECT,
    #         profile_metadata_file=PROFILE_SEARCH_PROFILE_METADATA_FILE,
    #         video_metadata_file=PROFILE_SEARCH_VIDEO_METADATA_FILE,
    #         profile_list=[eligible_profile],
    #         perform_audio_transcription=True,
    #         return_videos=True,
    #     )

    #     # Perform digital election polling on eligible profiles
    #     conduct_polling(
    #         project_name=PROJECT,
    #         profile=eligible_profile,
    #         profile_latest_videos=profile_latest_videos,
    #         polling_results_file=PROFILE_METADATA_POST_POLLING_FILE,
    #         poll_date=poll_date,
    #     )

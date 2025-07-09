import os
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm

tqdm.pandas()
from config.base_config import GPT_MODEL
from config.ai_election_polling_config import *
from src.utils import (
    perform_x_keyword_search,
    perform_x_profile_search,
    perform_x_profile_metadata_search,
    # build_profile_prompt,
    extract_llm_responses,
    perform_profile_interview,
    coalesce_columns_by_regex,
    # construct_system_prompt,
    # construct_user_prompt,
    # extract_video_transcripts,
    # calculate_profile_engagement,
)
from prompts.prompt_template import (
    x_entity_geographic_inclusion_system_prompt,
    x_entity_geographic_inclusion_user_prompt,
)

base_dir = os.path.dirname(os.path.abspath(__file__))


def apply_temporal_inclusion_criteria(
    project_name: str,
    execution_date: str,
    profile_metadata_file: str,
    output_file: str,
    polled_profiles_file: str,
) -> None:
    """
    Applies temporal inclusion criteria to filter profiles for polling based on recent polling history.

    This function reads profile metadata and a record of previously polled profiles, then excludes profiles
    that have been polled within a specified temporal inclusion period (defined by TEMPORAL_INCLUSION_PERIOD).
    The filtered profiles are saved to an output file for the current polling iteration, and the record of
    polled profiles is updated accordingly.

    Args:
        project_name (str): Name of the project, used to locate data directories.
        execution_date (str): Date of the current polling execution in the format "%d-%m-%Y".
        profile_metadata_file (str): Filename of the input profile metadata CSV.
        output_file (str): Filename for the output CSV containing profiles eligible for polling.
        polled_profiles_file (str): Filename for the CSV tracking all previously polled profiles.

    Returns:
        None
    """
    profile_metadata = pd.read_csv(
        os.path.join(
            base_dir, "../data", project_name, execution_date, profile_metadata_file
        )
    )

    polled_profiles_file_path = Path(
        os.path.join(base_dir, "../data", project_name, polled_profiles_file)
    )

    if polled_profiles_file_path.exists():
        polled_profiles = pd.read_csv(polled_profiles_file_path)
        polled_profiles["poll_date"] = pd.to_datetime(
            polled_profiles["poll_date"], format="%d-%m-%Y"
        )

        # Identify profiles that were polled within the last N days
        recently_polled_profiles = polled_profiles[
            polled_profiles["poll_date"]
            >= pd.to_datetime(execution_date, format="%d-%m-%Y")
            - timedelta(days=TEMPORAL_INCLUSION_PERIOD)
        ].reset_index(drop=True)

        # Exclude profiles that were polled within the last N days
        sampled_profile_metadata = profile_metadata[
            ~profile_metadata["account_id"].isin(recently_polled_profiles["account_id"])
        ]
        sampled_profile_metadata.to_csv(
            os.path.join(
                base_dir, "../data", project_name, execution_date, output_file
            ),
            index=False,
        )

        # Update polled profiles with profiles that will be polled in the current survey iteration
        newly_polled_profiles = sampled_profile_metadata[["account_id"]]
        newly_polled_profiles["poll_date"] = execution_date
        updated_polled_profiles = pd.concat(
            [recently_polled_profiles, newly_polled_profiles], ignore_index=True
        )
        updated_polled_profiles.to_csv(
            os.path.join(base_dir, "../data", project_name, polled_profiles_file),
            index=False,
        )

    else:  # If no profiles have been polled yet, all existing profiles will be polled
        sampled_profile_metadata = profile_metadata
        sampled_profile_metadata.to_csv(
            os.path.join(
                base_dir, "../data", project_name, execution_date, output_file
            ),
            index=False,
        )

        newly_polled_profiles = sampled_profile_metadata[["account_id"]]
        newly_polled_profiles["poll_date"] = execution_date
        newly_polled_profiles.to_csv(
            os.path.join(base_dir, "../data", project_name, polled_profiles_file),
            index=False,
        )


def apply_null_geography_exclusion_criteria(
    project_name: str,
    execution_date: str,
    profile_metadata_file: str,
    output_file: str,
) -> None:
    """
    Filters out profiles from a CSV file that do not have self-reported location information and saves the filtered data to a new CSV file.

    Args:
        project_name (str): Name of the project directory containing the data.
        execution_date (str): Date string specifying the execution folder.
        profile_metadata_file (str): Filename of the input CSV containing profile metadata.
        output_file (str): Filename for the output CSV to save filtered profiles.

    Returns:
        None
    """
    # Load profile metadata
    profile_metadata = pd.read_csv(
        os.path.join(
            base_dir, "../data", project_name, execution_date, profile_metadata_file
        )
    )

    # Exclude profiles without self-reported location information
    filtered_profile_metadata = profile_metadata[
        profile_metadata["location"].notnull()
    ].reset_index(drop=True)

    # Save profiles that meet the null geography exclusion criteria
    filtered_profile_metadata.to_csv(
        os.path.join(base_dir, "../data", project_name, execution_date, output_file),
        index=False,
    )


def apply_entity_geographic_inclusion_criteria(
    project_name: str,
    execution_date: str,
    country: str,
    profile_metadata_file: str,
    post_file: str,
    output_file: str,
) -> None:
    """
    Applies entity and geographic inclusion criteria to profile metadata for a given project and execution date.

    This function performs an interview process using a language model to assess whether social media profiles
    meet specified entity (individual vs. non-individual) and geographic (country-based) criteria. It processes
    the interview results, extracts relevant responses, merges columns as needed, and filters the profiles to
    retain only those that represent real-life individuals residing in the specified country. The filtered
    profiles are then saved to the specified output file.

    Args:
        project_name (str): Name of the project.
        execution_date (str): Date of execution (used for file path construction).
        country (str): Country to use for geographic inclusion criteria.
        profile_metadata_file (str): Path to the input profile metadata CSV file.
        post_file (str): Path to the post file used in the interview process.
        output_file (str): Name of the output CSV file to save filtered profiles.

    Returns:
        None
    """
    # Perform entity geographic inclusion criteria interview
    perform_profile_interview(
        project_name=project_name,
        execution_date=execution_date,
        gpt_model=GPT_MODEL,
        profile_metadata_file=profile_metadata_file,
        post_file=post_file,
        output_file=output_file,
        system_prompt_template=x_entity_geographic_inclusion_system_prompt.format(
            country=country
        ),
        user_prompt_template=x_entity_geographic_inclusion_user_prompt.format(
            country=country
        ),
        llm_response_field="entity_geographic_inclusion_llm_response",
        interview_type="x_ai_election_polling_entity_geographic_inclusion",
    )

    # Preprocess post interview results
    post_interview_profile_metadata = pd.read_csv(
        os.path.join(base_dir, "../data", project_name, execution_date, output_file)
    )
    extracted_responses = post_interview_profile_metadata[
        "entity_geographic_inclusion_llm_response"
    ].apply(extract_llm_responses)
    post_interview_profile_metadata = pd.concat(
        [post_interview_profile_metadata, extracted_responses], axis=1
    )

    # Merge identical columns from interview response
    post_interview_profile_metadata = coalesce_columns_by_regex(
        post_interview_profile_metadata, ENTITY_GEOGRAPHIC_INCLUSION_REGEX_PATTERNS
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
            f"Does the user of this X (formerly Twitter) account live in {country}? - category"
        ]
        == "Yes"
    ].reset_index(drop=True)

    # Save profiles that meet entity and geographic inclusion criteria
    filtered_profile_metadata.to_csv(
        os.path.join(base_dir, "../data", project_name, execution_date, output_file),
        index=False,
    )


# TODO to be updated
def apply_quota_inclusion_criteria(
    profile: pd.Series,
) -> pd.Series:
    return None


# TODO to be updated
def conduct_polling(
    project_name: str,
    execution_date: str,
    profile: pd.Series,
    profile_latest_videos: pd.DataFrame,
    polling_results_file: str,
    poll_date: datetime,
) -> None:
    # # Format past video transcripts
    # video_transcripts_combined = extract_video_transcripts(
    #     profile_id=profile["id"], video_metadata=profile_latest_videos
    # )

    # # Construct profile prompt
    # profile["profile_prompt"] = profile_prompt_template.format(
    #     profile_image=profile["avatar"],
    #     profile_name=profile["profile"],
    #     profile_nickname=profile["nickName"],
    #     verified_status=profile["verified"],
    #     private_account=profile["privateAccount"],
    #     region=profile["region"],
    #     tiktok_seller=profile["ttSeller"],
    #     profile_signature=profile["signature"],
    #     num_followers=profile["fans"],
    #     num_following=profile["following"],
    #     num_likes=profile["heart"],
    #     num_videos=profile["video"],
    #     num_digg=profile["digg"],
    #     total_likes_over_num_followers=calculate_profile_engagement(
    #         profile["heart"], profile["fans"]
    #     ),
    #     total_likes_over_num_videos=calculate_profile_engagement(
    #         profile["heart"], profile["video"]
    #     ),
    #     video_transcripts=video_transcripts_combined,
    # )
    # # TODO need to refer to technical paper on dependent and indepepdent features?
    # # TODO include construction of background-informed, feature building prompt

    # # Construct system prompt
    # profile["system_prompt"] = construct_system_prompt(
    #     profile, interview_type="polling"
    # )

    # # Construct user prompt
    # profile["user_prompt"] = construct_user_prompt(profile, interview_type="polling")

    # # Perform polling interview
    # interview_response = ""  # TODO to be implemented

    # # Preprocess post interview responses
    # extracted_interview_responses = extract_llm_responses(
    #     interview_response, substring_exclusion_list=[]
    # )
    # profile_with_interview_responses = pd.concat(
    #     [profile, extracted_interview_responses], ignore_index=True
    # )
    # profile_with_interview_responses["poll_date"] = poll_date

    # # Save the formatted polling interview responses
    # past_polling_results = pd.read_csv(
    #     f"{base_dir}/../data/{project_name}/{polling_results_file}"
    # )
    # updated_polling_results = pd.concat(
    #     [past_polling_results, profile_with_interview_responses.to_frame().T],
    #     ignore_index=True,
    # )
    # updated_polling_results.to_csv(
    #     f"{base_dir}/../data/{project_name}/{polling_results_file}", index=False
    # )

    return None


def define_pipeline_constants(country: str) -> dict:
    constants = {
        "pipeline_execution_date": PIPELINE_EXECUTION_DATE,
        "keyword_search_file": KEYWORD_SEARCH_FILE_X,
        "profile_metadata_search_file": PROFILE_METADATA_SEARCH_FILE_X,
        "temporal_inclusion_criteria_file": TEMPORAL_INCLUSION_CRITERIA_FILE_X,
        "polled_profiles": POLLED_PROFILES_FILE_X,
        "null_geography_exclusion_criteria_file": NULL_GEOGRAPHY_EXCLUSION_CRITERIA_FILE_X,
        "eligible_profile_pool_file": ELIGIBLE_PROFILE_POOL_FILE_X,
        "profile_search_file": PROFILE_SEARCH_FILE_X,
        "digital_polling_file": DIGITAL_POLLING_FILE_X,
    }

    if country.lower() == "chile":
        constants.update(
            {
                "project_name": PROJECT_NAME_X_CHILE,
                "search_term_list": SEARCH_TERMS_CHILE,
            }
        )
    elif country.lower() == "canada":
        constants.update(
            {
                "project_name": PROJECT_NAME_X_CANADA,
                "search_term_list": SEARCH_TERMS_CANADA,
            }
        )

    else:
        raise ValueError(
            f"Country {country} is currently not supported by the pipeline."
        )

    return constants


if __name__ == "__main__":
    # Load pipeline arguments (e.g., country information)
    parser = argparse.ArgumentParser(description="Run X election polling pipeline.")
    parser.add_argument(
        "--country", type=str, required=True, help="Country for the polling pipeline"
    )
    args = parser.parse_args()
    country = args.country

    # Step 0: Define pipeline constants based on country information
    constants = define_pipeline_constants(country=country)

    # Step 1: Get Pool
    print("Step 1: Get Pool")
    ## Perform key word search for X posts discussing the country's elections
    print("Perform keyword search using predefined list of search terms...")
    perform_x_keyword_search(
        project_name=constants["project_name"],
        execution_date=constants["pipeline_execution_date"],
        search_terms=constants["search_term_list"],
        output_file=constants["keyword_search_file"],
        num_post_per_keyword=NUM_POSTS_PER_KEYWORD,
    )

    ## Extract profile metadata for search results
    print("Perform profile metadata search for keyword search results...")
    perform_x_profile_metadata_search(
        project_name=constants["project_name"],
        execution_date=constants["pipeline_execution_date"],
        input_file=os.path.join(
            constants["pipeline_execution_date"], constants["keyword_search_file"]
        ),
        output_file=constants["profile_metadata_search_file"],
    )
    print()

    # Step 2: Identify Valid Users
    print("Step 2: Identify Valid Users")
    ## Apply temporal inclusion criteria (limit number of survey responses from a single user within a given timeframe)
    print("Applying temporal inclusion criteria...")
    apply_temporal_inclusion_criteria(
        project_name=constants["project_name"],
        execution_date=constants["pipeline_execution_date"],
        profile_metadata_file=constants["profile_metadata_search_file"],
        output_file=constants["temporal_inclusion_criteria_file"],
        polled_profiles_file=constants["polled_profiles"],
    )

    ## Apply null geography exclusion criteria (remove profiles without self-reported location information)
    print("Applying null geography exclusion criteria...")
    apply_null_geography_exclusion_criteria(
        project_name=constants["project_name"],
        execution_date=constants["pipeline_execution_date"],
        profile_metadata_file=constants["temporal_inclusion_criteria_file"],
        output_file=constants["null_geography_exclusion_criteria_file"],
    )

    ## Apply entity inclusion criteria (exclude profiles that do not belong to an individual (i.e., organisations, bots, etc) and geographic inclusion criteria (filter out profiles that are unlikely to reside in Level 1 geography (i.e., Canada))
    print("Applying entity inclusion criteria and geographic inclusion criteria...")
    apply_entity_geographic_inclusion_criteria(
        project_name=constants["project_name"],
        execution_date=constants["pipeline_execution_date"],
        country=country,
        profile_metadata_file=constants["null_geography_exclusion_criteria_file"],
        post_file=constants["keyword_search_file"],
        output_file=constants["entity_geographic_inclusion_criteria_file"],
    )
    print()

    # TODO CONTINUE FROM HERE
    # Step 3: Poll Valid Users
    print("Step 3: Poll Valid Users")
    ## Iterate through valid profile pool
    print("Iterate through valid profile pool and store polling results...")
    profile_pool = pd.read_csv(
        f"{base_dir}/../data/{constants['project_name']}/{constants['pipeline_execution_date']}/{constants['entity_geographic_inclusion_criteria_file']}"
    )
    polling_results = pd.read_csv(
        f"{base_dir}/../data/{constants['project_name']}/{constants['pipeline_execution_date']}/{constants['polled_profiles']}"
    )

    eligible_profile_pool = []
    for i in tqdm(range(len(profile_pool))):
        ## Apply quota inclusion criteria
        eligible_profile = apply_quota_inclusion_criteria(profile=profile_pool.iloc[i])

        if eligible_profile is None:  # Profile does not meet quota inclusion criteria
            continue

        else:
            eligible_profile_pool.append(eligible_profile)

    # TODO need to be updated to perform profile search and polling for the entire verified batch
    ## Sample latest videos from eligible profiles
    print("Extract posts from eligible profiles during polling period...")
    profile_latest_videos = perform_x_profile_search(
        project_name=constants["project_name"],
        execution_date=constants["pipeline_execution_date"],
        input_file_path=constants["eligible_profile_pool_file"],
        output_file_path=constants["profile_search_file"],
        start_date=PROFILE_SEARCH_START_DATE,
        end_date=PROFILE_SEARCH_END_DATE,
        num_posts_per_profile=NUM_POSTS_PER_PROFILE,
    )

    # Perform digital election polling on eligible profiles
    print("Perform digital election polling of eligible profiles...")
    conduct_polling(
        project_name=constants["project_name"],
        execution_date=constants["pipeline_execution_date"],
        profile=eligible_profile_pool,
        profile_latest_videos=profile_latest_videos,
        polling_results_file=constants["digital_polling_file"],
        poll_date=constants["pipeline_execution_date"],
    )

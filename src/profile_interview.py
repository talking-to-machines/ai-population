import os
import pandas as pd
from datetime import datetime
from config.config import (
    PROJECT,
    PROFILESEARCH_PROFILE_METADATA_FILE,
    PROFILESEARCH_VIDEO_METADATA_FILE,
    POST_IDENTIFICATION_FILE,
    PANEL_PROFILE_METADATA_FILE,
    POST_REFLECTION_FILE,
    POST_INTERVIEW_FILE,
    FORMATTED_POST_INTERVIEW_FILE,
    STOCK_RECOMMENDATION_FILE,
)
from src.utils import (
    extract_profile_id,
    extract_video_transcripts,
    construct_system_prompt,
    construct_user_prompt,
    create_batch_file,
    batch_query,
    extract_llm_responses,
    extract_stock_recommendations,
)

base_dir = os.path.dirname(os.path.abspath(__file__))


def perform_profile_interview(
    profile_metadata_file: str,
    video_metadata_file: str,
    output_file: str,
    system_prompt_field: str,
    user_prompt_field: str,
    llm_response_field: str,
    interview_type: str,
) -> None:

    # Load profile and video metadata
    print("Loading profile and video metadata...")
    profile_metadata = pd.read_csv(
        f"{base_dir}/../data/{PROJECT}/{profile_metadata_file}"
    )
    video_metadata = pd.read_csv(f"{base_dir}/../data/{PROJECT}/{video_metadata_file}")
    video_metadata["createTimeISO"] = pd.to_datetime(video_metadata["createTimeISO"])

    # Include interview date for digital interview
    if interview_type == "interview":
        profile_metadata["interview_date"] = datetime.today().date()

    # Preprocess profile and video metadata
    print("Preprocess profile and video metadata...")
    video_metadata["profile_id"] = video_metadata["authorMeta"].apply(
        extract_profile_id
    )
    video_metadata["profile_id"] = video_metadata["profile_id"].astype(str)
    profile_metadata["id"] = profile_metadata["id"].astype(str)

    # Generate system and user prompts
    print("Generate system and user prompts...")
    profile_metadata["transcripts_combined"] = profile_metadata["id"].apply(
        extract_video_transcripts, args=(video_metadata,)
    )

    profile_metadata[system_prompt_field] = profile_metadata.apply(
        construct_system_prompt, args=(interview_type,), axis=1
    )
    profile_metadata[user_prompt_field] = construct_user_prompt(interview_type)

    # Generate custom ids
    if "custom_id" not in profile_metadata.columns:
        profile_metadata = profile_metadata.reset_index(drop=False)
        profile_metadata.rename(columns={"index": "custom_id"}, inplace=True)

    # Create folder to contain batch files
    batch_file_dir = f"{base_dir}/../data/{PROJECT}/batch-files"
    os.makedirs(batch_file_dir, exist_ok=True)

    # Perform batch query for survey questions
    batch_file_dir = create_batch_file(
        profile_metadata,
        system_prompt_field=system_prompt_field,
        user_prompt_field=user_prompt_field,
        batch_file_name="batch_input.jsonl",
    )

    print("Perform batch query using OpenAI API...")
    llm_responses = batch_query(
        batch_input_file_dir="batch_input.jsonl",
        batch_output_file_dir="batch_output.jsonl",
    )
    llm_responses.rename(columns={"query_response": llm_response_field}, inplace=True)

    # Merge LLM response with original dataset
    print("Merge LLM response with original dataset...")
    profile_metadata["custom_id"] = profile_metadata["custom_id"].astype("int64")
    llm_responses["custom_id"] = llm_responses["custom_id"].astype("int64")
    profile_metadata_with_responses = pd.merge(
        left=profile_metadata,
        right=llm_responses[["custom_id", llm_response_field]],
        on="custom_id",
    )

    # Save profile metadata after analysis into CSV file
    print("Saving profile metadata with analysis...")
    profile_metadata_with_responses.to_csv(
        f"{base_dir}/../data/{PROJECT}/{output_file}", index=False
    )


def perform_finfluencer_identification() -> None:

    # Perform financial influencer identification interview
    perform_profile_interview(
        profile_metadata_file=PROFILESEARCH_PROFILE_METADATA_FILE,
        video_metadata_file=PROFILESEARCH_VIDEO_METADATA_FILE,
        output_file=POST_IDENTIFICATION_FILE,
        system_prompt_field="identification_system_prompt",
        user_prompt_field="identification_user_prompt",
        llm_response_field="identification_llm_response",
        interview_type="finfluencer_identification",
    )

    # Preprocess post identification results
    post_identification_results = pd.read_csv(
        f"{base_dir}/../data/{PROJECT}/{POST_IDENTIFICATION_FILE}"
    )
    extracted_responses = post_identification_results[
        "identification_llm_response"
    ].apply(extract_llm_responses)
    post_identification_results = pd.concat(
        [post_identification_results, extracted_responses], axis=1
    )

    # Filter out financial influencers that focuses on stock trading and equities, bonds and fixed income, or options trading and derivatives
    filtered_results = post_identification_results[
        (
            post_identification_results[
                "Which of these areas of finance are the primary focus of the influencer’s posts? - symbol"
            ].str.contains("B1|B2|B3", na=False)
        )
        & (
            post_identification_results[
                "Is this a finfluencer? - category"
            ].str.contains("Yes", na=False)
        )
    ]

    # Save identified financial influencers
    filtered_results.to_csv(
        f"{base_dir}/../data/{PROJECT}/{PANEL_PROFILE_METADATA_FILE}", index=False
    )

    return None


def generate_expert_reflections(
    role: str, profile_metadata_file: str, output_file: str
) -> None:

    if role == "portfolio_manager":
        system_prompt_field = "portfoliomanager_reflection_system_prompt"
        user_prompt_field = "portfoliomanager_reflection_user_prompt"
        llm_response_field = "expert_reflection_portfoliomanager"
        interview_type = "portfoliomanager_reflection"

    elif role == "investment_advisor":
        system_prompt_field = "investmentadvisor_reflection_system_prompt"
        user_prompt_field = "investmentadvisor_reflection_user_prompt"
        llm_response_field = "expert_reflection_investmentadvisor"
        interview_type = "investmentadvisor_reflection"

    elif role == "financial_analyst":
        system_prompt_field = "financialanalyst_reflection_system_prompt"
        user_prompt_field = "financialanalyst_reflection_user_prompt"
        llm_response_field = "expert_reflection_financialanalyst"
        interview_type = "financialanalyst_reflection"

    elif role == "economist":
        system_prompt_field = "economist_reflection_system_prompt"
        user_prompt_field = "economist_reflection_user_prompt"
        llm_response_field = "expert_reflection_economist"
        interview_type = "economist_reflection"

    else:
        raise ValueError(f"Role {role} is not supported.")

    perform_profile_interview(
        profile_metadata_file=profile_metadata_file,
        video_metadata_file=PROFILESEARCH_VIDEO_METADATA_FILE,
        output_file=output_file,
        system_prompt_field=system_prompt_field,
        user_prompt_field=user_prompt_field,
        llm_response_field=llm_response_field,
        interview_type=interview_type,
    )

    return None


def perform_digital_interview() -> None:

    perform_profile_interview(
        profile_metadata_file=POST_REFLECTION_FILE,
        video_metadata_file=PROFILESEARCH_VIDEO_METADATA_FILE,
        output_file=POST_INTERVIEW_FILE,
        system_prompt_field="digital_interview_system_prompt",
        user_prompt_field="digital_interview_user_prompt",
        llm_response_field="digital_interview_llm_response",
        interview_type="interview",
    )

    # Preprocess post interview results
    post_interview_results = pd.read_csv(
        f"{base_dir}/../data/{PROJECT}/{POST_INTERVIEW_FILE}"
    )
    extracted_responses = post_interview_results[
        "digital_interview_llm_response"
    ].apply(
        extract_llm_responses,
        args=(
            [
                "stock name",
                "If a list of stocks/stock tickers was provided in Question 9",
            ],
        ),
    )
    post_interview_results = pd.concat(
        [post_interview_results, extracted_responses], axis=1
    )

    # Extract stock recommendations
    combined_stock_recommendations = pd.DataFrame()
    for i in range(len(post_interview_results)):
        profile_stock_recommendations = extract_stock_recommendations(
            post_interview_results.iloc[i],
            llm_response_field="digital_interview_llm_response",
        )

        if profile_stock_recommendations is None:  # No stock recommendations
            continue

        profile_stock_recommendations["profile"] = post_interview_results.loc[
            i, "profile"
        ]
        profile_stock_recommendations["profile_url"] = post_interview_results.loc[
            i, "profileUrl"
        ]
        profile_stock_recommendations["followers"] = post_interview_results.loc[
            i, "fans"
        ]
        profile_stock_recommendations["influence"] = post_interview_results.loc[
            i,
            "Indicate on a scale of 0 to 100, how influential this influencer is – 0 means not at all influential and 100 means very influential with millions of followers and mainstream recognition? - value",
        ]
        profile_stock_recommendations["credibility"] = post_interview_results.loc[
            i,
            "Indicate on a scale of 0 to 100, how credible or authoritative this influencer is – 0 means not at all credible or authoritative and 100 means very credible and authoritative? - value",
        ]
        profile_stock_recommendations["interview_date"] = post_interview_results.loc[
            i, "interview_date"
        ]

        combined_stock_recommendations = pd.concat(
            [combined_stock_recommendations, profile_stock_recommendations],
            ignore_index=True,
        )

    # Save formatted interview results and stock recommendations
    post_interview_results.to_csv(
        f"{base_dir}/../data/{PROJECT}/{FORMATTED_POST_INTERVIEW_FILE}", index=False
    )
    combined_stock_recommendations.to_csv(
        f"{base_dir}/../data/{PROJECT}/{STOCK_RECOMMENDATION_FILE.format(interview_date=datetime.today().date())}",
        index=False,
    )

    return None


if __name__ == "__main__":
    perform_finfluencer_identification()
    generate_expert_reflections(
        role="portfolio_manager",
        profile_metadata_file=PANEL_PROFILE_METADATA_FILE,
        output_file=POST_REFLECTION_FILE,
    )
    generate_expert_reflections(
        role="investment_advisor",
        profile_metadata_file=POST_REFLECTION_FILE,
        output_file=POST_REFLECTION_FILE,
    )
    generate_expert_reflections(
        role="financial_analyst",
        profile_metadata_file=POST_REFLECTION_FILE,
        output_file=POST_REFLECTION_FILE,
    )
    generate_expert_reflections(
        role="economist",
        profile_metadata_file=POST_REFLECTION_FILE,
        output_file=POST_REFLECTION_FILE,
    )
    perform_digital_interview()

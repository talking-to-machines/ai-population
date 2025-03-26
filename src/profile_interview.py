import os
import pandas as pd
import re
import string
from tqdm import tqdm

tqdm.pandas()
from datetime import datetime
from config.market_signals_config import *
from src.utils import (
    extract_llm_responses,
    extract_stock_recommendations,
    perform_profile_interview,
)

base_dir = os.path.dirname(os.path.abspath(__file__))


def perform_finfluencer_identification() -> None:

    # Perform financial influencer identification interview
    perform_profile_interview(
        project_name=PROJECT,
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
        project_name=PROJECT,
        profile_metadata_file=profile_metadata_file,
        video_metadata_file=PROFILESEARCH_VIDEO_METADATA_FILE,
        output_file=output_file,
        system_prompt_field=system_prompt_field,
        user_prompt_field=user_prompt_field,
        llm_response_field=llm_response_field,
        interview_type=interview_type,
    )

    return None


def extract_stock_mentions_from_transcripts(row: pd.Series, russell_4000_stock) -> str:
    # Split the transcripts by double newline
    transcript_chunks = row["transcripts_combined"].strip().split("\n\n")

    # Prepare a list for storing the matched results
    found_mentions = []

    for chunk in transcript_chunks:
        # Initialize variables for creation date and transcript text
        creation_date = "Unknown"
        transcript_text = ""

        # Extract creation date using a regular expression
        creation_date_match = re.search(r"Creation Date:\s*(.+)", chunk)
        if creation_date_match:
            creation_date = creation_date_match.group(1).strip()

        # Extract video transcript using a regular expression
        transcript_match = re.search(r"Video Transcript:\s*(.+)", chunk)
        if transcript_match:
            transcript_text = transcript_match.group(1).strip()

        # Skip processing if no transcript text is found
        if not transcript_text:
            continue

        # Check each stock in the Russell 4000
        for _, row in russell_4000_stock.iterrows():
            full_stock_name = row["COMNAM"].strip()
            shorted_stock_name = row["SHORTEN_COMNAM"].strip()
            stock_ticker = row["TICKER"].strip()

            # Check if stock name is found in transcript chunk
            transcript_text = transcript_text.translate(
                str.maketrans(string.punctuation, " " * len(string.punctuation))
            )  # Remove all punctuation from transcript_text
            name_match = (
                re.search(
                    rf"\b{re.escape(shorted_stock_name.lower())}\b",
                    transcript_text.lower(),
                )
                is not None
            )

            if name_match:
                found_mentions.append(
                    {
                        "stock_name": full_stock_name,
                        "stock_ticker": stock_ticker,
                        "video_creation_date": creation_date,
                    }
                )

    # Build a DataFrame from the matches
    stock_mentions_df = pd.DataFrame(
        found_mentions, columns=["stock_name", "stock_ticker", "video_creation_date"]
    )

    # Remove duplicates if you only want unique (stock, date) pairs
    stock_mentions_df = stock_mentions_df.drop_duplicates().reset_index(drop=True)

    # Create a formatted text string containing all the stocks mentioned and the questions for each stock
    stock_mentions_formatted_str = ""
    stock_question_template = """**stock name: {stock_name}**
**stock ticker: {stock_ticker}**
**mention date: {video_creation_date}**"""

    for i in range(len(stock_mentions_df)):
        if i != 0:
            stock_mentions_formatted_str += "\n\n"
        stock_mentions_formatted_str += stock_question_template.format(
            stock_name=stock_mentions_df.loc[i, "stock_name"],
            stock_ticker=stock_mentions_df.loc[i, "stock_ticker"],
            video_creation_date=stock_mentions_df.loc[i, "video_creation_date"],
        )

    return stock_mentions_formatted_str


def extract_stock_mentions(input_file: str, output_file: str) -> None:
    # Load post reflection results
    post_reflection_results = pd.read_csv(f"{base_dir}/../data/{PROJECT}/{input_file}")

    # Extract stocks mention in past videos
    russell_4000_stock = pd.read_csv(
        f"{base_dir}/../config/{RUSSELL_4000_STOCK_TICKER_FILE}"
    )
    post_reflection_results["stock_mentions"] = post_reflection_results.progress_apply(
        extract_stock_mentions_from_transcripts, args=(russell_4000_stock,), axis=1
    )

    # Save formatted post reflection results
    post_reflection_results.to_csv(
        f"{base_dir}/../data/{PROJECT}/{output_file}", index=False
    )

    return None


def perform_digital_interview() -> None:

    perform_profile_interview(
        project_name=PROJECT,
        profile_metadata_file=POST_STOCK_EXTRACTION_FILE,
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
                "A list of Russell 4000 stocks was extracted from your past video transcripts",
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

        combined_stock_recommendations = pd.concat(
            [combined_stock_recommendations, profile_stock_recommendations],
            ignore_index=True,
        )

    # Remove duplicated entries and stocks that were not mentioned in the video transcripts
    valid_stock_recommendations = (
        combined_stock_recommendations.drop_duplicates().reset_index(drop=True)
    )

    # Sort by profile and mention date (descending order within each profile)
    valid_stock_recommendations["mention date"] = pd.to_datetime(
        valid_stock_recommendations["mention date"]
    )
    valid_stock_recommendations = valid_stock_recommendations.sort_values(
        by=["profile", "mention date"], ascending=[True, False]
    ).reset_index(drop=True)

    # Remove stocks that are not mentioned by the influencer
    valid_stock_recommendations = valid_stock_recommendations[
        combined_stock_recommendations["mentioned by influencer"] == "Yes"
    ].reset_index(drop=True)

    # Save formatted interview results and stock recommendations
    post_interview_results.to_csv(
        f"{base_dir}/../data/{PROJECT}/{FORMATTED_POST_INTERVIEW_FILE}", index=False
    )
    valid_stock_recommendations.to_csv(
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
    extract_stock_mentions(
        input_file=POST_REFLECTION_FILE,
        output_file=POST_STOCK_EXTRACTION_FILE,
    )
    perform_digital_interview()

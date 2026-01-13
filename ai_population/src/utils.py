import os, ast, yt_dlp, time, json, re, requests, warnings
import pandas as pd

pd.set_option("future.no_silent_downcasting", True)
from requests.auth import HTTPBasicAuth
from datetime import datetime, timezone
from tqdm import tqdm
from tqdm.auto import tqdm as tqdm_auto

tqdm.pandas()

from pydub import AudioSegment
from apify_client import ApifyClient
from openai import OpenAI, APITimeoutError
from concurrent.futures import ThreadPoolExecutor
from ai_population.prompts.prompt_template import (
    tiktok_video_prompt_template,
    x_tweet_prompt_template,
    tiktok_profile_prompt_template,
)
from ai_population.config.base_config import *
from ai_population.config.market_signals_config import (
    RUSSELL_4000_STOCK_TICKER_FILE,
)

openai_client = OpenAI(api_key=OPENAI_API_KEY)
base_dir = os.path.dirname(os.path.abspath(__file__))


def load_text_file(file_path) -> list:
    """
    Load search terms for market signals or profile list from text file.

    Args:
        file_path (str): The path to the text file containing search terms/profiles, one per line.

    Returns:
        list: A list of search terms/profiles as strings.
    """
    full_file_path = f"{base_dir}/../config/{file_path}"
    with open(full_file_path, "r") as file:
        return [line.strip() for line in file]


def update_video_metadata(
    project_name: str,
    video_metadata_file: str,
    client: ApifyClient,
    run: dict,
    profile_search: bool,
    filtering_list: list,
) -> None:
    """
    Updates the video metadata by fetching new data, appending it to the existing data,
    and removing duplicates.

    Args:
        client (ApifyClient): The Apify client used to fetch video metadata.
        run (dict): The run object containing the default dataset ID.
        profile_search (bool): A boolean indicating whether the search was for profiles or not.
        filtering_list (list): A list of search terms or profiles used to filter the search results.
    """
    # Fetch extracted video metadata
    video_metadata = pd.DataFrame(
        list(client.dataset(run["defaultDatasetId"]).iterate_items())
    )

    # Filter out videos based on search terms or profiles to remove irrelevant entries
    if profile_search:
        video_metadata.rename(columns={"input": "profile"}, inplace=True)
        video_metadata = video_metadata[
            video_metadata["profile"].isin(filtering_list)
        ].reset_index(drop=True)
    else:  # keyword search
        video_metadata = video_metadata[
            video_metadata["searchQuery"].isin(filtering_list)
        ].reset_index(drop=True)

    # Append extraction time to extracted video metadata
    video_metadata["extractionTime"] = pd.Timestamp.utcnow()

    # Extract profile id information
    video_metadata["profile_id"] = video_metadata["authorMeta"].apply(
        lambda x: x.get("id", None) if isinstance(x, dict) else None
    )

    # Define the file path
    video_metadata_path = f"{base_dir}/../data/{project_name}/{video_metadata_file}"

    if os.path.exists(video_metadata_path):
        # Load existing video metadata file
        old_video_metadata = pd.read_csv(video_metadata_path)
        old_video_metadata["id"] = old_video_metadata["id"].astype("str")

        # Append new data
        video_metadata = pd.concat([old_video_metadata, video_metadata])

    # Remove duplicated video entries based on video ID, keeping the latest entry
    video_metadata.drop_duplicates(
        subset="id",
        keep="last",
        inplace=True,
    )

    # Save updated video metadata
    video_metadata.to_csv(video_metadata_path, index=False)

    return None


def convert_str_to_dictionary(str_to_convert: str) -> dict:
    """
    Converts a string representation of a dictionary to an actual dictionary.

    Args:
        str_to_convert (str): The string to convert to a dictionary.

    Returns:
        dict: The converted dictionary. If conversion fails, returns a dictionary with a single key 'id' set to None.
    """
    try:
        return ast.literal_eval(str_to_convert)
    except Exception as e:
        return {"id": None}


def update_profile_metadata(
    project_name: str, profile_metadata_file: str, video_metadata_file: str
) -> None:
    """
    Updates the profile metadata for a given project by processing the video metadata.

    Args:
        profile_search (bool): A boolean indicating whether the search was for profiles or not.
    """
    # Load video metadata file
    video_metadata_path = f"{base_dir}/../data/{project_name}/{video_metadata_file}"
    video_metadata = pd.read_csv(video_metadata_path)

    # Extract the authorMeta field
    profile_metadata = video_metadata[["authorMeta", "extractionTime"]]

    # Convert the authorMeta dictionary to separate columns
    profile_metadata.loc[:, "authorMeta"] = profile_metadata["authorMeta"].apply(
        convert_str_to_dictionary
    )
    profile_metadata = pd.json_normalize(profile_metadata["authorMeta"]).join(
        profile_metadata["extractionTime"]
    )
    profile_metadata.rename(columns={"name": "profile"}, inplace=True)
    profile_metadata["id"] = profile_metadata["id"].astype("str")

    # Remove duplicates based on profile ID, keeping the latest entry
    profile_metadata.drop_duplicates(
        subset="id",
        keep="last",
        inplace=True,
    )

    # Drop invalid profiles
    profile_metadata = profile_metadata[
        (~profile_metadata["id"].isin(["nan", "None"]))
        & (~profile_metadata["id"].isnull())
    ].reset_index(drop=True)

    # Save profile metadata locally, overwrite existing profile metadata if it exist
    profile_metadata_path = f"{base_dir}/../data/{project_name}/{profile_metadata_file}"
    profile_metadata.to_csv(profile_metadata_path, index=False)

    return None


def identify_top_influencers(
    top_n_profiles: int, project_name: str, profile_metadata_file: str
) -> None:
    """
    Identifies the top N influencers based on the number of followers from a profile metadata file
    and saves their profiles to a text file.

    Args:
        top_n_profiles (int): The number of top profiles to identify based on the number of followers.

    Returns:
        None
    """
    # Load profile metadata file based on keyword search
    profile_metadata_path = f"{base_dir}/../data/{project_name}/{profile_metadata_file}"
    profile_metadata = pd.read_csv(profile_metadata_path)

    # Sort profiles based on number of followers
    profile_metadata_sorted = profile_metadata.sort_values(
        by="fans", ascending=False
    ).reset_index(drop=True)

    # Identify top n profiles based on number of followers
    profile_metadata_top_n_profiles = profile_metadata_sorted.head(top_n_profiles)

    # Save top n profiles to a text file
    profiles = profile_metadata_top_n_profiles["profile"].tolist()
    profiles_path = f"{base_dir}/../config/{project_name}_profiles.txt"

    with open(profiles_path, "w") as file:
        for profile in profiles:
            file.write(f"{profile}\n")

    return None


def download_video(row: pd.Series, project_name: str, execution_date: str) -> None:
    """
    Downloads a TikTok video using the provided information in the row.

    Args:
        row (pd.Series): A pandas Series containing the video information, including the 'webVideoUrl' and 'video_filename'.
        project_name (str): The project name used to construct the output file path.
        execution_date (str): The execution date used to construct the output file path.

    Returns:
        None
    """
    # The TikTok video link
    video_url = row["url"]

    # Output file name
    output_file = f"{base_dir}/../data/{project_name}/{execution_date}/video-downloads/{row['video_filename']}"

    # Skip if the video is already downloaded
    if os.path.exists(output_file):
        return None

    # Options for yt-dlp
    ydl_opts = {
        "outtmpl": output_file,  # Save the video with this file name
        "format": "best",  # Download the best quality available
    }

    # Download the video
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
    except Exception as e:
        print(f"An error occurred downloading {video_url}:", str(e))


def optimize_audio_file(input_file: str, output_file: str) -> None:
    """
    Optimize an audio file by downsampling it to 16 kHz and converting it to mono.

    Args:
        input_file (str): The path to the input audio file.
        output_file (str): The path where the optimized audio file will be saved.

    Returns:
        None
    """
    # Load the audio file
    audio = AudioSegment.from_file(input_file)

    # Downsample the audio to 16 kHz and convert to mono
    audio = audio.set_frame_rate(16000).set_channels(1)

    # Export the optimized audio file
    audio.export(output_file, format="wav")


def transcribe_videos(row: pd.Series, project_name: str, execution_date: str) -> str:
    """
    Transcribes the audio from a video file using the OpenAI Whisper model.
    Args:
        row (pd.Series): A pandas Series containing information about the video file.
                         It must include a 'video_filename' key with the name of the video file.
        project_name (str): The name of the project, used to construct the file paths.
        execution_date (str): The execution date, used to construct the file paths.
    Returns:
        str: The transcription of the audio if successful, otherwise None.
    Raises:
        FileNotFoundError: If the input video file is not found.
        Exception: For other errors encountered during transcription, including file size issues.
    """
    input_file_path = f"{base_dir}/../data/{project_name}/{execution_date}/video-downloads/{row['video_filename']}"
    optimized_file_path = f"{base_dir}/../data/{project_name}/{execution_date}/video-downloads/optimized_{row['video_filename'][:-4] + '.wav'}"

    max_retries = 3
    retry_delay = 60  # seconds

    for attempt in range(1, max_retries + 1):
        try:
            with open(input_file_path, "rb") as audio_file:
                transcription = openai_client.audio.transcriptions.create(
                    model="whisper-1", file=audio_file, response_format="text"
                )
            return transcription

        except FileNotFoundError:
            return None

        except APITimeoutError:
            print(
                f"Timeout error during transcription ({attempt}/{max_retries}). Retrying in {retry_delay}s..."
            )
            time.sleep(retry_delay)
            continue

        except Exception as e:
            status = getattr(e, "status", None) or getattr(e, "status_code", None)
            if status == 413:
                print(
                    f"Error: File {row['video_filename']} is too large to process. Optimizing the audio file..."
                )
                # Optimize the audio file
                try:
                    optimize_audio_file(input_file_path, optimized_file_path)
                    with open(optimized_file_path, "rb") as audio_file:
                        transcription = openai_client.audio.transcriptions.create(
                            model="whisper-1", file=audio_file, response_format="text"
                        )
                    return transcription
                except Exception as e2:
                    print(
                        f"Error: File {optimized_file_path} is still too large after optimisation: {e}"
                    )
                    return None
            else:
                print(
                    f"Error encountered when transcribing {row['video_filename']}: {e}"
                )
                return None

    print(f"Failed to transcribe {row['video_filename']} after {max_retries} attempts.")
    return None


def calculate_profile_engagement(num_likes: str, num_fans_videos: str) -> float:
    """
    Calculate the profile engagement based on the number of likes and the number of fans/videos posted.

    Args:
        num_likes (str): The number of likes as a string.
        num_fans_videos (str): The number of fans/videos posted.

    Returns:
        float: The profile engagement ratio. If the number of fans/videos posted is zero or cannot be converted to a number, returns 0.0.
    """
    num_likes = pd.to_numeric(num_likes, errors="coerce")
    num_fans_videos = pd.to_numeric(num_fans_videos, errors="coerce")

    # Replace NaN values with 0
    num_likes = num_likes if pd.notna(num_likes) else 0
    num_fans_videos = num_fans_videos if pd.notna(num_fans_videos) else 0

    profile_engagement = num_likes / num_fans_videos if num_fans_videos > 0 else 0.0
    return profile_engagement


def construct_system_prompt(
    row: pd.Series, system_prompt_template: str, interview_type: str
) -> str:
    if interview_type.startswith("tiktok"):
        profile_args = {
            "profile_image": row.get("profile_pic_url", ""),
            "profile_name": row.get("account_id", ""),
            "profile_nickname": row.get("nickname", ""),
            "profile_biography": row.get("biography", ""),
            "profile_signature": row.get("signature", ""),
            "profile_bio_link": row.get("bio_link", ""),
            "profile_url": row.get("url", ""),
            "profile_lang": row.get("predicted_lang", ""),
            "profile_creation": row.get("create_time", ""),
            "verified_status": row.get("is_verified", ""),
            "num_followers": row.get("followers", ""),
            "num_following": row.get("following", ""),
            "num_likes": row.get("likes", ""),
            "num_videos": row.get("videos_count", ""),
            "num_digg": row.get("digg_count", ""),
            "private_account": row.get("is_private", ""),
            "region": row.get("region", ""),
            "tiktok_seller": row.get("is_commerce_user", ""),
            "awg_engagement_rate": row.get("awg_engagement_rate", ""),
            "comment_engagement_rate": row.get("comment_engagement_rate", ""),
            "like_engagement_rate": row.get("like_engagement_rate", ""),
            "video_transcripts": row.get("posts_combined", ""),
        }
    elif interview_type.startswith("x"):
        profile_args = {
            "profile_picture": row.get("profilePicture", ""),
            "name": row.get("name", ""),
            "account_id": row.get("account_id", ""),
            "location": row.get("location", ""),
            "description": row.get("description", ""),
            "url": row.get("url", ""),
            "created_at": row.get("createdAt", ""),
            "is_verified": row.get("isVerified", ""),
            "is_blue_verified": row.get("isBlueVerified", ""),
            "protected": row.get("protected", ""),
            "followers": row.get("followers", ""),
            "following": row.get("following", ""),
            "statuses_count": row.get("statusesCount", ""),
            "favourites_count": row.get("favouritesCount", ""),
            "media_count": row.get("mediaCount", ""),
            "tweets": row.get("posts_combined", ""),
        }

    else:
        profile_args = {}

    if interview_type in [
        "tiktok_finfluencer_onboarding",
    ]:
        additional_args = {
            "expert_reflection_investmentadvisor": row[
                "tiktok_finfluencer_expert_reflection_investmentadvisor_response"
            ],
        }
        profile_args.update(additional_args)
    elif interview_type in [
        "x_finfluencer_onboarding",
    ]:
        additional_args = {
            "expert_reflection_investmentadvisor": row[
                "x_finfluencer_expert_reflection_investmentadvisor_response"
            ],
        }
        profile_args.update(additional_args)
    else:
        pass

    return system_prompt_template.format(**profile_args)


def construct_user_prompt(
    row: pd.Series, user_prompt_template: str, interview_type: str
) -> str:
    if interview_type in [
        "tiktok_finfluencer_daily_stock_pick",
        "x_finfluencer_daily_stock_pick",
    ]:
        # Load Russell 4000 stock tickers
        russell4000_stock_tickers = pd.read_csv(
            os.path.join(base_dir, "../config", RUSSELL_4000_STOCK_TICKER_FILE)
        )

        # Construct Russell 4000 stock ticker string
        russell4000_stock_tickers["combined_ticker"] = russell4000_stock_tickers.apply(
            lambda stock_row: f"{stock_row['COMNAM']} ({stock_row['TICKER']})", axis=1
        )
        russell4000_stock_ticker_list = russell4000_stock_tickers[
            "combined_ticker"
        ].to_list()
        russell4000_stock_ticker_str = ", ".join(russell4000_stock_ticker_list)

        # Construct user prompt
        return user_prompt_template.format(
            russell_4000_tickers=russell4000_stock_ticker_str,
        )

    if interview_type in [
        "tiktok_finfluencer_stock_recommendation",
        "x_finfluencer_stock_recommendation",
    ]:
        # Load stock mentioned and reference to post
        return user_prompt_template.format(
            stock_name=row.get("stock_name", ""),
            stock_ticker=row.get("stock_ticker", ""),
            mention_date=row.get("mention_date", ""),
            post=row.get("post", ""),
        )

    if interview_type == "x_digital_twin_voting_preference":
        municipality = (
            "NA" if pd.isna(row["COMUNA - category"]) else row["COMUNA - category"]
        )
        return user_prompt_template.format(municipality=municipality)

    return user_prompt_template


def extract_llm_responses(text, substring_exclusion_list: list = []) -> pd.Series:
    # Split the text by double newlines to separate different questions
    if (
        text is None
        or (isinstance(text, float) and pd.isna(text))
        or str(text).strip() == ""
    ):
        return pd.Series(dtype=object)

    questions_blocks = re.split(r"(?=\*\*question:)", text)
    questions_blocks = [
        block
        for block in questions_blocks
        if block
        and not any(substring in block for substring in substring_exclusion_list)
    ]  # remove blocks containing stock recommendations

    # Initialize lists to store the extracted data
    questions_list = []
    explanations_list = []
    symbols_list = []
    categories_list = []
    speculations_list = []
    values_list = []
    response_list = []
    stock_ticker_list = []
    recommendation_list = []
    confidence_list = []
    expected_holding_period_list = []
    primary_catalyst_type_list = []

    # Define regex patterns for each field
    question_pattern = r"\*\*question: (.*?)\*\*"
    explanation_pattern = r"\*\*explanation: (.*?)\*\*"
    symbol_pattern = r"\*\*symbol: (.*?)\*\*"
    category_pattern = r"\*\*category: (.*?)\*\*"
    speculation_pattern = r"\*\*speculation: (.*?)\*\*"
    value_pattern = r"\*\*value: (.*?)\*\*"
    response_pattern = r"\*\*response: (.*?)\*\*"
    stock_ticker_pattern = r"\*\*stock ticker: (.*?)\*\*"
    recommendation_pattern = r"\*\*recommendation: (.*?)\*\*"
    confidence_pattern = r"\*\*confidence: (.*?)\*\*"
    expected_holding_period_pattern = r"\*\*expected holding period: (.*?)\*\*"
    primary_catalyst_type_pattern = r"\*\*primary catalyst type: (.*?)\*\*"

    # Iterate through each question block and extract the fields
    for block in questions_blocks:
        if pd.isnull(block) or not block:
            continue
        question = re.search(question_pattern, block, re.DOTALL)
        explanation = re.search(explanation_pattern, block, re.DOTALL)
        symbol = re.search(symbol_pattern, block, re.DOTALL)
        category = re.search(category_pattern, block, re.DOTALL)
        speculation = re.search(speculation_pattern, block, re.DOTALL)
        value = re.search(value_pattern, block, re.DOTALL)
        response = re.search(response_pattern, block, re.DOTALL)
        stock_ticker = re.search(stock_ticker_pattern, block, re.DOTALL)
        recommendation = re.search(recommendation_pattern, block, re.DOTALL)
        confidence = re.search(confidence_pattern, block, re.DOTALL)
        expected_holding_period = re.search(
            expected_holding_period_pattern, block, re.DOTALL
        )
        primary_catalyst_type = re.search(
            primary_catalyst_type_pattern, block, re.DOTALL
        )

        questions_list.append(question.group(1).replace("”", "") if question else None)
        explanations_list.append(explanation.group(1) if explanation else None)
        symbols_list.append(symbol.group(1) if symbol else None)
        categories_list.append(category.group(1) if category else None)
        speculations_list.append(speculation.group(1) if speculation else None)
        values_list.append(value.group(1) if value else None)
        response_list.append(response.group(1) if response else None)
        stock_ticker_list.append(stock_ticker.group(1) if stock_ticker else None)
        recommendation_list.append(recommendation.group(1) if recommendation else None)
        confidence_list.append(confidence.group(1) if confidence else None)
        expected_holding_period_list.append(
            expected_holding_period.group(1) if expected_holding_period else None
        )
        primary_catalyst_type_list.append(
            primary_catalyst_type.group(1) if primary_catalyst_type else None
        )

    # Create a DataFrame
    data = {
        "question": questions_list,
        "explanation": explanations_list,
        "symbol": symbols_list,
        "category": categories_list,
        "speculation": speculations_list,
        "value": values_list,
        "response": response_list,
        "stock_ticker": stock_ticker_list,
        "recommendation": recommendation_list,
        "confidence": confidence_list,
        "expected_holding_period": expected_holding_period_list,
        "primary_catalyst_type": primary_catalyst_type_list,
    }
    df = pd.DataFrame(data)

    # Flatten the DataFrame into a single Series
    flattened_series = pd.Series()
    for _, row in df.iterrows():
        question_prefix = row["question"]
        if row["explanation"]:
            flattened_series[f"{question_prefix} - explanation"] = row["explanation"]
        if row["symbol"]:
            flattened_series[f"{question_prefix} - symbol"] = row["symbol"]
        if row["category"]:
            flattened_series[f"{question_prefix} - category"] = row["category"]
        if row["speculation"]:
            flattened_series[f"{question_prefix} - speculation"] = row["speculation"]
        if row["value"]:
            flattened_series[f"{question_prefix} - value"] = row["value"]
        if row["response"]:
            flattened_series[f"{question_prefix} - response"] = row["response"]
        if row["stock_ticker"]:
            flattened_series[f"{question_prefix} - stock ticker"] = row["stock_ticker"]
        if row["recommendation"]:
            flattened_series[f"{question_prefix} - recommendation"] = row[
                "recommendation"
            ]
        if row["confidence"]:
            flattened_series[f"{question_prefix} - confidence"] = row["confidence"]
        if row["expected_holding_period"]:
            flattened_series[f"{question_prefix} - expected holding period"] = row[
                "expected_holding_period"
            ]
        if row["primary_catalyst_type"]:
            flattened_series[f"{question_prefix} - primary catalyst type"] = row[
                "primary_catalyst_type"
            ]

    return flattened_series


def coalesce_columns_by_regex(data: pd.DataFrame, regex_list: list) -> pd.DataFrame:
    """
    Coalesces columns in a DataFrame that match any of the provided regex patterns.
    For each regex pattern in `regex_list`, finds all columns whose names match the pattern (case-insensitive).
    Among the matching columns, retains the one with the fewest missing values, and fills its missing values
    using the next best matching columns (row-wise, using backfill). All other matching columns are dropped.

    Parameters:
        data (pd.DataFrame): The input DataFrame whose columns are to be coalesced.
        regex_list (list): A list of regex patterns (strings) to match column names.

    Returns:
        pd.DataFrame: The DataFrame with coalesced columns, where for each pattern only one column remains,
        containing the most complete set of values from the original matching columns.
    """
    for pattern in regex_list:
        compiled_pattern = re.compile(pattern, flags=re.IGNORECASE)
        matching_cols = [col for col in data.columns if compiled_pattern.search(col)]
        matching_cols = list(matching_cols)
        if not matching_cols:
            continue

        # Sort matching columns by null count (fewest nulls first)
        sorted_cols = sorted(
            matching_cols, key=lambda col: data[col].isna().sum().sum()
        )

        # Fill in missing values in the best column using bfill along row-wise for sorted matching columns
        retained_col = sorted_cols[0]
        data[retained_col] = data[sorted_cols].bfill(axis=1).iloc[:, 0]

        # Drop all other matching columns
        cols_to_drop = sorted_cols[1:]
        data = data.drop(columns=cols_to_drop)
    return data


def format_stock_mentions(stock_mentions_str: str) -> pd.DataFrame:
    """
    Parses a formatted string containing multiple stock mentions and extracts structured information into a pandas DataFrame.

    Each stock mention in the input string should follow the format:
        **stock name: <name>**
        **stock ticker: <ticker>**
        **mention date: <date>**
        **post: <post content>**

    Args:
        stock_mentions_str (str): A string containing one or more stock mention blocks, each starting with '**stock name:'.

    Returns:
        pd.DataFrame: A DataFrame with columns ['stock_name', 'stock_ticker', 'mention_date', 'post'], where each row corresponds to a stock mention extracted from the input string.
    """
    # Split the stock mention string starting with "**stock name:" to separate different stock mentions
    stock_mention_list = re.split(r"(?=\*\*stock name:)", stock_mentions_str)

    # Initialize lists to store the extracted data
    stock_name_list = []
    stock_ticker_list = []
    mention_date_list = []
    post_list = []

    # Define regex patterns for each field
    stock_name_pattern = r"\*\*stock name: (.*?)\*\*"
    stock_ticker_pattern = r"\*\*stock ticker: (.*?)\*\*"
    mention_date_pattern = r"\*\*mention date: (.*?)\*\*"
    post_pattern = r"\*\*post: (.*?)\*\*"

    # Iterate through each question block and extract the fields
    for stock_mention_block in stock_mention_list:
        if pd.isnull(stock_mention_block) or not stock_mention_block:
            continue
        stock_name = re.search(stock_name_pattern, stock_mention_block, re.DOTALL)
        stock_ticker = re.search(stock_ticker_pattern, stock_mention_block, re.DOTALL)
        mention_date = re.search(mention_date_pattern, stock_mention_block, re.DOTALL)
        post = re.search(post_pattern, stock_mention_block, re.DOTALL)

        stock_name_list.append(stock_name.group(1) if stock_name else None)
        stock_ticker_list.append(stock_ticker.group(1) if stock_ticker else None)
        mention_date_list.append(mention_date.group(1) if mention_date else None)
        post_list.append(post.group(1) if post else None)

    # Create a DataFrame
    data = {
        "stock_name": stock_name_list,
        "stock_ticker": stock_ticker_list,
        "mention_date": mention_date_list,
        "post": post_list,
    }
    stock_mention_df = pd.DataFrame(data)

    return stock_mention_df


def format_stock_recommendations(stock_recommendation_str: str) -> pd.Series:
    # Define regex patterns for each field
    mentioned_by_finfluencer_pattern = r"\*\*mentioned_by_finfluencer: (.*?)\*\*"
    recommendation_pattern = r"\*\*recommendation: (.*?)\*\*"
    explanation_pattern = r"\*\*explanation: (.*?)\*\*"
    confidence_pattern = r"\*\*confidence: (.*?)\*\*"
    virality_pattern = r"\*\*virality: (.*?)\*\*"
    risks_pattern = r"\*\*risks: (.*?)\*\*"
    horizon_pattern = r"\*\*horizon: (.*?)\*\*"
    conflicts_pattern = r"\*\*conflicts: (.*?)\*\*"

    # Extract the relevant fields from the stock recommendation string
    if pd.isna(stock_recommendation_str) or str(stock_recommendation_str).strip() == "":
        return pd.Series(
            {
                "mentioned_by_finfluencer": None,
                "recommendation": None,
                "explanation": None,
                "confidence": None,
                "virality": None,
                "risks": None,
                "horizon": None,
                "conflicts": None,
            }
        )

    mentioned_by_finfluencer = re.search(
        mentioned_by_finfluencer_pattern, stock_recommendation_str, re.DOTALL
    )
    recommendation = re.search(
        recommendation_pattern, stock_recommendation_str, re.DOTALL
    )
    explanation = re.search(explanation_pattern, stock_recommendation_str, re.DOTALL)
    confidence = re.search(confidence_pattern, stock_recommendation_str, re.DOTALL)
    virality = re.search(virality_pattern, stock_recommendation_str, re.DOTALL)
    risks = re.search(risks_pattern, stock_recommendation_str, re.DOTALL)
    horizon = re.search(horizon_pattern, stock_recommendation_str, re.DOTALL)
    conflicts = re.search(conflicts_pattern, stock_recommendation_str, re.DOTALL)

    # Create a pandas series from stock recommendation string
    stock_recommendation_series = pd.Series(
        {
            "mentioned_by_finfluencer": (
                mentioned_by_finfluencer.group(1) if mentioned_by_finfluencer else None
            ),
            "recommendation": recommendation.group(1) if recommendation else None,
            "explanation": explanation.group(1) if explanation else None,
            "confidence": confidence.group(1) if confidence else None,
            "virality": virality.group(1) if virality else None,
            "risks": risks.group(1) if risks else None,
            "horizon": horizon.group(1) if horizon else None,
            "conflicts": conflicts.group(1) if conflicts else None,
        }
    )

    return stock_recommendation_series


def _coerce_history(x):
    # Accept list or JSON string; return list[{"role","content"}]
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return []
    return list(x)


def messages_to_input(messages: list) -> str:
    """
    Convert a list of chat messages (each a dict with 'role' and 'content')
    into a single transcript string for the Responses API 'input'.
    """
    lines = []
    for m in messages:
        role = str(m.get("role", "")).upper()
        content = str(m.get("content", "")).strip()
        if content:  # skip empty
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def create_batch_file(
    prompts: pd.DataFrame,
    project_name: str,
    execution_date: str,
    gpt_model: str,
    system_prompt_field: str,
    user_prompt_field: str = "question_prompt",
    history_field: str = None,
    batch_file_name: str = "batch_input.jsonl",
    vector_store_ids: list = [],
) -> str:
    # Creating an array of json tasks
    tasks = []

    for i in range(len(prompts)):
        custom_id = f"{prompts.loc[i, 'custom_id']}"
        sys_txt = (
            str(prompts.loc[i, system_prompt_field])
            if system_prompt_field in prompts.columns
            else ""
        )

        user_txt = (
            str(prompts.loc[i, user_prompt_field])
            if user_prompt_field in prompts.columns
            else ""
        )

        history = _coerce_history(
            prompts.get(history_field, [None])[i]
            if history_field in prompts.columns
            else []
        )

        # Build messages
        messages = []
        if sys_txt:
            messages.append({"role": "system", "content": sys_txt})

        if history:
            for m in history:
                r, c = m.get("role", "user"), m.get("content", "")
                messages.append({"role": r, "content": c})

        messages.append({"role": "user", "content": user_txt})

        if gpt_model.startswith("gpt-4"):
            if vector_store_ids:
                task = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": gpt_model,
                        "temperature": 0,
                        "input": messages_to_input(messages),
                        "tools": [
                            {
                                "type": "file_search",
                                "vector_store_ids": vector_store_ids,
                            }
                        ],
                    },
                }

            else:
                task = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": gpt_model,
                        "temperature": 0,
                        "messages": messages,
                    },
                }

        elif gpt_model.startswith("gpt-5"):
            if vector_store_ids:
                task = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": gpt_model,
                        "input": messages_to_input(messages),
                        "tools": [
                            {
                                "type": "file_search",
                                "vector_store_ids": vector_store_ids,
                            }
                        ],
                    },
                }

            else:
                task = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": gpt_model,
                        "messages": messages,
                    },
                }
        else:
            raise ValueError(f"Unsupported GPT model: {gpt_model}")

        tasks.append(task)

    # Creating batch file
    with open(
        f"{base_dir}/../data/{project_name}/{execution_date}/batch-files/{batch_file_name}",
        "w",
    ) as file:
        for obj in tasks:
            file.write(json.dumps(obj) + "\n")

    return batch_file_name


def batch_query(
    project_name: str,
    execution_date: str,
    batch_input_file_dir: str,
    batch_output_file_dir: str,
    vector_store_ids: list = [],
) -> pd.DataFrame:
    # Upload batch input file
    batch_file = openai_client.files.create(
        file=open(
            f"{base_dir}/../data/{project_name}/{execution_date}/batch-files/{batch_input_file_dir}",
            "rb",
        ),
        purpose="batch",
    )

    # Create batch job
    if vector_store_ids:
        batch_job = openai_client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/responses",
            completion_window="24h",
        )
    else:
        batch_job = openai_client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

    # Check batch status
    while True:
        batch_job = openai_client.batches.retrieve(batch_job.id)
        print(f"Batch job status: {batch_job.status}")
        if batch_job.status == "completed":
            break
        elif batch_job.status == "failed":
            raise Exception("Batch job failed.")
        else:
            # Wait for 5 minutes before checking again
            time.sleep(300)

    # Retrieve batch results
    result_file_id = batch_job.output_file_id
    results = openai_client.files.content(result_file_id).content

    # Save the batch output
    with open(
        f"{base_dir}/../data/{project_name}/{execution_date}/batch-files/{batch_output_file_dir}",
        "wb",
    ) as file:
        file.write(results)

    # Loading data from saved output file
    response_list = []
    with open(
        f"{base_dir}/../data/{project_name}/{execution_date}/batch-files/{batch_output_file_dir}",
        "r",
    ) as file:
        for line in file:
            # Parsing the JSON result string into a dict
            result = json.loads(line.strip())

            if vector_store_ids:
                try:
                    response_list.append(
                        {
                            "custom_id": f'{result["custom_id"]}',
                            "query_response": result["response"]["body"]["output"][1][
                                "content"
                            ][0]["text"],
                        }
                    )
                except IndexError:
                    response_list.append(
                        {
                            "custom_id": f'{result["custom_id"]}',
                            "query_response": result["response"]["body"]["output"][0][
                                "content"
                            ][0]["text"],
                        }
                    )
                except:
                    raise

            else:
                response_list.append(
                    {
                        "custom_id": f'{result["custom_id"]}',
                        "query_response": result["response"]["body"]["choices"][0][
                            "message"
                        ]["content"],
                    }
                )

    return pd.DataFrame(response_list)


def extract_profile_id(author_metadata: str) -> str:
    """
    Extracts the profile ID from the given author metadata string.

    Args:
        author_metadata (str): A string representation of a dictionary containing author metadata.

    Returns:
        str: The profile ID extracted from the author metadata.
    """
    author_metadata_dict = ast.literal_eval(author_metadata)
    return str(author_metadata_dict.get("id"))


def extract_tagged_users(tagged_str: str, is_tiktok: bool = True) -> str:
    """
    Extracts user handles from a string representation of a list of tagged users.

    Args:
        tagged_str (str): A string representation of a list of dictionaries,
                          where each dictionary contains a "user_handle" key.
        is_tiktok (bool): A boolean indicating whether the tagged users are from TikTok.

    Returns:
        str: A comma-separated string of user handles. If the input is invalid
             or an error occurs, an empty string is returned.
    """
    try:
        user_list = []
        tagged_list = ast.literal_eval(tagged_str)
        for tag in tagged_list:
            if is_tiktok:
                user_list.append(tag.get("user_handle", ""))
            else:  # For X (formerly Twitter)
                user_list.append(tag.get("profile_name", ""))

        return ", ".join([user for user in user_list if user != ""])

    except Exception as e:
        return ""


def extract_hashtags(hashtags_str: str) -> str:
    """
    Extracts hashtags from a raw string representation of a list og hashtags.
    Args:
        hashtags_str (str): A string representation of a list of hashtags.
    Returns:
        str: A comma-separated string of hashtag names. If an error occurs,
             an empty string is returned.
    """
    try:
        hashtags_list = ast.literal_eval(hashtags_str)
        return ", ".join([hashtag for hashtag in hashtags_list if hashtag != ""])

    except Exception as e:
        return ""


def calculate_video_engagement(video_data: pd.Series) -> float:
    """
    Calculate the engagement rate of a video based on its interaction metrics.

    The engagement rate is calculated as the sum of likes, shares, comments, and saves
    divided by the number of views. If the number of views is zero, the engagement rate
    is set to 0.0 to avoid division by zero.

    Args:
        video_data (pd.Series): A pandas Series containing the video's interaction metrics.
            Expected keys are:
            - "digg_count": Number of likes.
            - "share_count": Number of shares.
            - "comment_count": Number of comments.
            - "collect_count": Number of saves.
            - "play_count": Number of views.

    Returns:
        float: The engagement rate of the video.
    """
    num_likes = pd.to_numeric(video_data["digg_count"], errors="coerce")
    num_shares = pd.to_numeric(video_data["share_count"], errors="coerce")
    num_comments = pd.to_numeric(video_data["comment_count"], errors="coerce")
    num_saves = pd.to_numeric(video_data["collect_count"], errors="coerce")
    num_views = pd.to_numeric(video_data["play_count"], errors="coerce")

    # Replace NaN values with 0
    num_likes = num_likes if pd.notna(num_likes) else 0
    num_shares = num_shares if pd.notna(num_shares) else 0
    num_comments = num_comments if pd.notna(num_comments) else 0
    num_saves = num_saves if pd.notna(num_saves) else 0
    num_views = num_views if pd.notna(num_views) else 0

    video_engagement = (
        (num_likes + num_shares + num_comments + num_saves) / num_views
        if num_views > 0
        else 0.0
    )
    return video_engagement


def extract_video_transcripts(profile_id: str, video_metadata: pd.DataFrame) -> str:
    """
    Extracts and combines video transcripts for a given profile ID from the provided video metadata.

    Args:
        profile_id (str): The unique identifier for the profile whose video transcripts are to be extracted.
        video_metadata (pd.DataFrame): A pandas DataFrame containing metadata for videos, including columns such as
            'profile_id', 'create_time', 'description', 'video_duration', 'digg_count', 'share_count', 'play_count',
            'collect_count', 'comment_count', 'tagged_user', 'hashtags', and 'video_transcript'.

    Returns:
        str: A single string containing the combined video transcripts, formatted with additional metadata such as
        creation date, description, duration, engagement metrics, tagged users, and hashtags, separated by newlines.
    """
    # Filter the rows where profile_id matches
    filtered_videos = video_metadata[video_metadata["account_id"] == profile_id].copy()

    # Sort the filtered videos by creation time from latest to oldest
    filtered_videos = filtered_videos.sort_values(
        by="create_time", ascending=False
    ).reset_index(drop=True)

    # Join the list of video transcripts into a single string, separated by newlines
    video_transcripts_list = []
    for i in range(len(filtered_videos)):
        video_transcripts_list += [
            tiktok_video_prompt_template.format(
                video_creation_date=filtered_videos.loc[i, "create_time"],
                video_description=(
                    filtered_videos.loc[i, "description"].replace("\n", " ")
                    if not pd.isnull(filtered_videos.loc[i, "description"])
                    else ""
                ),
                video_duration=filtered_videos.loc[i, "video_duration"],
                num_likes=filtered_videos.loc[i, "digg_count"],
                num_shares=filtered_videos.loc[i, "share_count"],
                view_count=filtered_videos.loc[i, "play_count"],
                num_saves=filtered_videos.loc[i, "collect_count"],
                num_comments=filtered_videos.loc[i, "comment_count"],
                total_engagement_over_num_views=calculate_video_engagement(
                    filtered_videos.loc[i, :]
                ),
                tagged_users=extract_tagged_users(
                    filtered_videos.loc[i, "tagged_user"]
                ),
                hashtags=extract_hashtags(filtered_videos.loc[i, "hashtags"]),
                video_transcript=filtered_videos.loc[i, "video_transcript"],
            )
        ]

    return "\n".join(video_transcripts_list)


def extract_tweets(profile_id: str, tweet_metadata: pd.DataFrame) -> str:
    # Filter the rows where profile_id matches
    filtered_tweets = tweet_metadata[tweet_metadata["account_id"] == profile_id].copy()

    # Sort the filtered videos by creation time from latest to oldest
    filtered_tweets = filtered_tweets.sort_values(
        by="createdAt", ascending=False
    ).reset_index(drop=True)

    # Join the list of tweets into a single string, separated by newlines
    tweets_list = []
    for i in range(len(filtered_tweets)):
        tweets_list += [
            x_tweet_prompt_template.format(
                created_at=(
                    filtered_tweets.loc[i, "createdAt"]
                    if "createdAt" in filtered_tweets.columns
                    else ""
                ),
                text=(
                    filtered_tweets.loc[i, "text"]
                    if "text" in filtered_tweets.columns
                    else ""
                ),
                like_count=(
                    filtered_tweets.loc[i, "likeCount"]
                    if "likeCount" in filtered_tweets.columns
                    else ""
                ),
                view_count=(
                    filtered_tweets.loc[i, "viewCount"]
                    if "viewCount" in filtered_tweets.columns
                    else ""
                ),
                retweet_count=(
                    filtered_tweets.loc[i, "retweetCount"]
                    if "retweetCount" in filtered_tweets.columns
                    else ""
                ),
                reply_count=(
                    filtered_tweets.loc[i, "replyCount"]
                    if "replyCount" in filtered_tweets.columns
                    else ""
                ),
                quote_count=(
                    filtered_tweets.loc[i, "quoteCount"]
                    if "quoteCount" in filtered_tweets.columns
                    else ""
                ),
                bookmark_count=(
                    filtered_tweets.loc[i, "bookmarkCount"]
                    if "bookmarkCount" in filtered_tweets.columns
                    else ""
                ),
                lang=(
                    filtered_tweets.loc[i, "lang"]
                    if "lang" in filtered_tweets.columns
                    else ""
                ),
                tagged_users=(
                    filtered_tweets.loc[i, "tagged_users"]
                    if "tagged_users" in filtered_tweets.columns
                    else ""
                ),
                hashtags=(
                    filtered_tweets.loc[i, "hashtags"]
                    if "hashtags" in filtered_tweets.columns
                    else ""
                ),
            )
        ]

    return "\n\n".join(tweets_list)


def row_query(row: pd.Series, args: list) -> str:
    system_prompt = row[args[0][0]]
    user_prompt = row[args[0][1]]
    gpt_model = args[0][2]
    enable_web_search = args[0][3]

    # Skip if system_prompt/user_prompt is empty or NaN (depending on your logic)
    if not isinstance(system_prompt, str) or not isinstance(user_prompt, str):
        return ""

    # Make a chat completion request
    try:
        if enable_web_search:
            response = openai_client.responses.create(
                model=gpt_model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tools=[
                    {
                        "type": "web_search",
                        "search_context_size": "medium",
                        "user_location": {"type": "approximate", "country": "US"},
                    }
                ],
                tool_choice="required",
                # temperature=0,
            )
        else:
            response = openai_client.responses.create(
                model=gpt_model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
            )
        return response.output_text

    except Exception as e:
        # Handle errors (rate limits, etc.)
        print(f"Error processing row: {e}")
        return "Error or Timeout"


def perform_profile_interview(
    project_name: str,
    execution_date: str,
    gpt_model: str,
    profile_metadata_file: str,
    post_file: str,
    output_file: str,
    system_prompt_template: str,
    user_prompt_template: str,
    llm_response_field: str,
    interview_type: str,
    history_field: str = None,
    vector_store_ids: list = [],
    use_row_query: bool = False,
    enable_web_search: bool = False,
) -> None:
    # Create the project subfolder within the data folder if it does not exist
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(base_dir, "../data"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "../data", project_name), exist_ok=True)
    os.makedirs(
        os.path.join(base_dir, "../data", project_name, execution_date), exist_ok=True
    )

    # Load profile and post metadata
    profile_metadata = pd.read_csv(
        os.path.join(
            base_dir, "../data", project_name, execution_date, profile_metadata_file
        )
    )
    post_metadata = pd.read_csv(
        os.path.join(base_dir, "../data", project_name, execution_date, post_file),
        on_bad_lines="skip",
    )
    if "warning_code" in post_metadata.columns:
        post_metadata = post_metadata[
            post_metadata["warning_code"] != "dead_page"
        ].reset_index(drop=True)
    if "error_code" in post_metadata.columns:
        post_metadata = post_metadata[
            post_metadata["error_code"] != "crawl_failed"
        ].reset_index(drop=True)

    # Generate system and user prompts
    if interview_type.startswith("tiktok"):
        post_metadata["create_time"] = pd.to_datetime(post_metadata["create_time"])
        profile_metadata["posts_combined"] = profile_metadata["account_id"].apply(
            extract_video_transcripts, args=(post_metadata,)
        )
    elif interview_type.startswith("x"):
        try:
            post_metadata["createdAt"] = pd.to_datetime(
                post_metadata["createdAt"], format="%a %b %d %H:%M:%S %z %Y"
            )
        except ValueError:
            post_metadata["createdAt"] = pd.to_datetime(post_metadata["createdAt"])
        profile_metadata["posts_combined"] = profile_metadata["account_id"].apply(
            extract_tweets, args=(post_metadata,)
        )
    else:
        raise ValueError(f"Interview type: {interview_type} not supported.")

    if system_prompt_template:
        profile_metadata[f"{interview_type}_system_prompt"] = profile_metadata.apply(
            construct_system_prompt,
            args=(system_prompt_template, interview_type),
            axis=1,
        )
    profile_metadata[f"{interview_type}_user_prompt"] = profile_metadata.apply(
        construct_user_prompt, args=(user_prompt_template, interview_type), axis=1
    )

    # Generate custom ids
    if "custom_id" in profile_metadata.columns:
        profile_metadata.drop(columns="custom_id", inplace=True)

    profile_metadata = profile_metadata.reset_index(drop=False)
    profile_metadata.rename(columns={"index": "custom_id"}, inplace=True)

    # Create folder to contain batch files
    os.makedirs(
        os.path.join(base_dir, "../data", project_name, execution_date, "batch-files"),
        exist_ok=True,
    )

    if (
        use_row_query or enable_web_search
    ):  # When performing row-wise queries or enabling web search
        profile_metadata_with_responses = profile_metadata.copy()
        row_query_args = [
            f"{interview_type}_system_prompt",
            f"{interview_type}_user_prompt",
            gpt_model,
            enable_web_search,
        ]

        # Choose how many parallel calls you want (tune for your rate limits)
        max_workers = NUM_PARALLEL_PROCESSES

        # Prepare rows in order so results line up with the DataFrame
        rows = [row for _, row in profile_metadata.iterrows()]

        def run_row_query(row):
            # row_query(row, args=(...)) matches your previous progress_apply usage
            return row_query(
                row,
                args=(row_query_args,),
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                tqdm_auto(executor.map(run_row_query, rows), total=len(rows))
            )

        # Assign results back to the DataFrame in the same order
        profile_metadata_with_responses[llm_response_field] = results

    else:  # Perform batch queries to save cost
        # Perform batch query for survey questions
        create_batch_file(
            profile_metadata,
            project_name=project_name,
            execution_date=execution_date,
            gpt_model=gpt_model,
            system_prompt_field=f"{interview_type}_system_prompt",
            user_prompt_field=f"{interview_type}_user_prompt",
            history_field=history_field,
            batch_file_name="batch_input.jsonl",
            vector_store_ids=vector_store_ids,
        )

        llm_responses = batch_query(
            project_name=project_name,
            execution_date=execution_date,
            batch_input_file_dir="batch_input.jsonl",
            batch_output_file_dir="batch_output.jsonl",
            vector_store_ids=vector_store_ids,
        )
        llm_responses.rename(
            columns={"query_response": llm_response_field}, inplace=True
        )

        # Merge LLM response with original dataset
        profile_metadata["custom_id"] = profile_metadata["custom_id"].astype("int64")
        llm_responses["custom_id"] = llm_responses["custom_id"].astype("int64")
        profile_metadata_with_responses = pd.merge(
            left=profile_metadata,
            right=llm_responses[["custom_id", llm_response_field]],
            on="custom_id",
        )

    # Save profile metadata after analysis into CSV file
    profile_metadata_with_responses.to_csv(
        os.path.join(base_dir, "../data", project_name, execution_date, output_file),
        index=False,
    )


def build_profile_prompt(
    project_name: str,
    profile_metadata_input_file: str,
    profile_metadata_output_file: str,
    video_metadata_file: str,
) -> None:
    # Load profile and video metadata
    print("Loading profile and video metadata...")
    profile_metadata = pd.read_csv(
        f"{base_dir}/../data/{project_name}/{profile_metadata_input_file}"
    )
    video_metadata = pd.read_csv(
        f"{base_dir}/../data/{project_name}/{video_metadata_file}"
    )
    video_metadata["createTimeISO"] = pd.to_datetime(video_metadata["createTimeISO"])

    # Preprocess profile and video metadata
    print("Preprocess profile and video metadata...")
    video_metadata["profile_id"] = video_metadata["authorMeta"].apply(
        extract_profile_id
    )
    video_metadata["profile_id"] = video_metadata["profile_id"].astype(str)
    profile_metadata["id"] = profile_metadata["id"].astype(str)

    # Construct past transcripts
    print("Construct past transcripts...")
    profile_metadata["posts_combined"] = profile_metadata["id"].apply(
        extract_video_transcripts, args=(video_metadata,)
    )

    # Construct profile prompt
    print("Construct profile prompt...")
    profile_metadata["profile_prompt"] = profile_metadata.apply(
        lambda row: tiktok_profile_prompt_template.format(
            profile_image=row.get("avatar", ""),
            profile_name=row.get("profile", ""),
            profile_nickname=row.get("nickName", ""),
            verified_status=row.get("verified", ""),
            private_account=row.get("privateAccount", ""),
            region=row.get("region", ""),
            tiktok_seller=row.get("ttSeller", ""),
            profile_signature=row.get("signature", ""),
            num_followers=row.get("fans", ""),
            num_following=row.get("following", ""),
            num_likes=row.get("heart", ""),
            num_videos=row.get("video", ""),
            num_digg=row.get("digg", ""),
            total_likes_over_num_followers=calculate_profile_engagement(
                row["heart"], row["fans"]
            ),
            total_likes_over_num_videos=calculate_profile_engagement(
                row["heart"], row["video"]
            ),
            video_transcripts=row.get("posts_combined", ""),
        ),
        axis=1,
    )

    # Save updated profile metadata
    profile_metadata.to_csv(
        f"{base_dir}/../data/{project_name}/{profile_metadata_output_file}", index=False
    )

    return None


def perform_video_transcription(
    project_name: str, execution_date: str, video_file: str
) -> None:
    """
    Perform video transcription for a given project by downloading videos, transcribing them,
    and updating the video metadata file.

    Args:
        project_name (str): The name of the project. Used to organize data and video files.
        execution_date (str): The date of the pipeline execution, used to create a unique directory name.
        video_metadata_file (str): The name of the CSV file containing video metadata.
                                   This file should be located in the project's data folder.

    Raises:
        FileNotFoundError: If the specified video metadata file does not exist.
    """
    # Create the video downloads folder for project if it does not exist
    base_dir = os.path.dirname(os.path.abspath(__file__))
    video_download_folder_path = os.path.join(
        base_dir, "../data", project_name, execution_date, "video-downloads"
    )
    os.makedirs(video_download_folder_path, exist_ok=True)

    # Load video metadata
    video_metadata_path = os.path.join(
        base_dir, "../data", project_name, execution_date, video_file
    )
    if not os.path.exists(video_metadata_path):
        raise FileNotFoundError(f"{video_metadata_path} not found.")
    else:
        video_metadata = pd.read_csv(video_metadata_path)

    if "video_transcript" not in video_metadata.columns:
        video_metadata = video_metadata.dropna(
            subset=["post_id"], inplace=False
        ).reset_index(drop=True)
        video_metadata["post_id"] = video_metadata["post_id"].astype(str)
        video_metadata["video_filename"] = video_metadata["post_id"].apply(
            lambda x: x + ".mp4"
        )
        video_metadata["video_transcript"] = None

    # Filter out videos that have not been transcribed
    video_metadata_without_transcript = (
        video_metadata[video_metadata["video_transcript"].isnull()]
        .copy()
        .reset_index(drop=True)
    )
    video_metadata_without_transcript.dropna(subset=["post_id"], inplace=True)
    video_metadata_without_transcript["post_id"] = video_metadata_without_transcript[
        "post_id"
    ].astype(str)
    video_metadata_without_transcript["video_filename"] = (
        video_metadata_without_transcript["post_id"].apply(lambda x: x + ".mp4")
    )

    # Download videos that have not been transcribed
    video_metadata_without_transcript.progress_apply(
        download_video, args=(project_name, execution_date), axis=1
    )

    # Perform transcription on downloaded videos and save transcripts along the way
    for i in tqdm(range(len(video_metadata))):
        if video_metadata.loc[i, "video_transcript"] is None or pd.isna(
            video_metadata.loc[i, "video_transcript"]
        ):
            video_metadata.loc[i, "video_transcript"] = transcribe_videos(
                video_metadata.loc[i, :], project_name, execution_date
            )
            video_metadata.to_csv(video_metadata_path, index=False)

        else:
            continue

    # Clean up downloaded videos to save disk space
    for file in os.listdir(video_download_folder_path):
        file_path = os.path.join(video_download_folder_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)


def update_verified_profile_pool(
    project_name: str,
    execution_date: str,
    input_file: str,
    verified_profile_pool: str,
    prediction_threshold: float,
    filter_by_stock_recommendation: bool = True,
) -> None:
    """
    Updates the verified profile pool by adding new financial influencers identified from the onboarding interview data.

    This function reads the interviewed profiles and the existing verified profile pool from CSV files, filters out profiles that meet or exceed the specified finfluencer likelihood threshold and have stock recommendations, and appends these new profiles to the verified profile pool. The updated pool is then saved back to the CSV file.

    Args:
        project_name (str): Name of the project directory containing the data.
        execution_date (str): Date of execution, used to locate the input file and as the inclusion date for new profiles.
        input_file (str): Relative path to the CSV file containing interviewed profiles.
        verified_profile_pool (str): Filename of the verified profile pool CSV.
        prediction_threshold (float): Minimum likelihood score to consider a profile as a financial influencer.
        filter_by_stock_recommendation (bool): Whether to filter profiles by the presence of stock recommendations.

    Returns:
        None
    """
    interviewed_profiles = pd.read_csv(
        os.path.join(base_dir, "../data", project_name, execution_date, input_file)
    )
    verified_profiles = pd.read_csv(
        os.path.join(base_dir, "../data", project_name, verified_profile_pool)
    )

    # Filter out financial influencers identified during onboarding interview
    finfluencer_likelihood_col = "Indicate on a scale of 0 to 100, how likely this content creator is a finfluencer (0 means most definitely not a finfluencer and 100 means most definitely a finfluencer)? - value"
    interviewed_profiles[finfluencer_likelihood_col] = interviewed_profiles[
        finfluencer_likelihood_col
    ].astype(float)
    finfluencer_profiles = interviewed_profiles[
        interviewed_profiles[finfluencer_likelihood_col] >= prediction_threshold
    ].reset_index(drop=True)

    # Filter out financial influencers that had a stock recommendation
    if filter_by_stock_recommendation:
        finfluencer_profiles = finfluencer_profiles[
            finfluencer_profiles["stock_mentions"].notna()
            & (finfluencer_profiles["stock_mentions"] != "")
        ].reset_index(drop=True)

    # Add new financial influencers to the verified profile pool
    if not finfluencer_profiles.empty:
        verified_profiles = pd.concat(
            [
                verified_profiles,
                pd.DataFrame(
                    {
                        "account_id": finfluencer_profiles["account_id"].tolist(),
                        "inclusion_date": execution_date,
                        "influence": finfluencer_profiles[
                            "Indicate on a scale of 0 to 100, how influential this influencer is (0 means not at all influential and 100 means very influential with millions of followers and mainstream recognition)? Please consider quantitative thresholds such as follower count and engagement rate when answering this question. For example, a micro-influencer will be in the 20-40 range, whereas an account with hundreds of thousands of followers and high engagement might rate 80+. - value"
                        ].tolist(),
                        "credibility": finfluencer_profiles[
                            "Indicate on a scale of 0 to 100, how credible or authoritative this influencer is (0 means not at all credible or authoritative and 100 means very credible and authoritative)? - value"
                        ].tolist(),
                    }
                ),
            ],
            ignore_index=True,
        )

        # Save updated verified profile pool
        verified_profiles.to_csv(
            os.path.join(base_dir, "../data", project_name, verified_profile_pool),
            index=False,
        )
    else:
        pass


def extract_stock_mentions_from_posts(
    row: pd.Series, russell_4000_stock: pd.DataFrame
) -> str:
    """
    Extracts stock mentions from a user's posts based on a list of Russell 4000 stocks.

    This function processes a row containing combined posts, splits the posts into individual chunks,
    and searches each chunk for mentions of stocks from the provided Russell 4000 stock DataFrame.
    Mentions are detected by matching the full stock name, shortened stock name, or ticker symbol
    (with optional $ or # prefix). For each mention found, the function records the stock name,
    ticker, post date, and the post content.

    Args:
        row (pd.Series): A pandas Series representing a row from a DataFrame, expected to contain a
            "posts_combined" field with the user's posts as a single string.
        russell_4000_stock (pd.DataFrame): A DataFrame containing Russell 4000 stock information,
            with columns "COMNAM" (full name), "SHORTEN_COMNAM" (shortened name), and "TICKER" (ticker symbol).

    Returns:
        str: A formatted string listing all detected stock mentions, including stock name, ticker,
            mention date, and the corresponding post content.
    """
    # Split the transcripts by double newline
    if not pd.isnull(row["posts_combined"]):
        transcript_chunks = re.split(r"(?=Creation Date:)", row["posts_combined"])
    else:
        transcript_chunks = []

    # Prepare a list for storing the matched results
    found_mentions = []

    for chunk in transcript_chunks:
        if pd.isnull(chunk) or not chunk:
            continue

        # Initialize variables for creation date and transcript text
        creation_date = "Unknown"

        # Extract creation date using a regular expression
        creation_date_match = re.search(r"Creation Date:\s*(.+)", chunk)
        if creation_date_match:
            creation_date = creation_date_match.group(1).strip()

        # Check each stock in the Russell 4000
        for _, row in russell_4000_stock.iterrows():
            full_stock_name = row["COMNAM"].strip()
            shorted_stock_name = row["SHORTEN_COMNAM"].strip()
            stock_ticker = row["TICKER"].strip()

            # Check if stock name is found in transcript chunk
            name_match = (
                re.search(
                    rf"\b{re.escape(full_stock_name.lower())}\b",
                    chunk.lower(),
                )
                is not None
                or re.search(
                    rf"\b{re.escape(shorted_stock_name.lower())}\b",
                    chunk.lower(),
                )
                is not None
                or re.search(
                    rf"\$\b{re.escape(stock_ticker.lower())}\b",
                    chunk.lower(),
                )
                is not None
                or re.search(
                    rf"\#\b{re.escape(stock_ticker.lower())}\b",
                    chunk.lower(),
                )
                is not None
            )

            if name_match:
                found_mentions.append(
                    {
                        "stock_name": full_stock_name,
                        "stock_ticker": stock_ticker,
                        "post_date": creation_date,
                        "post": chunk,
                    }
                )

    # Build a DataFrame from the matches
    stock_mentions_df = pd.DataFrame(
        found_mentions, columns=["stock_name", "stock_ticker", "post_date", "post"]
    )

    # Remove duplicates if you only want unique (stock, date) pairs
    stock_mentions_df = stock_mentions_df.drop_duplicates().reset_index(drop=True)

    # Create a formatted text string containing all the stocks mentioned and the questions for each stock
    stock_mentions_formatted_str = ""
    stock_question_template = """**stock name: {stock_name}**
**stock ticker: {stock_ticker}**
**mention date: {post_date}**
**post: {post}**"""

    for i in range(len(stock_mentions_df)):
        if i != 0:
            stock_mentions_formatted_str += "\n\n"
        stock_mentions_formatted_str += stock_question_template.format(
            stock_name=stock_mentions_df.loc[i, "stock_name"],
            stock_ticker=stock_mentions_df.loc[i, "stock_ticker"],
            post_date=stock_mentions_df.loc[i, "post_date"],
            post=stock_mentions_df.loc[i, "post"],
        )

    return stock_mentions_formatted_str


def extract_stock_mentions(
    project_name: str,
    execution_date: str,
    profile_metadata_file: str,
    post_file: str,
    output_file: str,
    interview_type: str,
) -> None:
    """
    Extracts stock mentions from influencer posts and saves the results to a CSV file.

    This function loads influencer profile metadata and post metadata, processes the posts to extract combined post content for each influencer based on the interview type (e.g., TikTok or X), and then identifies mentions of Russell 4000 stocks in the posts. The results, including the extracted stock mentions, are saved to an output CSV file.

    Args:
        project_name (str): Name of the project directory.
        execution_date (str): Date string representing the execution date (used for file paths).
        profile_metadata_file (str): Filename of the influencer profile metadata CSV.
        post_file (str): Filename of the post metadata CSV.
        output_file (str): Filename for the output CSV with extracted stock mentions.
        interview_type (str): Type of interview or platform (e.g., "tiktok", "x") to determine post processing logic.

    Raises:
        ValueError: If the provided interview_type is not supported.
    """
    # Load influencer profile metadata and post metadata files
    profile_metadata = pd.read_csv(
        os.path.join(
            base_dir, "../data", project_name, execution_date, profile_metadata_file
        )
    )
    post_metadata = pd.read_csv(
        os.path.join(base_dir, "../data", project_name, execution_date, post_file)
    )

    if interview_type.startswith("tiktok"):
        post_metadata["create_time"] = pd.to_datetime(post_metadata["create_time"])
        profile_metadata["posts_combined"] = profile_metadata["account_id"].apply(
            extract_video_transcripts, args=(post_metadata,)
        )
    elif interview_type.startswith("x"):
        try:
            post_metadata["createdAt"] = pd.to_datetime(
                post_metadata["createdAt"], format="%a %b %d %H:%M:%S %z %Y"
            )
        except ValueError:
            post_metadata["createdAt"] = pd.to_datetime(post_metadata["createdAt"])
        profile_metadata["posts_combined"] = profile_metadata["account_id"].apply(
            extract_tweets, args=(post_metadata,)
        )
    else:
        raise ValueError(f"Interview type: {interview_type} not supported.")

    # Extract stocks mention in past posts
    russell_4000_stock = pd.read_csv(
        os.path.join(base_dir, "../config", RUSSELL_4000_STOCK_TICKER_FILE)
    )
    profile_metadata["stock_mentions"] = profile_metadata.progress_apply(
        extract_stock_mentions_from_posts, args=(russell_4000_stock,), axis=1
    )

    # Save formatted post reflection results
    profile_metadata.to_csv(
        os.path.join(base_dir, "../data", project_name, execution_date, output_file),
        index=False,
    )


def perform_tiktok_keyword_search(
    project_name: str,
    execution_date: str,
    search_terms: list,
    output_file: str,
    num_post_per_keyword: int,
) -> pd.DataFrame:
    """
    Perform a TikTok keyword search using the Bright Data API and save the results to a CSV file.

    Args:
        project_name (str): The name of the project. A subfolder with this name will be created
            within the data folder to store the output file.
        execute_date (str): The date of the pipeline execution, used to create a unique directory name.
        search_terms (list): The list containing the search terms,
            one term per line.
        output_file (str): The file path where the resulting CSV file will be saved.
        num_post_per_keyword (int): The maximum number of posts that should be returned per keyword search.

    Returns:
        pd.DataFrame: Returns the keyword search results as a pandas Dataframe.

    Raises:
        requests.exceptions.RequestException: If there is an issue with the API request.
        KeyError: If the response from the API does not contain the expected keys.
        ValueError: If the response data is not in the expected format.
    """
    # Create the project subfolder within the data folder if it does not exist
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(base_dir, "../data"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "../data", project_name), exist_ok=True)
    os.makedirs(
        os.path.join(base_dir, "../data", project_name, execution_date), exist_ok=True
    )

    # Initialise keyword search job
    data = [
        {"search_keyword": keyword, "num_of_posts": num_post_per_keyword, "country": ""}
        for keyword in search_terms
    ]
    response = requests.post(
        "https://api.brightdata.com/datasets/v3/trigger",
        headers={
            "Authorization": f"Bearer {BRIGHTDATA_API}",
            "Content-Type": "application/json",
        },
        params={
            "dataset_id": "gd_lu702nij2f790tmv9h",
            "format": "csv",
            "uncompressed_webhook": "true",
            "force_deliver": "true",
            "include_errors": "true",
            "type": "discover_new",
            "discover_by": "keyword",
        },
        json=data,
    )
    snapshot_id = response.json().get("snapshot_id")

    # Retrieve keyword search results
    response_json = {"status": "running"}
    while response_json.get("status") != "ready":
        time.sleep(WAIT_TIME_BETWEEN_RETRIEVAL_REQUESTS)
        response = requests.get(
            f"https://api.brightdata.com/datasets/v3/progress/{snapshot_id}",
            headers={
                "Authorization": f"Bearer {BRIGHTDATA_API}",
            },
        )
        response_json = response.json()

    retries = 0
    while True:
        try:
            response = requests.get(
                f"https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}",
                headers={
                    "Authorization": f"Bearer {BRIGHTDATA_API}",
                },
                params={
                    "format": "json",
                },
            )
            response_json = response.json()
            keyword_search_results = pd.DataFrame(response_json)
            break

        except (requests.RequestException, ValueError, json.JSONDecodeError) as err:
            retries += 1
            if retries > MAX_RETRIES:
                raise RuntimeError(
                    f"Failed to retrieve snapshot after {MAX_RETRIES} attempts: {err}"
                )
            time.sleep(WAIT_TIME_BETWEEN_RETRIEVAL_REQUESTS)

    if "warning_code" in keyword_search_results.columns:
        keyword_search_results = keyword_search_results[
            keyword_search_results["warning_code"] != "dead_page"
        ].reset_index(drop=True)
    if "error_code" in keyword_search_results.columns:
        keyword_search_results = keyword_search_results[
            keyword_search_results["error_code"] != "crawl_failed"
        ].reset_index(drop=True)
    keyword_search_results.to_csv(
        os.path.join(base_dir, "../data", project_name, execution_date, output_file),
        index=False,
    )

    return keyword_search_results


def perform_tiktok_profile_search(
    project_name: str,
    execution_date: str,
    input_file: str,
    output_file: str,
    start_date: str,
    end_date: str,
    num_posts_per_profile: int,
    local_file: str = None,
) -> pd.DataFrame:
    # Create the project subfolder within the data folder if it does not exist
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(base_dir, "../data"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "../data", project_name), exist_ok=True)
    os.makedirs(
        os.path.join(base_dir, "../data", project_name, execution_date), exist_ok=True
    )

    # Define search parameters
    profile_list = pd.read_csv(
        os.path.join(base_dir, "../data", project_name, input_file)
    )["account_id"].tolist()

    if local_file is None:  # Perform API search
        # Initialise profile search job
        data = [
            {
                "url": f"https://www.tiktok.com/@{profile}",
                "num_of_posts": num_posts_per_profile,
                "posts_to_not_include": "",
                "start_date": start_date,
                "end_date": end_date,
                "what_to_collect": "Posts",
                "post_type": "Video Posts",
                "country": "",
            }
            for profile in profile_list
        ]

        # Initialise profile search job
        response = requests.post(
            "https://api.brightdata.com/datasets/v3/trigger",
            headers={
                "Authorization": f"Bearer {BRIGHTDATA_API}",
                "Content-Type": "application/json",
            },
            params={
                "dataset_id": "gd_lu702nij2f790tmv9h",
                "include_errors": "true",
                "type": "discover_new",
                "discover_by": "profile_url",
            },
            json=data,
        )
        snapshot_id = response.json().get("snapshot_id")

        # Retrieve profile search results
        response_json = {"status": "running"}
        while response_json.get("status") != "ready":
            time.sleep(WAIT_TIME_BETWEEN_RETRIEVAL_REQUESTS)
            response = requests.get(
                f"https://api.brightdata.com/datasets/v3/progress/{snapshot_id}",
                headers={
                    "Authorization": f"Bearer {BRIGHTDATA_API}",
                },
            )
            response_json = response.json()

        retries = 0
        while True:
            try:
                response = requests.get(
                    f"https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}",
                    headers={
                        "Authorization": f"Bearer {BRIGHTDATA_API}",
                    },
                    params={
                        "format": "json",
                    },
                )
                response_json = response.json()
                profile_search_results = pd.DataFrame(response_json)
                break

            except (requests.RequestException, ValueError, json.JSONDecodeError) as err:
                retries += 1
                if retries > MAX_RETRIES:
                    raise RuntimeError(
                        f"Failed to retrieve snapshot after {MAX_RETRIES} attempts: {err}"
                    )
                time.sleep(WAIT_TIME_BETWEEN_RETRIEVAL_REQUESTS)

        if "warning_code" in profile_search_results.columns:
            profile_search_results = profile_search_results[
                profile_search_results["warning_code"] != "dead_page"
            ].reset_index(drop=True)
        if "error_code" in profile_search_results.columns:
            profile_search_results = profile_search_results[
                profile_search_results["error_code"] != "crawl_failed"
            ].reset_index(drop=True)

    else:  # Perform local search
        local_profile_search = pd.read_csv(local_file)
        local_profile_search["create_time_processed"] = pd.to_datetime(
            local_profile_search["create_time"], utc=True
        )
        profile_search_results = pd.DataFrame()

        for profile in tqdm(profile_list):
            # Filter by account id, and post start and end date
            filtered_profile_search = local_profile_search[
                (local_profile_search["account_id"] == profile)
                & (
                    local_profile_search["create_time_processed"]
                    >= pd.to_datetime(start_date, utc=True)
                )
                & (
                    local_profile_search["create_time_processed"]
                    < pd.to_datetime(end_date, utc=True)
                )
            ].reset_index(drop=True)

            if filtered_profile_search.empty:
                continue

            profile_search_results = pd.concat(
                [
                    profile_search_results,
                    filtered_profile_search.drop(columns=["create_time_processed"]),
                ],
                ignore_index=True,
            )

    profile_search_results.to_csv(
        os.path.join(base_dir, "../data", project_name, execution_date, output_file),
        index=False,
    )

    return profile_search_results


def perform_tiktok_profile_metadata_search(
    project_name: str,
    execution_date: str,
    input_file: str,
    output_file: str = "",
    local_file: str = None,
) -> pd.DataFrame:
    # Create the project subfolder within the data folder if it does not exist
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(base_dir, "../data"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "../data", project_name), exist_ok=True)
    os.makedirs(
        os.path.join(base_dir, "../data", project_name, execution_date), exist_ok=True
    )

    # Define list of profiles for search
    profile_data = pd.read_csv(
        os.path.join(base_dir, "../data", project_name, input_file)
    )
    assert (
        "account_id" in profile_data.columns
    ), "Input file must contain 'account_id' column."
    profile_list = list(set(profile_data["account_id"].tolist()))

    if local_file is None:  # Perform API search
        # Initialise profile metadata search job
        data = [
            {"url": f"https://www.tiktok.com/@{profile}", "country": ""}
            for profile in profile_list
        ]
        response = requests.post(
            "https://api.brightdata.com/datasets/v3/trigger",
            headers={
                "Authorization": f"Bearer {BRIGHTDATA_API}",
                "Content-Type": "application/json",
            },
            params={
                "dataset_id": "gd_l1villgoiiidt09ci",
                "include_errors": "true",
            },
            json=data,
        )
        snapshot_id = response.json().get("snapshot_id")

        # Retrieve profile metadata search results
        response_json = {"status": "running"}
        while response_json.get("status") != "ready":
            time.sleep(WAIT_TIME_BETWEEN_RETRIEVAL_REQUESTS)
            response = requests.get(
                f"https://api.brightdata.com/datasets/v3/progress/{snapshot_id}",
                headers={
                    "Authorization": f"Bearer {BRIGHTDATA_API}",
                },
            )
            response_json = response.json()

        retries = 0
        while True:
            try:
                response = requests.get(
                    f"https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}",
                    headers={
                        "Authorization": f"Bearer {BRIGHTDATA_API}",
                    },
                    params={
                        "format": "json",
                    },
                )
                response_json = response.json()
                profile_metadata = pd.DataFrame(response_json)
                break

            except (requests.RequestException, ValueError, json.JSONDecodeError) as err:
                retries += 1
                if retries > MAX_RETRIES:
                    raise RuntimeError(
                        f"Failed to retrieve snapshot after {MAX_RETRIES} attempts: {err}"
                    )
                time.sleep(WAIT_TIME_BETWEEN_RETRIEVAL_REQUESTS)

        if "warning_code" in profile_metadata.columns:
            profile_metadata = profile_metadata[
                profile_metadata["warning_code"] != "dead_page"
            ].reset_index(drop=True)
        if "error_code" in profile_metadata.columns:
            profile_metadata = profile_metadata[
                profile_metadata["error_code"] != "crawl_failed"
            ].reset_index(drop=True)

    else:  # Perform local search
        local_profile_metadata = pd.read_csv(local_file)
        profile_metadata = local_profile_metadata[
            local_profile_metadata["account_id"].isin(profile_list)
        ].reset_index(drop=True)

    profile_metadata.to_csv(
        os.path.join(base_dir, "../data", project_name, execution_date, output_file),
        index=False,
    )

    return profile_metadata


def perform_x_keyword_search(
    project_name: str,
    execution_date: str,
    search_terms: list,
    output_file: str,
    num_posts_per_keyword: int,
) -> pd.DataFrame:
    def batched(iterable, n):
        """Yield successive n-sized batches from iterable."""
        for i in range(0, len(iterable), n):
            yield iterable[i : i + n]

    # Create the project subfolder within the data folder if it does not exist
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(base_dir, "../data"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "../data", project_name), exist_ok=True)
    os.makedirs(
        os.path.join(base_dir, "../data", project_name, execution_date), exist_ok=True
    )

    # Perform keyword search in batches of 1 (due to limitations of API call)
    all_search_results = []
    for batch_terms in batched(search_terms, 1):
        print(batch_terms)
        try:
            response = requests.get(
                "https://abundance.it.com/get_tweets_by_search_term",
                params={
                    "search_term": batch_terms,
                    "or_operator": 1,
                    "max_tweets": num_posts_per_keyword * len(batch_terms),
                },
                auth=HTTPBasicAuth(X_API_USERNAME, X_API_PASSWORD),
            )
            all_search_results += response.json()
        except requests.exceptions.JSONDecodeError:
            warnings.warn(
                f"JSONDecodeError encountered for search terms: {batch_terms}. Skipping these terms."
            )
            continue

    keyword_search_results = pd.DataFrame(all_search_results)
    keyword_search_results = keyword_search_results.drop_duplicates(
        subset="id"
    ).reset_index(drop=True)
    keyword_search_results["account_id"] = keyword_search_results["author"].apply(
        lambda x: x.get("userName")
    )
    keyword_search_results["hashtags"] = keyword_search_results["entities"].apply(
        extract_hashtags
    )
    keyword_search_results["tagged_users"] = keyword_search_results["entities"].apply(
        extract_tagged_users
    )
    keyword_search_results.to_csv(
        os.path.join(base_dir, "../data", project_name, execution_date, output_file),
        index=False,
    )

    return keyword_search_results


def perform_x_profile_search(
    project_name: str,
    execution_date: str,
    input_file: str,
    output_file: str,
    start_date: str,
    end_date: str,
    num_posts_per_profile: int,
    local_file: str = None,
    historical_post_file: str = None,
) -> pd.DataFrame:
    # Create the project subfolder within the data folder if it does not exist
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(base_dir, "../data"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "../data", project_name), exist_ok=True)
    os.makedirs(
        os.path.join(base_dir, "../data", project_name, execution_date), exist_ok=True
    )

    # Define search parameters
    profile_list = pd.read_csv(
        os.path.join(base_dir, "../data", project_name, input_file)
    )["account_id"].tolist()

    # Peform profile search
    if local_file is None:  # Perform API search
        response_list = []
        for profile in tqdm(profile_list):
            attempt = 0

            while attempt < MAX_RETRIES:
                attempt += 1
                try:
                    response = requests.get(
                        "https://abundance.it.com/get_tweets",
                        params={
                            "user": profile,
                            "max_tweets_per_user": num_posts_per_profile,
                            "cut_off_time": f"{start_date}T00:00:00",  # YYYY-MM-DDTHH:MM:SS
                        },
                        auth=HTTPBasicAuth(X_API_USERNAME, X_API_PASSWORD),
                    )
                    response_list += response.json()[0]
                    time.sleep(3)
                    break

                except requests.exceptions.JSONDecodeError:
                    warnings.warn(
                        f"JSONDecodeError for profile {profile} (attempt {attempt}/{MAX_RETRIES}). Retrying..."
                    )
                except requests.exceptions.ReadTimeout:
                    warnings.warn(
                        f"ReadTimeout for profile {profile} (attempt {attempt}/{MAX_RETRIES}). Retrying..."
                    )
                except requests.exceptions.ConnectTimeout:
                    warnings.warn(
                        f"ConnectTimeout for profile {profile} (attempt {attempt}/{MAX_RETRIES}). Retrying..."
                    )
                except requests.exceptions.HTTPError as e:
                    warnings.warn(
                        f"HTTP error for profile {profile}: {e}. Skipping profile."
                    )
                    break
                except requests.exceptions.RequestException as e:
                    warnings.warn(
                        f"RequestException for profile {profile}: {e}. Retrying (attempt {attempt}/{MAX_RETRIES})..."
                    )

            else:
                warnings.warn(
                    f"Failed to fetch info for profile {profile} after {MAX_RETRIES} attempts. Skipping."
                )

        profile_search_results = pd.DataFrame([r for r in response_list if r])
        profile_search_results["account_id"] = profile_search_results["author"].apply(
            lambda x: x.get("userName")
        )
        profile_search_results["hashtags"] = profile_search_results["entities"].apply(
            extract_hashtags
        )
        profile_search_results["tagged_users"] = profile_search_results[
            "entities"
        ].apply(extract_tagged_users)

        # Filter posts that happen before start_date
        profile_search_results["createdAt"] = pd.to_datetime(
            profile_search_results["createdAt"], format="%a %b %d %H:%M:%S %z %Y"
        )
        profile_search_results = profile_search_results[
            profile_search_results["createdAt"]
            >= datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        ].reset_index(drop=True)

    else:  # Perform local search
        local_profile_search = pd.read_csv(local_file)
        local_profile_search["create_time_processed"] = pd.to_datetime(
            local_profile_search["createdAt"], utc=True
        )
        profile_search_results = pd.DataFrame()

        for profile in tqdm(profile_list):
            # Filter by account id, and post start and end date
            filtered_profile_search = local_profile_search[
                (local_profile_search["account_id"] == profile)
                & (
                    local_profile_search["create_time_processed"]
                    >= pd.to_datetime(start_date, utc=True)
                )
                & (
                    local_profile_search["create_time_processed"]
                    < pd.to_datetime(end_date, utc=True)
                )
            ].reset_index(drop=True)

            if filtered_profile_search.empty:
                continue

            profile_search_results = pd.concat(
                [
                    profile_search_results,
                    filtered_profile_search.drop(columns=["create_time_processed"]),
                ],
                ignore_index=True,
            )

    profile_search_results.to_csv(
        os.path.join(base_dir, "../data", project_name, execution_date, output_file),
        index=False,
    )

    if historical_post_file:
        historical_post_file_path = os.path.join(
            base_dir, "../data", project_name, execution_date, historical_post_file
        )
        historical_posts = pd.read_csv(historical_post_file_path, on_bad_lines="skip")
        historical_posts = (
            pd.concat(
                [historical_posts, profile_search_results],
                ignore_index=True,
            )
            .drop_duplicates(subset="id", keep="last")
            .reset_index(drop=True)
        )
        historical_posts.to_csv(historical_post_file_path, index=False)

    return profile_search_results


def perform_x_profile_metadata_search(
    project_name: str,
    execution_date: str,
    input_file: str,
    output_file: str = "",
    local_file: str = None,
) -> pd.DataFrame:
    # Create the project subfolder within the data folder if it does not exist
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(base_dir, "../data"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "../data", project_name), exist_ok=True)
    os.makedirs(
        os.path.join(base_dir, "../data", project_name, execution_date), exist_ok=True
    )

    # Define list of profiles for search
    profile_data = pd.read_csv(
        os.path.join(base_dir, "../data", project_name, input_file)
    )
    assert (
        "account_id" in profile_data.columns
    ), "Input file must contain 'account_id' column."
    profile_list = list(set(profile_data["account_id"].tolist()))

    if local_file is None:  # Perform API search
        # Perform profile metadata search
        response_list = []
        for profile in tqdm(profile_list):
            attempt = 0

            while attempt < MAX_RETRIES:
                attempt += 1
                try:
                    response = requests.get(
                        "https://abundance.it.com/get_user_info",
                        params={
                            "user": profile,
                        },
                        auth=HTTPBasicAuth(X_API_USERNAME, X_API_PASSWORD),
                    )
                    response_list += response.json()
                    time.sleep(3)
                    break

                except requests.exceptions.JSONDecodeError:
                    warnings.warn(
                        f"JSONDecodeError for profile {profile} (attempt {attempt}/{MAX_RETRIES}). Retrying..."
                    )
                except requests.exceptions.ReadTimeout:
                    warnings.warn(
                        f"ReadTimeout for profile {profile} (attempt {attempt}/{MAX_RETRIES}). Retrying..."
                    )
                except requests.exceptions.ConnectTimeout:
                    warnings.warn(
                        f"ConnectTimeout for profile {profile} (attempt {attempt}/{MAX_RETRIES}). Retrying..."
                    )
                except requests.exceptions.HTTPError as e:
                    warnings.warn(
                        f"HTTP error for profile {profile}: {e}. Skipping profile."
                    )
                    break
                except requests.exceptions.RequestException as e:
                    warnings.warn(
                        f"RequestException for profile {profile}: {e}. Retrying (attempt {attempt}/{MAX_RETRIES})..."
                    )

            else:
                warnings.warn(
                    f"Failed to fetch info for profile {profile} after {MAX_RETRIES} attempts. Skipping."
                )

        profile_metadata = pd.DataFrame([r for r in response_list if r])
        profile_metadata.rename(columns={"userName": "account_id"}, inplace=True)

    else:  # Perform local search
        local_profile_metadata = pd.read_csv(local_file)
        profile_metadata = local_profile_metadata[
            local_profile_metadata["account_id"].isin(profile_list)
        ].reset_index(drop=True)

    profile_metadata.to_csv(
        os.path.join(base_dir, "../data", project_name, execution_date, output_file),
        index=False,
    )

    return profile_metadata

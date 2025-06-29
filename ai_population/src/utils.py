import pandas as pd
import os
import ast
import yt_dlp
import time
import json
import re
from tqdm import tqdm

tqdm.pandas()

from pydub import AudioSegment
from apify_client import ApifyClient
from openai import OpenAI
from ai_population.prompts.prompt_template import (
    tiktok_video_prompt_template,
    x_tweet_prompt_template,
    tiktok_profile_prompt_template,
    finfluencer_interview_user_prompt,
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


def optimize_audio_file(input_file_path: str, output_file_path: str) -> None:
    """
    Optimize an audio file by downsampling it to 16 kHz and converting it to mono.

    Args:
        input_file_path (str): The path to the input audio file.
        output_file_path (str): The path where the optimized audio file will be saved.

    Returns:
        None
    """
    # Load the audio file
    audio = AudioSegment.from_file(input_file_path)

    # Downsample the audio to 16 kHz and convert to mono
    audio = audio.set_frame_rate(16000).set_channels(1)

    # Export the optimized audio file
    audio.export(output_file_path, format="wav")


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

    try:
        with open(input_file_path, "rb") as audio_file:
            transcription = openai_client.audio.transcriptions.create(
                model="whisper-1", file=audio_file, response_format="text"
            )
        return transcription

    except FileNotFoundError:
        return None

    except Exception as e:
        if e.status_code == 413:
            print(
                f"Error: File {row['video_filename']} is too large to process. Optimizing the audio file..."
            )
            # Optimize the audio file
            optimize_audio_file(input_file_path, optimized_file_path)
            try:
                with open(optimized_file_path, "rb") as audio_file:
                    transcription = openai_client.audio.transcriptions.create(
                        model="whisper-1", file=audio_file, response_format="text"
                    )
                return transcription
            except Exception as e:
                print(
                    f"Error: File {optimized_file_path} is still too large after optimisation: {e}"
                )
                return None
        else:
            print(f"Error encountered when transcribing {row['video_filename']}: {e}")
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
    if interview_type.startswith("tiktok_finfluencer"):
        profile_args = {
            "profile_image": row["profile_pic_url"],
            "profile_name": row["account_id"],
            "profile_nickname": row["nickname"],
            "profile_biography": row["biography"],
            "profile_signature": row["signature"],
            "profile_bio_link": row["bio_link"],
            "profile_url": row["url"],
            "profile_lang": row["predicted_lang"],
            "profile_creation": row["create_time"],
            "verified_status": row["is_verified"],
            "num_followers": row["followers"],
            "num_following": row["following"],
            "num_likes": row["likes"],
            "num_videos": row["videos_count"],
            "num_digg": row["digg_count"],
            "private_account": row["is_private"],
            "region": row["region"],
            "tiktok_seller": row["is_commerce_user"],
            "awg_engagement_rate": row["awg_engagement_rate"],
            "comment_engagement_rate": row["comment_engagement_rate"],
            "like_engagement_rate": row["like_engagement_rate"],
            "video_transcripts": row["posts_combined"],
        }
    elif interview_type.startswith("x_finfluencer"):
        profile_args = {
            "profile_picture": row["profilePicture"],
            "name": row["name"],
            "account_id": row["account_id"],
            "location": row["location"],
            "description": row["description"],
            "url": row["url"],
            "created_at": row["createdAt"],
            "is_verified": row["isVerified"],
            "is_blue_verified": row["isBlueVerified"],
            "protected": row["protected"],
            "followers": row["followers"],
            "following": row["following"],
            "statuses_count": row["statusesCount"],
            "favourites_count": row["favouritesCount"],
            "media_count": row["mediaCount"],
            "tweets": row["posts_combined"],
        }

    else:
        profile_args = {}

    if interview_type in [
        "tiktok_finfluencer_onboarding",
    ]:
        additional_args = {
            "expert_reflection_portfoliomanager": row[
                "tiktok_finfluencer_expert_reflection_portfoliomanager_response"
            ],
            "expert_reflection_investmentadvisor": row[
                "tiktok_finfluencer_expert_reflection_investmentadvisor_response"
            ],
            "expert_reflection_financialanalyst": row[
                "tiktok_finfluencer_expert_reflection_financialanalyst_response"
            ],
            "expert_reflection_economist": row[
                "tiktok_finfluencer_expert_reflection_economist_response"
            ],
        }
        profile_args.update(additional_args)
    elif interview_type in [
        "x_finfluencer_onboarding",
    ]:
        additional_args = {
            "expert_reflection_portfoliomanager": row[
                "x_finfluencer_expert_reflection_portfoliomanager_response"
            ],
            "expert_reflection_investmentadvisor": row[
                "x_finfluencer_expert_reflection_investmentadvisor_response"
            ],
            "expert_reflection_financialanalyst": row[
                "x_finfluencer_expert_reflection_financialanalyst_response"
            ],
            "expert_reflection_economist": row[
                "x_finfluencer_expert_reflection_economist_response"
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
        "tiktok_finfluencer_interview",
        "x_finfluencer_interview",
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
            stock_mentions=row["stock_mentions"],
        )

    if interview_type in [
        "tiktok_finfluencer_stock_recommendation",
        "x_finfluencer_stock_recommendation",
    ]:
        # Load stock mentioned and reference to post
        return user_prompt_template.format(
            stock_name=row["stock_name"],
            stock_ticker=row["stock_ticker"],
            mention_date=row["mention_date"],
            post=row["post"],
        )

    else:
        return user_prompt_template


def extract_llm_responses(text, substring_exclusion_list: list = []) -> pd.Series:
    # Split the text by double newlines to separate different questions
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

    # Define regex patterns for each field
    question_pattern = r"\*\*question: (.*?)\*\*"
    explanation_pattern = r"\*\*explanation: (.*?)\*\*"
    symbol_pattern = r"\*\*symbol: (.*?)\*\*"
    category_pattern = r"\*\*category: (.*?)\*\*"
    speculation_pattern = r"\*\*speculation: (.*?)\*\*"
    value_pattern = r"\*\*value: (.*?)\*\*"
    response_pattern = r"\*\*response: (.*?)\*\*"

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

        questions_list.append(question.group(1).replace("”", "") if question else None)
        explanations_list.append(explanation.group(1) if explanation else None)
        symbols_list.append(symbol.group(1) if symbol else None)
        categories_list.append(category.group(1) if category else None)
        speculations_list.append(speculation.group(1) if speculation else None)
        values_list.append(value.group(1) if value else None)
        response_list.append(response.group(1) if response else None)

    # Create a DataFrame
    data = {
        "question": questions_list,
        "explanation": explanations_list,
        "symbol": symbols_list,
        "category": categories_list,
        "speculation": speculations_list,
        "value": values_list,
        "response": response_list,
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
        if not matching_cols:
            continue

        # Sort matching columns by null count (fewest nulls first)
        sorted_cols = sorted(matching_cols, key=lambda col: data[col].isna().sum())

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
    """
    Extracts structured stock recommendation information from a formatted string.

    The function parses a string containing stock recommendation details, where each field is denoted by a specific pattern (e.g., '**mentioned_by_finfluencer: ...**'), and returns the extracted fields as a pandas Series.

    Args:
        stock_recommendation_str (str): The input string containing stock recommendation details, formatted with specific markers for each field.

    Returns:
        pd.Series: A pandas Series with the following keys:
            - "mentioned_by_finfluencer": The name of the finfluencer who mentioned the stock, or None if not found.
            - "recommendation": The recommendation text, or None if not found.
            - "explanation": The explanation for the recommendation, or None if not found.
            - "confidence": The confidence level of the recommendation, or None if not found.
            - "virality": The virality score or description, or None if not found.
    """
    # Define regex patterns for each field
    mentioned_by_finfluencer_pattern = r"\*\*mentioned_by_finfluencer: (.*?)\*\*"
    recommendation_pattern = r"\*\*recommendation: (.*?)\*\*"
    explanation_pattern = r"\*\*explanation: (.*?)\*\*"
    confidence_pattern = r"\*\*confidence: (.*?)\*\*"
    virality_pattern = r"\*\*virality: (.*?)\*\*"

    # Extract the relevant fields from the stock recommendation string
    mentioned_by_finfluencer = re.search(
        mentioned_by_finfluencer_pattern, stock_recommendation_str, re.DOTALL
    )
    recommendation = re.search(
        recommendation_pattern, stock_recommendation_str, re.DOTALL
    )
    explanation = re.search(explanation_pattern, stock_recommendation_str, re.DOTALL)
    confidence = re.search(confidence_pattern, stock_recommendation_str, re.DOTALL)
    virality = re.search(virality_pattern, stock_recommendation_str, re.DOTALL)

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
        }
    )

    return stock_recommendation_series


def create_batch_file(
    prompts: pd.DataFrame,
    project_name: str,
    execution_date: str,
    gpt_model: str,
    system_prompt_field: str,
    user_prompt_field: str = "question_prompt",
    batch_file_name: str = "batch_input.jsonl",
) -> str:
    """
    Creates a batch file in JSON Lines format from a DataFrame of prompts.

    Args:
        prompts (pd.DataFrame): DataFrame containing the prompts data.
        project_name (str): The name of the project.
        execution_date (str): The date of the pipeline execution, used to create a unique directory name.
        system_prompt_field (str): The column name in the DataFrame for the system prompt content.
        user_prompt_field (str, optional): The column name in the DataFrame for the user prompt content. Defaults to "question_prompt".
        batch_file_name (str, optional): The name of the output batch file. Defaults to "batch_input.jsonl".

    Returns:
        str: The name of the created batch file.
    """
    # Creating an array of json tasks
    tasks = []
    for i in range(len(prompts)):
        task = {
            "custom_id": f'{prompts.loc[i, "custom_id"]}',
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": gpt_model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": prompts.loc[i, system_prompt_field]},
                    {"role": "user", "content": prompts.loc[i, user_prompt_field]},
                ],
            },
        }
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
) -> pd.DataFrame:
    """
    Executes a batch query using the OpenAI API and processes the results into a pandas DataFrame.

    Args:
        project_name (str): The name of the project.
        execution_date (str): The date of the pipeline execution, used to create a unique directory name.
        batch_input_file_dir (str): The directory path of the batch input file.
        batch_output_file_dir (str): The directory path where the batch output file will be saved.

    Returns:
        pd.DataFrame: A DataFrame containing the processed results from the batch query.
    """
    # Upload batch input file
    batch_file = openai_client.files.create(
        file=open(
            f"{base_dir}/../data/{project_name}/{execution_date}/batch-files/{batch_input_file_dir}",
            "rb",
        ),
        purpose="batch",
    )

    # Create batch job
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
                created_at=filtered_tweets.loc[i, "createdAt"],
                text=filtered_tweets.loc[i, "text"],
                like_count=filtered_tweets.loc[i, "likeCount"],
                view_count=filtered_tweets.loc[i, "viewCount"],
                retweet_count=filtered_tweets.loc[i, "retweetCount"],
                reply_count=filtered_tweets.loc[i, "replyCount"],
                quote_count=filtered_tweets.loc[i, "quoteCount"],
                bookmark_count=filtered_tweets.loc[i, "bookmarkCount"],
                lang=filtered_tweets.loc[i, "lang"],
                tagged_users=filtered_tweets.loc[i, "tagged_users"],
                hashtags=filtered_tweets.loc[i, "hashtags"],
            )
        ]

    return "\n\n".join(tweets_list)


def row_query(row: pd.Series, args: list) -> str:
    system_prompt = row[args[0]]
    user_prompt = row[args[1]]
    gpt_model = args[2]

    # Skip if system_prompt/user_prompt is empty or NaN (depending on your logic)
    if not isinstance(system_prompt, str) or not isinstance(user_prompt, str):
        return ""

    # Make a chat completion request
    try:
        response = openai_client.chat.completions.create(
            model=gpt_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )

        # Extract the assistant's response
        return response.choices[0].message.content

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
    batch_interview: bool = True,
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
        os.path.join(base_dir, "../data", project_name, execution_date, post_file)
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

    profile_metadata[f"{interview_type}_system_prompt"] = profile_metadata.apply(
        construct_system_prompt, args=(system_prompt_template, interview_type), axis=1
    )
    profile_metadata[f"{interview_type}_user_prompt"] = profile_metadata.apply(
        construct_user_prompt, args=(user_prompt_template, interview_type), axis=1
    )

    if batch_interview:
        # Generate custom ids
        if "custom_id" in profile_metadata.columns:
            profile_metadata.drop(columns="custom_id", inplace=True)

        profile_metadata = profile_metadata.reset_index(drop=False)
        profile_metadata.rename(columns={"index": "custom_id"}, inplace=True)

        # Create folder to contain batch files
        os.makedirs(
            os.path.join(
                base_dir, "../data", project_name, execution_date, "batch-files"
            ),
            exist_ok=True,
        )

        # Perform batch query for survey questions
        create_batch_file(
            profile_metadata,
            project_name=project_name,
            execution_date=execution_date,
            gpt_model=gpt_model,
            system_prompt_field=f"{interview_type}_system_prompt",
            user_prompt_field=f"{interview_type}_user_prompt",
            batch_file_name="batch_input.jsonl",
        )

        llm_responses = batch_query(
            project_name=project_name,
            execution_date=execution_date,
            batch_input_file_dir="batch_input.jsonl",
            batch_output_file_dir="batch_output.jsonl",
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
            os.path.join(
                base_dir, "../data", project_name, execution_date, output_file
            ),
            index=False,
        )

    else:
        profile_metadata[llm_response_field] = profile_metadata.progress_apply(
            row_query,
            args=(
                [
                    f"{interview_type}_system_prompt",
                    f"{interview_type}_user_prompt",
                    gpt_model,
                ],
            ),
            axis=1,
        )

        # Save profile metadata after analysis into CSV file
        profile_metadata.to_csv(
            os.path.join(
                base_dir, "../data", project_name, execution_date, output_file
            ),
            index=False,
        )


def perform_profile_interview_shorten(
    project_name: str,
    execution_date: str,
    gpt_model: str,
    profile_metadata_input_file: str,
    profile_metadata_output_file: str,
    system_prompt_field: str,
    user_prompt_field: str,
    llm_response_field: str,
    interview_type: str,
    batch_interview: bool = True,
) -> None:

    print("Loading profile metadata...")
    profile_metadata = pd.read_csv(
        f"{base_dir}/../data/{project_name}/{profile_metadata_input_file}"
    )

    print("Generate system and user prompts...")
    profile_metadata[system_prompt_field] = profile_metadata.apply(
        construct_system_prompt, args=(interview_type,), axis=1
    )
    profile_metadata[user_prompt_field] = profile_metadata.apply(
        construct_user_prompt, args=(interview_type,), axis=1
    )

    if batch_interview:
        # Generate custom ids
        if "custom_id" in profile_metadata.columns:
            profile_metadata.drop(columns="custom_id", inplace=True)

        profile_metadata = profile_metadata.reset_index(drop=False)
        profile_metadata.rename(columns={"index": "custom_id"}, inplace=True)

        # Create folder to contain batch files
        batch_file_dir = f"{base_dir}/../data/{project_name}/batch-files"
        os.makedirs(batch_file_dir, exist_ok=True)

        # Perform batch query for survey questions
        batch_file_dir = create_batch_file(
            profile_metadata,
            project_name=project_name,
            execution_date=execution_date,
            gpt_model=gpt_model,
            system_prompt_field=system_prompt_field,
            user_prompt_field=user_prompt_field,
            batch_file_name="batch_input.jsonl",
        )

        print("Perform batch query using OpenAI API...")
        llm_responses = batch_query(
            project_name=project_name,
            execution_date=execution_date,
            batch_input_file_dir="batch_input.jsonl",
            batch_output_file_dir="batch_output.jsonl",
        )
        llm_responses.rename(
            columns={"query_response": llm_response_field}, inplace=True
        )

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
        print("Saving profile metadata after interview...")
        profile_metadata_with_responses.to_csv(
            f"{base_dir}/../data/{project_name}/{profile_metadata_output_file}",
            index=False,
        )

    else:
        print("Querying the OpenAI Chat Completion API (one row at a time)...")
        profile_metadata[llm_response_field] = profile_metadata.progress_apply(
            row_query,
            args=([system_prompt_field, user_prompt_field, gpt_model],),
            axis=1,
        )

        # Save profile metadata after analysis into CSV file
        print("Saving profile metadata after interview...")
        profile_metadata.to_csv(
            f"{base_dir}/../data/{project_name}/{profile_metadata_output_file}",
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
            profile_image=row["avatar"],
            profile_name=row["profile"],
            profile_nickname=row["nickName"],
            verified_status=row["verified"],
            private_account=row["privateAccount"],
            region=row["region"],
            tiktok_seller=row["ttSeller"],
            profile_signature=row["signature"],
            num_followers=row["fans"],
            num_following=row["following"],
            num_likes=row["heart"],
            num_videos=row["video"],
            num_digg=row["digg"],
            total_likes_over_num_followers=calculate_profile_engagement(
                row["heart"], row["fans"]
            ),
            total_likes_over_num_videos=calculate_profile_engagement(
                row["heart"], row["video"]
            ),
            video_transcripts=row["posts_combined"],
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
    s
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
        video_metadata.dropna(subset=["post_id"], inplace=True)
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

    # Download videos that have not been transcribed and perform transcription
    video_metadata_without_transcript.progress_apply(
        download_video, args=(project_name, execution_date), axis=1
    )
    video_metadata_without_transcript["video_transcript"] = (
        video_metadata_without_transcript.progress_apply(
            transcribe_videos, args=(project_name, execution_date), axis=1
        )
    )

    # Merge newly transcribed videos with existing video metadata
    video_metadata_with_transcript = video_metadata[
        ~video_metadata["video_transcript"].isnull()
    ].reset_index(drop=True)
    video_metadata = pd.concat(
        [video_metadata_with_transcript, video_metadata_without_transcript],
        ignore_index=True,
    )
    video_metadata.to_csv(video_metadata_path, index=False)

    # Clean up downloaded videos to save disk space
    for file in os.listdir(video_download_folder_path):
        file_path = os.path.join(video_download_folder_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)


def update_verified_profile_pool(
    project_name: str,
    execution_date: str,
    input_file_path: str,
    verified_profile_pool: str,
    prediction_threshold: float,
) -> None:
    """
    Updates the verified profile pool by adding new financial influencers identified from the onboarding interview data.

    This function reads the interviewed profiles and the existing verified profile pool from CSV files, filters out profiles that meet or exceed the specified finfluencer likelihood threshold and have stock recommendations, and appends these new profiles to the verified profile pool. The updated pool is then saved back to the CSV file.

    Args:
        project_name (str): Name of the project directory containing the data.
        execution_date (str): Date of execution, used to locate the input file and as the inclusion date for new profiles.
        input_file_path (str): Relative path to the CSV file containing interviewed profiles.
        verified_profile_pool (str): Filename of the verified profile pool CSV.
        prediction_threshold (float): Minimum likelihood score to consider a profile as a financial influencer.

    Returns:
        None
    """
    interviewed_profiles = pd.read_csv(
        os.path.join(base_dir, "../data", project_name, execution_date, input_file_path)
    )
    verified_profiles = pd.read_csv(
        os.path.join(base_dir, "../data", project_name, verified_profile_pool)
    )

    # Filter out financial influencers identified during onboarding interview
    finfluencer_likelihood_col = "Indicate on a scale of 0 to 100, how likely this creator is a finfluencer (0 means most definitely not a finfluencer and 100 means most definitely a finfluencer)? - value"
    interviewed_profiles[finfluencer_likelihood_col] = interviewed_profiles[
        finfluencer_likelihood_col
    ].astype(float)
    finfluencer_profiles = interviewed_profiles[
        interviewed_profiles[finfluencer_likelihood_col] >= prediction_threshold
    ].reset_index(drop=True)

    # Filter out financial influencers that had a stock recommendation
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
                            "Indicate on a scale of 0 to 100, how influential this influencer is (0 means not at all influential and 100 means very influential with millions of followers and mainstream recognition)? - value"
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

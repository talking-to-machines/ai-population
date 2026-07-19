import os
import pandas as pd
import argparse
from tqdm import tqdm
import json
from datetime import datetime

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
    PROFILE_SEARCH_TODAY,
    MAX_NUM_POSTS_PER_KEYWORD,
    NUM_POSTS_PER_PROFILE_FROM_KEYWORD_SEARCH,
    NUM_POSTS_PER_PROFILE,
    VOTER_DEMOGRAPHIC_INTERVIEW_REGEX_PATTERNS,
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
    x_jointllm_voter_demographic_interview_system_prompt,
    tiktok_jointllm_voter_demographic_interview_system_prompt,
    jointllm_voter_demographic_interview_user_prompt,
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
    model_name: str = GPT_MODEL,
    together_ai_endpoint: str = None,
    grok_endpoint: str = None,
) -> None:
    perform_profile_interview(
        project_name=project_name,
        execution_date=execution_date,
        model_name=model_name,
        profile_metadata_file=profile_metadata_file,
        post_file=post_file,
        output_file=output_file,
        system_prompt_template=system_prompt_template,
        user_prompt_template=user_prompt_template,
        llm_response_field="jointllm_voter_demographic_interview_llm_response",
        interview_type=interview_type,
        together_ai_endpoint=together_ai_endpoint,
        grok_endpoint=grok_endpoint,
    )

    # Append raw interview results to a cumulative ledger so the keyword-search
    # step in subsequent loop iterations can skip already-interviewed profiles.
    output_path = os.path.join(
        base_dir, "../data", project_name, execution_date, output_file
    )
    ledger_path = os.path.join(
        base_dir, "../data", project_name, execution_date, f"raw_{output_file}"
    )
    new_results = pd.read_csv(output_path)
    if os.path.exists(ledger_path):
        prior_ledger = pd.read_csv(ledger_path)
        pd.concat([prior_ledger, new_results], ignore_index=True).to_csv(
            ledger_path, index=False
        )
    else:
        new_results.to_csv(ledger_path, index=False)

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
        VOTER_DEMOGRAPHIC_INTERVIEW_REGEX_PATTERNS,
    )

    # Bucket the smaller-party VOTE_FEDERAL symbols (VO_FEDERAL_7..VO_FEDERAL_15)
    # into a single VO_FEDERAL_7-15 category so they align with the
    # stratification frame. Written to a NEW column so the original
    # 'VOTE_FEDERAL - symbol' value from the LLM is preserved for analysis.
    if "VOTE_FEDERAL - symbol" in post_interview_profile_metadata.columns:
        small_party_symbols = {f"VO_FEDERAL_{i}" for i in range(7, 16)}
        post_interview_profile_metadata["VOTE_FEDERAL - symbol_bucketed"] = (
            post_interview_profile_metadata["VOTE_FEDERAL - symbol"].apply(
                lambda s: "VO_FEDERAL_7-15" if s in small_party_symbols else s
            )
        )

    # Bucket EDUCATION symbols (EDU1..EDU9) into three coarse classes used by
    # the stratification frame: EDU_COMPULSORY (EDU1-3), EDU_SECONDARY (EDU4-7),
    # EDU_TERTIARY (EDU8-9). Written to a NEW column so the original
    # 'EDUCATION - symbol' value from the LLM is preserved for analysis.
    if "EDUCATION - symbol" in post_interview_profile_metadata.columns:
        education_buckets = {
            **{f"EDU{i}": "EDU_COMPULSORY" for i in range(1, 4)},
            **{f"EDU{i}": "EDU_SECONDARY" for i in range(4, 8)},
            **{f"EDU{i}": "EDU_TERTIARY" for i in range(8, 10)},
        }
        post_interview_profile_metadata["EDUCATION - symbol_bucketed"] = (
            post_interview_profile_metadata["EDUCATION - symbol"].apply(
                lambda s: education_buckets.get(s, s)
            )
        )

    # Filter out profiles that are non-individuals (entity inclusion criteria)
    filtered_profile_metadata = post_interview_profile_metadata[
        post_interview_profile_metadata["ENTITY - symbol"] == "ENT1"
    ].reset_index(drop=True)

    # Filter out profiles that are not based in Swiss (geographic inclusion criteria)
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
    # Load target stratification frame (expects a 'cell_id' identifier and a
    # per-row 'count' target; 'target_share' is bookkeeping metadata and
    # ignored by the matching loop).
    target_stratification_df = pd.read_csv(
        os.path.join(base_dir, "../data", project_name, target_stratification_frame)
    )
    required_cols = {"cell_id", "count"}
    missing = required_cols - set(target_stratification_df.columns)
    if missing:
        raise ValueError(
            f"Target stratification frame {target_stratification_frame} is missing "
            f"required column(s): {sorted(missing)}."
        )

    # Load or initialize current stratification frame
    current_strat_path = os.path.join(
        base_dir, "../data", project_name, execution_date, current_stratification_frame
    )
    if os.path.exists(current_strat_path):
        current_stratification_df = pd.read_csv(current_strat_path)
    else:
        current_stratification_df = target_stratification_df.copy()
        current_stratification_df["count"] = 0

    # Load input file (voter profiles with demographic attributes)
    voters_df = pd.read_csv(
        os.path.join(base_dir, "../data", project_name, execution_date, input_file)
    )

    # Stratification dimensions are all strat-frame columns except bookkeeping
    # ones. The strat frame stores them as bare names (e.g. 'GENDER'); the voter
    # dataframe stores the LLM answer under '{col} - symbol'.
    strat_dims = [
        col
        for col in target_stratification_df.columns
        if col not in ("count", "cell_id", "target_share")
    ]
    # VOTE_FEDERAL and EDUCATION are matched against bucketed columns
    # (VO_FEDERAL_7..15 → VO_FEDERAL_7-15; EDU1..9 → EDU_COMPULSORY /
    # EDU_SECONDARY / EDU_TERTIARY); other dims use the raw LLM symbol.
    bucketed_dims = {"VOTE_FEDERAL", "EDUCATION"}
    voter_cols = {
        col: (f"{col} - symbol_bucketed" if col in bucketed_dims else f"{col} - symbol")
        for col in strat_dims
    }

    # Map each voter to a cell in the stratification frame
    eligible_voters = []
    for _, voter in voters_df.iterrows():
        # Skip voter if any strat-dim symbol is missing from their data
        if not all(voter_cols[col] in voter.index for col in strat_dims):
            continue

        # Find the matching cell based on strat-dim symbols
        mask = pd.Series(True, index=target_stratification_df.index)
        for col in strat_dims:
            mask = mask & (
                target_stratification_df[col].astype(str) == str(voter[voter_cols[col]])
            )

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
            current_stratification_df.loc[cell_idx, "count"]
            >= target_stratification_df.loc[cell_idx, "count"]
        ):
            continue

        # Assign voter to cell, increment count, and stamp the cell_id of the
        # matched cell on the voter so they can be traced back.
        current_stratification_df.loc[cell_idx, "count"] += 1
        voter_with_cell = voter.copy()
        voter_with_cell["cell_id"] = target_stratification_df.loc[cell_idx, "cell_id"]
        eligible_voters.append(voter_with_cell)

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
        current_stratification_df["count"] == target_stratification_df["count"]
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
    model_name: str = GPT_MODEL,
    together_ai_endpoint: str = None,
    grok_endpoint: str = None,
) -> None:
    perform_profile_interview(
        project_name=project_name,
        execution_date=execution_date,
        model_name=model_name,
        profile_metadata_file=profile_metadata_file,
        post_file=post_file,
        output_file=output_file,
        system_prompt_template=system_prompt_template,
        user_prompt_template=user_prompt_template,
        llm_response_field="jointllm_voter_digital_polling_llm_response",
        interview_type=interview_type,
        history_field="history",
        together_ai_endpoint=together_ai_endpoint,
        grok_endpoint=grok_endpoint,
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
    post_interview_results["model"] = model_name

    # Save formatted interview results
    post_interview_results.to_csv(
        os.path.join(base_dir, "../data", project_name, execution_date, output_file),
        index=False,
    )


def filter_keyword_search_by_ledger(
    project_name: str,
    execution_date: str,
    keyword_search_file: str,
    demographic_interview_file: str,
) -> None:
    """Drop keyword-search rows whose author was already interviewed in a prior
    iteration, so downstream profile metadata / profile search / video
    transcription / demographic interview API calls are skipped for them.
    """
    data_dir = os.path.join(base_dir, "../data", project_name, execution_date)
    ledger_path = os.path.join(data_dir, f"raw_{demographic_interview_file}")
    if not os.path.exists(ledger_path):
        return

    keyword_search_path = os.path.join(data_dir, keyword_search_file)
    keyword_search_df = pd.read_csv(keyword_search_path)
    interviewed_ids = set(pd.read_csv(ledger_path)["account_id"].astype(str))
    before = len(keyword_search_df)
    keyword_search_df = keyword_search_df[
        ~keyword_search_df["account_id"].astype(str).isin(interviewed_ids)
    ].reset_index(drop=True)
    keyword_search_df.to_csv(keyword_search_path, index=False)
    print(
        f"[idempotency] Dropped {before - len(keyword_search_df)} of {before} "
        f"keyword search rows for already-interviewed authors; "
        f"{len(keyword_search_df)} unseen-profile rows remain."
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
            "demographic_interview_file": VOTER_ENTITY_GEOGRAPHIC_EXCLUSION_CRITERIA_FILE_X,
            "demographic_interview_system_prompt": x_jointllm_voter_demographic_interview_system_prompt,
            "demographic_interview_user_prompt": jointllm_voter_demographic_interview_user_prompt,
            "demographic_interview_interview_type": "x_jointllm_voter_demographic_interview",
            "quota_inclusion_criteria_file": VOTER_QUOTA_INCLUSION_CRITERIA_FILE_X,
            "eligible_profile_posts_file": VOTER_ELIGIBLE_PROFILE_SEARCH_FILE_X,
            "target_stratification_frame": VOTER_TARGET_STRATIFICATION_FRAME_X,
            "current_stratification_frame": VOTER_CURRENT_STRATIFICATION_FRAME_X,
            "digital_polling_file": VOTER_DIGITAL_POLLING_FILE_X,
            "digital_polling_system_prompt": x_jointllm_voter_demographic_interview_system_prompt,
            "digital_polling_user_prompt": jointllm_voter_digital_polling_user_prompt,
            "digital_polling_interview_type": "x_jointllm_voter_digital_polling_interview",
            "profile_search_start_date": datetime.strptime(
                PROFILE_SEARCH_START_DATE, "%m-%d-%Y"
            ).strftime("%Y-%m-%d"),
            "profile_search_end_date": datetime.strptime(
                PROFILE_SEARCH_END_DATE, "%m-%d-%Y"
            ).strftime("%Y-%m-%d"),
            "profile_search_today": datetime.strptime(
                PROFILE_SEARCH_TODAY, "%m-%d-%Y"
            ).strftime("%Y-%m-%d"),
        }
    else:  # tiktok
        constants = {
            "project_name": PROJECT_NAME,
            "pipeline_name": VOTER_PIPELINE_TIKTOK,
            "search_term_list": VOTER_SEARCH_TERMS_TIKTOK,
            "keyword_search_file": VOTER_KEYWORD_SEARCH_FILE_TIKTOK,
            "keyword_profile_metadata_file": VOTER_KEYWORD_PROFILE_METADATA_FILE_TIKTOK,
            "keyword_profile_posts_file": VOTER_KEYWORD_PROFILE_POSTS_FILE_TIKTOK,
            "demographic_interview_file": VOTER_ENTITY_GEOGRAPHIC_EXCLUSION_CRITERIA_FILE_TIKTOK,
            "demographic_interview_system_prompt": tiktok_jointllm_voter_demographic_interview_system_prompt,
            "demographic_interview_user_prompt": jointllm_voter_demographic_interview_user_prompt,
            "demographic_interview_interview_type": "tiktok_jointllm_voter_demographic_interview",
            "quota_inclusion_criteria_file": VOTER_QUOTA_INCLUSION_CRITERIA_FILE_TIKTOK,
            "eligible_profile_posts_file": VOTER_ELIGIBLE_PROFILE_SEARCH_FILE_TIKTOK,
            "target_stratification_frame": VOTER_TARGET_STRATIFICATION_FRAME_TIKTOK,
            "current_stratification_frame": VOTER_CURRENT_STRATIFICATION_FRAME_TIKTOK,
            "digital_polling_file": VOTER_DIGITAL_POLLING_FILE_TIKTOK,
            "digital_polling_system_prompt": tiktok_jointllm_voter_demographic_interview_system_prompt,
            "digital_polling_user_prompt": jointllm_voter_digital_polling_user_prompt,
            "digital_polling_interview_type": "tiktok_jointllm_voter_digital_polling_interview",
            "profile_search_start_date": PROFILE_SEARCH_START_DATE,
            "profile_search_end_date": PROFILE_SEARCH_END_DATE,
            "profile_search_today": PROFILE_SEARCH_TODAY,
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
    parser.add_argument(
        "--model-name",
        type=str,
        default=GPT_MODEL,
        help="Model name to use for the interview. Pass an OpenAI model id (default) "
        "or a Together AI model id when --together-ai-endpoint is also set.",
    )
    endpoint_group = parser.add_mutually_exclusive_group()
    endpoint_group.add_argument(
        "--together-ai-endpoint",
        type=str,
        default=None,
        help="Together AI base URL (serverless or dedicated endpoint). When set, "
        "the interview is routed through Together AI instead of OpenAI.",
    )
    endpoint_group.add_argument(
        "--grok-endpoint",
        type=str,
        default=None,
        help="xAI (Grok) base URL, typically https://api.x.ai/v1. When set, "
        "the interview is routed through xAI instead of OpenAI. "
        "Mutually exclusive with --together-ai-endpoint.",
    )
    args = parser.parse_args()
    platform = args.platform
    model_name = args.model_name
    together_ai_endpoint = args.together_ai_endpoint
    grok_endpoint = args.grok_endpoint

    if platform not in ["x", "tiktok"]:
        raise ValueError(
            "Invalid platform specified. Supported platforms are 'x' and 'tiktok'."
        )

    # Step 0: Define pipeline constants based on social media platform (i.e., X vs. TikTok)
    constants = define_pipeline_constants(platform=platform)

    STRATIFICATION_FRAME_NOT_FILED = True
    iteration = 3
    # num_posts_per_keyword is reassigned at the top of every iteration. The 0
    # seed lets the while condition pass on the first check.
    num_posts_per_keyword = 0
    while (
        STRATIFICATION_FRAME_NOT_FILED
        and num_posts_per_keyword < MAX_NUM_POSTS_PER_KEYWORD
    ):
        # Ramp the keyword-search depth: 10 on the first iteration, +10 each
        # iteration, capped at MAX_NUM_POSTS_PER_KEYWORD (100). Keeps cost low early
        # and only widens the net if the stratification frame still isn't full.
        num_posts_per_keyword = min(10 * (iteration + 1), MAX_NUM_POSTS_PER_KEYWORD)
        print(
            f"Iteration {iteration + 1}: keyword search depth = {num_posts_per_keyword} posts/term"
        )

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
                num_posts_per_keyword=num_posts_per_keyword,
            )
        else:
            perform_tiktok_keyword_search(
                project_name=constants["project_name"],
                execution_date=constants["pipeline_name"],
                search_terms=constants["search_term_list"],
                output_file=constants["keyword_search_file"],
                num_posts_per_keyword=num_posts_per_keyword,
            )

        ## Skip authors already interviewed in a prior iteration so we do not
        ## re-pay for profile metadata, profile posts, video transcription,
        ## and demographic interview API calls on the same profiles.
        filter_keyword_search_by_ledger(
            project_name=constants["project_name"],
            execution_date=constants["pipeline_name"],
            keyword_search_file=constants["keyword_search_file"],
            demographic_interview_file=constants["demographic_interview_file"],
        )

        ## If every author from the keyword search has already been interviewed,
        ## the filter empties the CSV. Short-circuit to the next iteration so we
        ## widen the keyword-search depth instead of crashing downstream API
        ## calls that expect a non-empty 'account_id' column.
        keyword_search_path = os.path.join(
            base_dir,
            "../data",
            constants["project_name"],
            constants["pipeline_name"],
            constants["keyword_search_file"],
        )
        if pd.read_csv(keyword_search_path).empty:
            print(
                f"Iteration {iteration + 1}: no unseen authors in keyword search; "
                "widening pool on next iteration."
            )
            iteration += 1
            continue

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
                project_name=constants["project_name"],
                execution_date=constants["pipeline_name"],
                input_file=os.path.join(
                    constants["pipeline_name"],
                    constants["keyword_profile_metadata_file"],
                ),
                output_file=constants["keyword_profile_posts_file"],
                start_date=constants["profile_search_start_date"],
                end_date=constants["profile_search_today"],
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
                project_name=constants["project_name"],
                execution_date=constants["pipeline_name"],
                input_file=os.path.join(
                    constants["pipeline_name"],
                    constants["keyword_profile_metadata_file"],
                ),
                output_file=constants["keyword_profile_posts_file"],
                start_date=constants["profile_search_start_date"],
                end_date=constants["profile_search_end_date"],
                num_posts_per_profile=NUM_POSTS_PER_PROFILE_FROM_KEYWORD_SEARCH,
            )
            perform_video_transcription(
                project_name=constants["project_name"],
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
            output_file=constants["demographic_interview_file"],
            system_prompt_template=constants["demographic_interview_system_prompt"],
            user_prompt_template=constants["demographic_interview_user_prompt"],
            interview_type=constants["demographic_interview_interview_type"],
            model_name=model_name,
            together_ai_endpoint=together_ai_endpoint,
            grok_endpoint=grok_endpoint,
        )

        # Step 3: Apply quota inclusion criteria to identify eligible profiles for polling
        print(
            "Step 3: Apply quota inclusion criteria to identify eligible profiles for polling"
        )
        ## Apply quota inclusion criteria to identify eligible profiles for polling based on pre-defined stratification frame
        quota_inclusion_criteria_result = apply_quota_inclusion_criteria(
            project_name=constants["project_name"],
            execution_date=constants["pipeline_name"],
            input_file=constants["demographic_interview_file"],
            output_file=constants["quota_inclusion_criteria_file"],
            target_stratification_frame=constants["target_stratification_frame"],
            current_stratification_frame=constants["current_stratification_frame"],
        )
        # apply_quota_inclusion_criteria returns True when every cell is at quota,
        # so the loop should continue only while the frame is NOT yet filled.
        STRATIFICATION_FRAME_NOT_FILED = not quota_inclusion_criteria_result
        iteration += 1

    # Step 4: Extract posts from eligible profiles during polling period
    print("Step 4: Extract posts from eligible profiles during polling period")
    if platform == "x":
        profile_latest_videos = perform_x_profile_search(
            project_name=constants["project_name"],
            execution_date=constants["pipeline_name"],
            input_file=os.path.join(
                constants["pipeline_name"], constants["quota_inclusion_criteria_file"]
            ),
            output_file=constants["eligible_profile_posts_file"],
            start_date=constants["profile_search_start_date"],
            end_date=constants["profile_search_end_date"],
            num_posts_per_profile=NUM_POSTS_PER_PROFILE,
        )
    else:
        profile_latest_videos = perform_tiktok_profile_search(
            project_name=constants["project_name"],
            execution_date=constants["pipeline_name"],
            input_file=os.path.join(
                constants["pipeline_name"], constants["quota_inclusion_criteria_file"]
            ),
            output_file=constants["eligible_profile_posts_file"],
            start_date=constants["profile_search_start_date"],
            end_date=constants["profile_search_end_date"],
            num_posts_per_profile=NUM_POSTS_PER_PROFILE,
        )
        perform_video_transcription(
            project_name=constants["project_name"],
            execution_date=constants["pipeline_name"],
            video_file=constants["eligible_profile_posts_file"],
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
        model_name=model_name,
        together_ai_endpoint=together_ai_endpoint,
        grok_endpoint=grok_endpoint,
    )

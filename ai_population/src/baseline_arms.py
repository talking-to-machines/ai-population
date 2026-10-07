"""
Baseline arms for the X market-signals pipeline.

Builds the arm prompts from the day's production drop, parses the arm responses
and runs the QC that is appended to the baseline run log. The model calls are
made by perform_x_baseline_arms in market_signals_x.py.
"""

import hashlib
import json
import os
import random
import re
from datetime import datetime, time as dtime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ai_population.config.market_signals_config import (
    BASELINE_DIR_SUFFIX_X,
    BASELINE_RUN_LOG_FILE_X,
    BASELINE_ARMS_CELLS_X,
    BASELINE_TX81_ROSTER_X,
    BASELINE_BARE_SYSTEM_PROMPT_X,
    BASELINE_GENERIC_REPLACEMENT_X,
    BASELINE_PERSONA_HEADER_X,
    BASELINE_SEARCH_OFF_SPLICES_X,
    BASELINE_ARM_CLOCK_ET_X,
    BASELINE_MIN_YES_BATCH_SECONDS_X,
    BASELINE_MODULES_X,
    BASELINE_MODEL_PRICING_X,
    BASELINE_BATCH_DISCOUNT_X,
    BASELINE_TWIN_NO_COST_WARN_USD_X,
    BASELINE_QC_THRESHOLDS_X,
    FINFLUENCER_INTERVIEW_REGEX_PATTERNS,
    FINFLUENCER_DAILY_STOCK_PICK_REGEX_PATTERNS,
)
from ai_population.prompts.prompt_template import daily_stock_pick_user_prompts
from ai_population.src.utils import (
    LLM_ERROR_RESPONSE,
    LLM_SKIPPED_RESPONSE,
    _detect_provider,
    construct_user_prompt,
    create_batch_file,
    extract_llm_responses,
    coalesce_columns_by_regex,
)

base_dir = os.path.dirname(os.path.abspath(__file__))

ET = ZoneInfo("America/New_York")
MODULES = ("post_interview", "daily_stock_pick")
CONDITIONS = ("yes", "no")
ARM_RESPONSE_FIELD = "baseline_arm_response"
ARM_RAW_RESPONSE_FIELD = f"{ARM_RESPONSE_FIELD}_raw"
ARM_TIMESTAMP_COL = "baseline_arm_interview_datetime"
ARM_SYSTEM_PROMPT_COL = "arm_system_prompt"
EMPTY_POST_FILE = "x_baseline_empty_posts.csv"
STATUS_CODES = {"OK": 0, "WARN": 1, "FAIL": 2}
# Output cell name -> BASELINE_ARMS_CELLS_X key
CELL_CONFIG_KEYS = {
    "bare_yes": "bare_yes",
    "bare_no": "bare_no",
    "generic_yes": "generic_yes",
    "generic_no": "generic_no",
    "twin_no": "twin_no",
    "twin_yes_replicate": "twin_replicate",
    "twin_no_replicate": "twin_replicate",
}
PARSED_FIELDS = (
    "explanation",
    "symbol",
    "category",
    "speculation",
    "value",
    "response",
    "stock ticker",
    "recommendation",
    "confidence",
    "expected holding period",
    "primary catalyst type",
)
TIMING_SEVERITY = {"clean": 0, "unknown": 1, "contaminated": 2, "lost": 3}


class BaselineArmsError(Exception):
    """Raised when the baseline arms cannot run as specified (a FAIL status)."""


def sha16(text: str) -> str:
    """
    Hashes a text for the prompt manifest and QC checks.

    Args:
        text (str): Text to hash.

    Returns:
        str: The first 16 hex characters of the SHA256 digest.
    """
    return hashlib.sha256(str(text).encode("utf-8", "replace")).hexdigest()[:16]


def baseline_execution_date(execution_date: str) -> str:
    """
    Gets the name of the baseline folder for a drop date.

    Args:
        execution_date (str): Drop date in DD-MM-YYYY format.

    Returns:
        str: The baseline folder name, e.g. "06-10-2026-baseline".
    """
    return f"{execution_date}{BASELINE_DIR_SUFFIX_X}"


def data_path(project_name: str, *parts: str) -> str:
    """
    Builds a path inside a project's data folder.

    Args:
        project_name (str): Name of the project directory.
        *parts (str): Path components below the project folder.

    Returns:
        str: The joined path.
    """
    return os.path.join(base_dir, "../data", project_name, *parts)


def module_chunks(module: str) -> list:
    """
    Lists the chunk numbers of a module's battery.

    Args:
        module (str): "post_interview" or "daily_stock_pick".

    Returns:
        list: Chunk numbers, starting at 1.
    """
    if module == "daily_stock_pick":
        return list(range(1, len(daily_stock_pick_user_prompts) + 1))
    return [1]


def arm_interview_type(module: str, condition: str, chunk: int) -> str:
    """
    Gets the interview type of an arm call. Stock-pick types keep the production
    prefix so that construct_user_prompt renders the battery as it does for production.

    Args:
        module (str): "post_interview" or "daily_stock_pick".
        condition (str): "yes" (web search on) or "no" (web search off).
        chunk (int): Chunk number of the battery.

    Returns:
        str: The interview type.
    """
    if module == "daily_stock_pick":
        return f"x_finfluencer_daily_stock_pick_baseline_{condition}_{chunk}"
    return f"x_baseline_post_interview_{condition}"


def arm_input_file(module: str, condition: str, execution_date: str) -> str:
    """
    Gets the name of the arm input CSV for a module and condition.

    Args:
        module (str): "post_interview" or "daily_stock_pick".
        condition (str): "yes" or "no".
        execution_date (str): Drop date in DD-MM-YYYY format.

    Returns:
        str: The file name.
    """
    return f"x_baseline_{module}_{condition}_input_{execution_date}.csv"


def arm_output_file(
    module: str, condition: str, chunk: int, execution_date: str
) -> str:
    """
    Gets the name of the interview output CSV for a module, condition and chunk.

    Args:
        module (str): "post_interview" or "daily_stock_pick".
        condition (str): "yes" or "no".
        chunk (int): Chunk number of the battery.
        execution_date (str): Drop date in DD-MM-YYYY format.

    Returns:
        str: The file name.
    """
    return f"x_baseline_{module}_{condition}_{chunk}_{execution_date}.csv"


def resolve_enabled_cells(cells: list = None) -> list:
    """
    Gets the cells to run: the cells enabled in BASELINE_ARMS_CELLS_X, optionally
    narrowed to a subset.

    Args:
        cells (list, optional): Subset of BASELINE_ARMS_CELLS_X keys. Defaults to None (all enabled cells).

    Returns:
        list: BASELINE_ARMS_CELLS_X keys of the cells to run.

    Raises:
        BaselineArmsError: If a requested cell is not in BASELINE_ARMS_CELLS_X.
    """
    enabled = [key for key, cfg in BASELINE_ARMS_CELLS_X.items() if cfg["enabled"]]
    if not cells:
        return enabled
    unknown = [c for c in cells if c not in BASELINE_ARMS_CELLS_X]
    if unknown:
        raise BaselineArmsError(
            f"Unknown arm cells {unknown}; choose from {list(BASELINE_ARMS_CELLS_X)}."
        )
    return [c for c in cells if c in enabled]


def arm_clock(execution_date: str) -> dict:
    """
    Converts the arm-clock thresholds (New York times on the day after the drop
    date) to UTC.

    Args:
        execution_date (str): Drop date in DD-MM-YYYY format.

    Returns:
        dict: Threshold name (as in BASELINE_ARM_CLOCK_ET_X) to tz-aware UTC datetime.
    """
    next_day = datetime.strptime(execution_date, "%d-%m-%Y").date() + timedelta(days=1)
    clock = {}
    for name, hhmm in BASELINE_ARM_CLOCK_ET_X.items():
        hour, minute = (int(part) for part in hhmm.split(":"))
        clock[name] = datetime.combine(
            next_day, dtime(hour, minute), tzinfo=ET
        ).astimezone(timezone.utc)
    return clock


def timing_flag(timestamp, clock: dict) -> str:
    """
    Classifies the time of a search-on call against the arm clock.

    Args:
        timestamp (datetime): Tz-aware UTC time of the call, or None.
        clock (dict): Output of arm_clock.

    Returns:
        str: "clean" (before the clean cutoff), "contaminated" (before the lost
            cutoff), "lost" (after it) or "unknown" (no timestamp).
    """
    if timestamp is None or pd.isna(timestamp):
        return "unknown"
    if timestamp <= clock["clean_cutoff"]:
        return "clean"
    if timestamp <= clock["lost_cutoff"]:
        return "contaminated"
    return "lost"


def yes_cell_schedule(run_start: datetime, clock: dict, use_row_query: bool) -> dict:
    """
    Decides how the search-on cells run given the arm clock.

    Args:
        run_start (datetime): Tz-aware UTC start time of the arms.
        clock (dict): Output of arm_clock.
        use_row_query (bool): Whether row mode was requested.

    Returns:
        dict: {"skip": True} after the skip cutoff. Otherwise "skip", "use_row_query",
            "batch_timeout_seconds" (time left until the batch deadline) and
            "row_deadline_utc" (the hard stop).
    """
    if run_start > clock["skip_cutoff"]:
        return {"skip": True}
    seconds_to_deadline = (clock["yes_batch_deadline"] - run_start).total_seconds()
    return {
        "skip": False,
        "use_row_query": use_row_query
        or seconds_to_deadline < BASELINE_MIN_YES_BATCH_SECONDS_X,
        "batch_timeout_seconds": max(int(seconds_to_deadline), 0),
        "row_deadline_utc": clock["yes_hard_stop"],
    }


def load_drop(project_name: str, execution_date: str, module: str) -> tuple:
    """
    Reads a module's production drop (_full.csv) as strings.

    Args:
        project_name (str): Name of the project directory.
        execution_date (str): Drop date in DD-MM-YYYY format.
        module (str): "post_interview" or "daily_stock_pick".

    Returns:
        tuple: The rows with non-empty prompts, with the prompt and timestamp columns
            renamed to system_prompt, user_prompt and production_datetime, and the drop path.

    Raises:
        BaselineArmsError: If the drop is missing, lacks the prompt columns, has no
            prompts or has duplicated account_ids.
    """
    spec = BASELINE_MODULES_X[module]
    last_chunk = module_chunks(module)[-1]
    system_col = spec["system_prompt_col"].format(i=last_chunk)
    user_col = spec["user_prompt_col"].format(i=last_chunk)
    path = data_path(
        project_name, execution_date, spec["drop_file"].format(d=execution_date)
    )
    if not os.path.exists(path):
        raise BaselineArmsError(f"production drop not found: {path}")

    drop = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in ("account_id", system_col, user_col) if c not in drop.columns]
    if missing:
        raise BaselineArmsError(f"{path} is missing columns {missing}")
    drop = drop.rename(
        columns={
            system_col: "system_prompt",
            user_col: "user_prompt",
            spec["timestamp_col"]: "production_datetime",
        }
    )
    if "production_datetime" not in drop.columns:
        drop["production_datetime"] = ""
    has_prompts = (drop["system_prompt"].str.strip() != "") & (
        drop["user_prompt"].str.strip() != ""
    )
    drop = drop[has_prompts].reset_index(drop=True)
    if drop.empty:
        raise BaselineArmsError(f"{path} has no rows with non-empty prompts")
    if drop["account_id"].duplicated().any():
        raise BaselineArmsError(f"{path} has duplicated account_ids")
    return drop, path


def module_batteries(
    project_name: str, execution_date: str, module: str, drop: pd.DataFrame
) -> list:
    """
    Builds the day's battery (user prompts) for each chunk of a module and checks it
    against production. The post interview battery is taken from the drop; the stock
    pick batteries are rendered with construct_user_prompt and checked against the
    drop (last chunk) and the chunk files when they exist.

    Args:
        project_name (str): Name of the project directory.
        execution_date (str): Drop date in DD-MM-YYYY format.
        module (str): "post_interview" or "daily_stock_pick".
        drop (pd.DataFrame): Output of load_drop.

    Returns:
        list: One dict per chunk with "chunk", "template", "user_prompt", "user_hash"
            and "verified_against" ("drop", "chunk_file" or "unverified").

    Raises:
        BaselineArmsError: If the drop has more than one battery or a rendered battery
            differs from production.
    """
    production_hashes = drop["user_prompt"].map(sha16).unique()
    if len(production_hashes) != 1:
        raise BaselineArmsError(
            f"{module}: the production drop has {len(production_hashes)} distinct batteries"
        )
    production_prompt = drop.loc[0, "user_prompt"]

    if module == "post_interview":
        return [
            {
                "chunk": 1,
                "template": production_prompt,
                "user_prompt": production_prompt,
                "user_hash": sha16(production_prompt),
                "verified_against": "drop",
            }
        ]

    spec = BASELINE_MODULES_X[module]
    chunks = module_chunks(module)
    batteries = []
    for chunk, template in zip(chunks, daily_stock_pick_user_prompts):
        rendered = construct_user_prompt(
            pd.Series(dtype=object), template, arm_interview_type(module, "yes", chunk)
        )
        reference, source = None, "unverified"
        if chunk == chunks[-1]:
            reference, source = production_prompt, "drop"
        else:
            chunk_path = data_path(
                project_name,
                execution_date,
                spec["chunk_file"].format(d=execution_date, i=chunk),
            )
            user_col = spec["user_prompt_col"].format(i=chunk)
            if os.path.exists(chunk_path):
                chunk_drop = pd.read_csv(
                    chunk_path,
                    usecols=[user_col],
                    nrows=1,
                    dtype=str,
                    keep_default_na=False,
                )
                reference, source = chunk_drop.loc[0, user_col], "chunk_file"
        if reference is not None and sha16(reference) != sha16(rendered):
            raise BaselineArmsError(
                f"{module} chunk {chunk}: rendered battery differs from production ({source})"
            )
        batteries.append(
            {
                "chunk": chunk,
                "template": template,
                "user_prompt": rendered,
                "user_hash": sha16(rendered),
                "verified_against": source,
            }
        )
    return batteries


def _find_once(text: str, substring: str, what: str) -> int:
    """
    Finds a substring that must occur exactly once.

    Args:
        text (str): Text to search.
        substring (str): Substring to find.
        what (str): Description of the substring for the error message.

    Returns:
        int: Index of the substring.

    Raises:
        BaselineArmsError: If the substring does not occur exactly once.
    """
    count = text.count(substring)
    if count != 1:
        raise BaselineArmsError(
            f"{what}: expected exactly one {substring[:60]!r}, found {count}. "
            f"The production prompt structure may have changed; inspect it before launching."
        )
    return text.index(substring)


def split_persona_prompt(prompt: str) -> tuple:
    """
    Splits a production system prompt around its persona block.

    Args:
        prompt (str): Production system prompt.

    Returns:
        tuple: The survey framing before the persona header, and the text from the
            web-search paragraph onwards (web-search guidance and instructions).

    Raises:
        BaselineArmsError: If the persona header or web-search paragraph is not found
            exactly once, or they are out of order.
    """
    header = _find_once(prompt, BASELINE_PERSONA_HEADER_X, "persona header")
    web_search = _find_once(
        prompt,
        BASELINE_SEARCH_OFF_SPLICES_X["remove_paragraph_start"],
        "web-search paragraph",
    )
    if web_search < header:
        raise BaselineArmsError("web-search paragraph precedes the persona header")
    return prompt[:header], prompt[web_search:]


def build_generic_prompt(donor_prompt: str, drop_prompts: pd.Series) -> str:
    """
    Builds the generic system prompt by replacing a donor prompt's persona block with
    BASELINE_GENERIC_REPLACEMENT_X. The framing and instructions must be identical for
    every account in the drop, so that they carry no account-specific content.

    Args:
        donor_prompt (str): Production system prompt of the donor account.
        drop_prompts (pd.Series): Production system prompts of every account in the drop.

    Returns:
        str: The generic system prompt.

    Raises:
        BaselineArmsError: If any prompt cannot be split or the framing or instructions
            differ across accounts.
    """
    block_a, block_cd = split_persona_prompt(donor_prompt)
    for prompt in drop_prompts:
        other_a, other_cd = split_persona_prompt(prompt)
        if other_a != block_a or other_cd != block_cd:
            raise BaselineArmsError(
                "survey framing or instructions differ across accounts; the generic "
                "surgery would leak persona content"
            )
    return (
        block_a.rstrip("\n")
        + "\n\n"
        + BASELINE_GENERIC_REPLACEMENT_X
        + "\n\n"
        + block_cd
    )


def apply_search_off_splices(prompt: str) -> str:
    """
    Applies the two search-off edits: removes the web-search paragraph and replaces
    the web-search instruction bullet (BASELINE_SEARCH_OFF_SPLICES_X).

    Args:
        prompt (str): System prompt with the web-search paragraph and bullet.

    Returns:
        str: The search-off system prompt.

    Raises:
        BaselineArmsError: If the paragraph or bullet is not found exactly once.
    """
    splices = BASELINE_SEARCH_OFF_SPLICES_X
    start = _find_once(
        prompt, splices["remove_paragraph_start"], "web-search paragraph"
    )
    end = prompt.find(splices["remove_paragraph_end"], start)
    if end < 0 or "\n\n" in prompt[start:end]:
        raise BaselineArmsError("end of the web-search paragraph not found")
    end += len(splices["remove_paragraph_end"])
    if prompt[end : end + 2] != "\n\n":
        raise BaselineArmsError("web-search paragraph is not followed by a blank line")
    spliced = prompt[:start] + prompt[end + 2 :]
    _find_once(spliced, splices["replace_line"], "web-search instruction bullet")
    return spliced.replace(splices["replace_line"], splices["replace_with"])


def sort_by_system_prompt_length(drop: pd.DataFrame) -> pd.DataFrame:
    """
    Sorts drop rows by system prompt length, keeping file order for ties.

    Args:
        drop (pd.DataFrame): Output of load_drop.

    Returns:
        pd.DataFrame: The sorted rows with a fresh index.
    """
    order = sorted(range(len(drop)), key=lambda j: len(drop.loc[j, "system_prompt"]))
    return drop.loc[order].reset_index(drop=True)


def draw_replicate_donors(
    sorted_drop: pd.DataFrame, execution_date: str, module: str
) -> pd.DataFrame:
    """
    Draws the replicate donors with a seed derived from the date and module.

    Args:
        sorted_drop (pd.DataFrame): Output of sort_by_system_prompt_length.
        execution_date (str): Drop date in DD-MM-YYYY format.
        module (str): "post_interview" or "daily_stock_pick".

    Returns:
        pd.DataFrame: The donor rows.
    """
    cfg = BASELINE_ARMS_CELLS_X["twin_replicate"]
    rng = random.Random(f'{cfg["donor_seed"]}-{execution_date}-{module}')
    picks = rng.sample(range(len(sorted_drop)), min(cfg["k_donors"], len(sorted_drop)))
    return sorted_drop.loc[picks].reset_index(drop=True)


def _config_snapshot(model_name: str, provider: str) -> dict:
    """
    Collects the settings covered by the manifest's config hash.

    Args:
        model_name (str): Model id.
        provider (str): Resolved provider.

    Returns:
        dict: The baseline arm settings.
    """
    return {
        "model": model_name,
        "provider": provider,
        "cells": BASELINE_ARMS_CELLS_X,
        "bare_system_prompt": BASELINE_BARE_SYSTEM_PROMPT_X,
        "generic_replacement": BASELINE_GENERIC_REPLACEMENT_X,
        "persona_header": BASELINE_PERSONA_HEADER_X,
        "search_off_splices": BASELINE_SEARCH_OFF_SPLICES_X,
        "arm_clock_et": BASELINE_ARM_CLOCK_ET_X,
        "modules": BASELINE_MODULES_X,
        "pricing": BASELINE_MODEL_PRICING_X,
        "batch_discount": BASELINE_BATCH_DISCOUNT_X,
    }


def build_arm_plan(
    project_name: str,
    execution_date: str,
    model_name: str,
    provider: str = None,
    cells: list = None,
) -> dict:
    """
    Builds every arm row for a drop date and the prompt manifest.

    Args:
        project_name (str): Name of the project directory.
        execution_date (str): Drop date in DD-MM-YYYY format.
        model_name (str): Model id.
        provider (str, optional): Provider override. Defaults to None.
        cells (list, optional): Subset of BASELINE_ARMS_CELLS_X keys to run. Defaults to None.

    Returns:
        dict: "execution_date", "model", "provider", "rows" (one row per arm row with its
            system prompt), "drops" (production drops by module), "batteries" (by module)
            and "manifest".

    Raises:
        BaselineArmsError: If the drop, battery or prompt construction fails checks, or
            there are no rows to run.
    """
    enabled = resolve_enabled_cells(cells)
    weekday = datetime.strptime(execution_date, "%d-%m-%Y").strftime("%A")
    replicate_cfg = BASELINE_ARMS_CELLS_X["twin_replicate"]
    replicate_day = "twin_replicate" in enabled and weekday == replicate_cfg["weekday"]
    roster = sorted(set(BASELINE_TX81_ROSTER_X))
    twin_no_scope = BASELINE_ARMS_CELLS_X["twin_no"]["scope"]

    rows, drops, batteries, module_info = [], {}, {}, {}
    for module in MODULES:
        drop, drop_path = load_drop(project_name, execution_date, module)
        drops[module] = drop
        batteries[module] = module_batteries(project_name, execution_date, module, drop)
        sorted_drop = sort_by_system_prompt_length(drop)
        donor = sorted_drop.loc[len(sorted_drop) // 2]
        generic_prompt = build_generic_prompt(
            donor["system_prompt"], drop["system_prompt"]
        )

        def add(
            condition, cell, account_id, system_prompt, donor_account=None, replicate=0
        ):
            """
            Appends one arm row for the current module.

            Args:
                condition (str): "yes" or "no".
                cell (str): Output cell name (a CELL_CONFIG_KEYS key).
                account_id (str): Pseudo-ID or real account_id of the row.
                system_prompt (str): System prompt of the row.
                donor_account (str, optional): Production account the prompt comes from. Defaults to None.
                replicate (int, optional): Replicate number, 0 for non-replicates. Defaults to 0.
            """
            cfg = BASELINE_ARMS_CELLS_X[CELL_CONFIG_KEYS[cell]]
            rows.append(
                {
                    "arm_row_id": f"{cell}|{account_id}",
                    "module": module,
                    "condition": condition,
                    "cell": cell,
                    "alias": cfg["alias"],
                    "setup": cfg["setup"],
                    "web_search": condition == "yes",
                    "account_id": account_id,
                    "donor_account": donor_account,
                    "replicate": replicate,
                    "tx81": donor_account in roster if donor_account else False,
                    "system_hash": sha16(system_prompt),
                    ARM_SYSTEM_PROMPT_COL: system_prompt,
                }
            )

        for condition in CONDITIONS:
            cell = f"bare_{condition}"
            if cell in enabled:
                cfg = BASELINE_ARMS_CELLS_X[cell]
                for k in range(1, cfg["k"] + 1):
                    add(
                        condition,
                        cell,
                        f'{cfg["id_prefix"]}_{k:02d}',
                        BASELINE_BARE_SYSTEM_PROMPT_X,
                    )
            cell = f"generic_{condition}"
            if cell in enabled:
                cfg = BASELINE_ARMS_CELLS_X[cell]
                prompt = (
                    generic_prompt
                    if condition == "yes"
                    else apply_search_off_splices(generic_prompt)
                )
                for k in range(1, cfg["k"] + 1):
                    add(
                        condition,
                        cell,
                        f'{cfg["id_prefix"]}_{k:02d}',
                        prompt,
                        donor_account=donor["account_id"],
                    )

        if "twin_no" in enabled:
            twin_rows = (
                drop
                if twin_no_scope == "all"
                else drop[drop["account_id"].isin(roster)]
            )
            for _, twin in twin_rows.iterrows():
                add(
                    "no",
                    "twin_no",
                    twin["account_id"],
                    apply_search_off_splices(twin["system_prompt"]),
                    donor_account=twin["account_id"],
                )

        replicate_donors = []
        if replicate_day:
            n = 0
            for _, rep_donor in draw_replicate_donors(
                sorted_drop, execution_date, module
            ).iterrows():
                replicate_donors.append(rep_donor["account_id"])
                for rep in range(1, replicate_cfg["replicates_per_donor"] + 1):
                    n += 1
                    add(
                        "yes",
                        "twin_yes_replicate",
                        f'{replicate_cfg["id_prefix"]["yes"]}_{n:02d}',
                        rep_donor["system_prompt"],
                        donor_account=rep_donor["account_id"],
                        replicate=rep,
                    )
                    add(
                        "no",
                        "twin_no_replicate",
                        f'{replicate_cfg["id_prefix"]["no"]}_{n:02d}',
                        apply_search_off_splices(rep_donor["system_prompt"]),
                        donor_account=rep_donor["account_id"],
                        replicate=rep,
                    )

        module_info[module] = {
            "drop_file": drop_path,
            "drop_mtime_utc": datetime.fromtimestamp(
                os.path.getmtime(drop_path), timezone.utc
            ).isoformat(),
            "n_production_rows": len(drop),
            "n_tx81_in_drop": int(drop["account_id"].isin(roster).sum()),
            "production_datetime": drop["production_datetime"].max(),
            "generic_donor_account": donor["account_id"],
            "replicate_donors": replicate_donors,
            "batteries": [
                {k: v for k, v in b.items() if k not in ("template", "user_prompt")}
                for b in batteries[module]
            ],
        }

    rows = pd.DataFrame(rows)
    if rows.empty:
        raise BaselineArmsError("no arm rows to run (all cells disabled?)")
    duplicated = rows.duplicated(["module", "arm_row_id"])
    if duplicated.any():
        raise BaselineArmsError(
            f"duplicated arm rows: {rows.loc[duplicated, 'arm_row_id'].tolist()[:5]}"
        )

    # Each search-on replicate prompt must equal its donor's production prompt
    replicate_draw1_ok = True
    for module, drop in drops.items():
        production_hashes = dict(
            zip(drop["account_id"], drop["system_prompt"].map(sha16))
        )
        reps = rows[(rows["module"] == module) & (rows["cell"] == "twin_yes_replicate")]
        if (reps["system_hash"] != reps["donor_account"].map(production_hashes)).any():
            replicate_draw1_ok = False

    provider_resolved = _detect_provider(model_name=model_name, provider=provider)
    prompt_texts = rows.drop_duplicates("system_hash").set_index("system_hash")[
        ARM_SYSTEM_PROMPT_COL
    ]
    manifest = {
        "date": execution_date,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "provider": provider_resolved,
        "config_hash": sha16(
            json.dumps(
                _config_snapshot(model_name, provider_resolved),
                sort_keys=True,
                default=str,
            )
        ),
        "tx81_roster_hash": sha16("\n".join(roster)),
        "tx81_roster_size": len(roster),
        "enabled_cells": enabled,
        "replicate_day": replicate_day,
        "replicate_draw1_ok": replicate_draw1_ok,
        "twin_no_scope": twin_no_scope,
        "modules": module_info,
        "jobs": rows.drop(columns=[ARM_SYSTEM_PROMPT_COL]).to_dict(orient="records"),
        "prompts": {
            "system": prompt_texts.to_dict(),
            "user": {
                b["user_hash"]: b["user_prompt"]
                for module in MODULES
                for b in batteries[module]
            },
        },
    }
    return {
        "execution_date": execution_date,
        "model": model_name,
        "provider": provider_resolved,
        "rows": rows,
        "drops": drops,
        "batteries": batteries,
        "manifest": manifest,
    }


def write_arm_plan_files(plan: dict, project_name: str) -> str:
    """
    Writes the prompt manifest, the input CSV of each module and condition, and an
    empty post file to the baseline folder.

    Args:
        plan (dict): Output of build_arm_plan.
        project_name (str): Name of the project directory.

    Returns:
        str: Path of the baseline folder.
    """
    execution_date = plan["execution_date"]
    out_dir = data_path(project_name, baseline_execution_date(execution_date))
    os.makedirs(out_dir, exist_ok=True)

    # Prompts are pre-rendered, so perform_profile_interview gets no posts
    pd.DataFrame(columns=["account_id", "createdAt", "text"]).to_csv(
        os.path.join(out_dir, EMPTY_POST_FILE), index=False
    )
    for (module, condition), rows in plan["rows"].groupby(["module", "condition"]):
        rows.to_csv(
            os.path.join(out_dir, arm_input_file(module, condition, execution_date)),
            index=False,
        )
    manifest_path = os.path.join(out_dir, f"prompts_manifest_{execution_date}.json")
    with open(manifest_path, "w") as f:
        json.dump(plan["manifest"], f, indent=1, ensure_ascii=False, default=str)
    return out_dir


def write_arm_batch_inputs(plan: dict, project_name: str) -> list:
    """
    Writes the batch input JSONL of every arm task without submitting it (dry run).

    Args:
        plan (dict): Output of build_arm_plan.
        project_name (str): Name of the project directory.

    Returns:
        list: Names of the written batch input files.
    """
    execution_date = plan["execution_date"]
    batch_date = baseline_execution_date(execution_date)
    os.makedirs(data_path(project_name, batch_date, "batch-files"), exist_ok=True)
    written = []
    for (module, condition), rows in plan["rows"].groupby(["module", "condition"]):
        for battery in plan["batteries"][module]:
            interview_type = arm_interview_type(module, condition, battery["chunk"])
            prompts = rows.reset_index(drop=True).copy()
            prompts["custom_id"] = prompts.index
            prompts[f"{interview_type}_system_prompt"] = prompts[ARM_SYSTEM_PROMPT_COL]
            prompts[f"{interview_type}_user_prompt"] = battery["user_prompt"]
            written.append(
                create_batch_file(
                    prompts,
                    project_name=project_name,
                    execution_date=batch_date,
                    model_name=plan["model"],
                    system_prompt_field=f"{interview_type}_system_prompt",
                    user_prompt_field=f"{interview_type}_user_prompt",
                    batch_file_name=f"{interview_type}_batch_input.jsonl",
                    enable_web_search=condition == "yes",
                    provider=plan["provider"],
                    responses_api_without_tools=condition == "no",
                    include_web_search_sources=True,
                )
            )
    return written


def print_plan_summary(plan: dict) -> None:
    """
    Prints the model, hashes and the number of rows and calls of each cell.

    Args:
        plan (dict): Output of build_arm_plan.

    Returns:
        None
    """
    manifest = plan["manifest"]
    print(
        f"Baseline arms {plan['execution_date']}: model={plan['model']} "
        f"replicate_day={manifest['replicate_day']} config_hash={manifest['config_hash']} "
        f"tx81_roster_hash={manifest['tx81_roster_hash']}"
    )
    counts = plan["rows"].groupby(["module", "cell"]).size()
    for (module, cell), n in counts.items():
        n_chunks = len(module_chunks(module))
        print(f"  {module:16s} {cell:20s} rows={n:4d} calls={n * n_chunks:5d}")


def parse_raw_response(raw) -> dict:
    """
    Extracts usage, web search activity and the response time from a raw response:
    an OpenAI batch output line, an OpenAI Responses object (row mode), or an
    Anthropic batch result or message.

    Args:
        raw (str): Raw response JSON.

    Returns:
        dict: call_path ("batch" or "row"), response_status, response_created_utc,
            input_tokens, cached_input_tokens, output_tokens, reasoning_tokens,
            n_search_calls, search_queries, search_sources and cited_urls.
    """
    parsed = {
        "call_path": None,
        "response_status": None,
        "response_created_utc": None,
        "input_tokens": np.nan,
        "cached_input_tokens": np.nan,
        "output_tokens": np.nan,
        "reasoning_tokens": np.nan,
        "n_search_calls": 0,
        "search_queries": [],
        "search_sources": [],
        "cited_urls": [],
    }
    if not isinstance(raw, str) or not raw.strip():
        return parsed
    try:
        obj = json.loads(raw)
    except ValueError:
        return parsed

    if "custom_id" in obj and "response" in obj:  # OpenAI batch output line
        parsed["call_path"] = "batch"
        body = (obj.get("response") or {}).get("body") or {}
    elif "custom_id" in obj and "result" in obj:  # Anthropic batch result
        parsed["call_path"] = "batch"
        body = (obj.get("result") or {}).get("message") or {}
    else:
        parsed["call_path"] = "row"
        body = obj

    usage = body.get("usage") or {}
    parsed["input_tokens"] = usage.get(
        "input_tokens", usage.get("prompt_tokens", np.nan)
    )
    input_details = (
        usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    )
    parsed["cached_input_tokens"] = input_details.get(
        "cached_tokens", usage.get("cache_read_input_tokens", 0)
    )
    parsed["output_tokens"] = usage.get(
        "output_tokens", usage.get("completion_tokens", np.nan)
    )
    parsed["reasoning_tokens"] = (usage.get("output_tokens_details") or {}).get(
        "reasoning_tokens", np.nan
    )
    parsed["response_status"] = body.get("status") or body.get("stop_reason")
    created = body.get("completed_at") or body.get("created_at") or body.get("created")
    if isinstance(created, (int, float)):
        parsed["response_created_utc"] = datetime.fromtimestamp(created, timezone.utc)

    # OpenAI Responses output items
    for item in body.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "web_search_call":
            parsed["n_search_calls"] += 1
            action = item.get("action") or {}
            # `queries` repeats `query`; older payloads only have `query`
            queries = action.get("queries") or (
                [action["query"]] if action.get("query") else []
            )
            parsed["search_queries"].extend(queries)
            parsed["search_sources"].extend(
                s.get("url") for s in action.get("sources") or [] if isinstance(s, dict)
            )
        elif item.get("type") == "message":
            for content in item.get("content") or []:
                for annotation in (content or {}).get("annotations") or []:
                    if annotation.get("type") == "url_citation":
                        parsed["cited_urls"].append(annotation.get("url"))
    # Anthropic content blocks
    for block in body.get("content") or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "server_tool_use" and block.get("name") == "web_search":
            parsed["n_search_calls"] += 1
            query = (block.get("input") or {}).get("query")
            if query:
                parsed["search_queries"].append(query)
        elif block.get("type") == "web_search_tool_result":
            content = block.get("content")
            if isinstance(content, list):
                parsed["search_sources"].extend(
                    r.get("url") for r in content if isinstance(r, dict)
                )
        for citation in block.get("citations") or []:
            if isinstance(citation, dict) and citation.get("url"):
                parsed["cited_urls"].append(citation["url"])
    return parsed


def cost_usd(
    model_name: str,
    input_tokens,
    cached_input_tokens,
    output_tokens,
    n_search_calls,
    call_path: str,
) -> float:
    """
    Computes the cost of a call from BASELINE_MODEL_PRICING_X. Batch calls get the
    batch discount on token costs.

    Args:
        model_name (str): Model id.
        input_tokens (int): Input tokens, including cached tokens.
        cached_input_tokens (int): Cached input tokens.
        output_tokens (int): Output tokens.
        n_search_calls (int): Number of web search calls.
        call_path (str): "batch" or "row".

    Returns:
        float: Cost in USD, or NaN if the model is not priced or usage is missing.
    """
    price = BASELINE_MODEL_PRICING_X.get(model_name)
    if price is None or pd.isna(input_tokens) or pd.isna(output_tokens):
        return np.nan
    cached = 0 if pd.isna(cached_input_tokens) else cached_input_tokens
    token_cost = (
        max(input_tokens - cached, 0) * price["input"]
        + cached * price["cached_input"]
        + output_tokens * price["output"]
    ) / 1e6
    if call_path == "batch":
        token_cost *= BASELINE_BATCH_DISCOUNT_X
    return token_cost + (n_search_calls or 0) * price["web_search_per_1k_calls"] / 1000


def _with_raw_metadata(
    calls: pd.DataFrame, raw_col: str, model_name: str
) -> pd.DataFrame:
    """
    Adds the parse_raw_response fields and the cost to each call.

    Args:
        calls (pd.DataFrame): One row per call.
        raw_col (str): Column holding the raw response JSON.
        model_name (str): Model id.

    Returns:
        pd.DataFrame: The calls with the added columns.
    """
    parsed = pd.DataFrame(
        [parse_raw_response(raw) for raw in calls[raw_col]], index=calls.index
    )
    calls = pd.concat([calls, parsed], axis=1)
    calls["cost_usd"] = [
        cost_usd(
            model_name,
            r.input_tokens,
            r.cached_input_tokens,
            r.output_tokens,
            r.n_search_calls,
            r.call_path,
        )
        for r in calls.itertuples()
    ]
    return calls


def collect_arm_calls(plan: dict, project_name: str, clock: dict) -> pd.DataFrame:
    """
    Collects one row per arm call from the interview outputs, with the arm row
    metadata, sent prompt hashes, usage, cost, timing flag and time since the
    production interview.

    Args:
        plan (dict): Output of build_arm_plan.
        project_name (str): Name of the project directory.
        clock (dict): Output of arm_clock.

    Returns:
        pd.DataFrame: One row per call, or an empty DataFrame if there are no outputs.
    """
    execution_date = plan["execution_date"]
    out_dir = data_path(project_name, baseline_execution_date(execution_date))
    frames = []
    for (module, condition), rows in plan["rows"].groupby(["module", "condition"]):
        for battery in plan["batteries"][module]:
            chunk = battery["chunk"]
            path = os.path.join(
                out_dir, arm_output_file(module, condition, chunk, execution_date)
            )
            if not os.path.exists(path):
                continue
            interview_type = arm_interview_type(module, condition, chunk)
            out = pd.read_csv(path, dtype=str, keep_default_na=False)
            calls = pd.DataFrame(
                {
                    "arm_row_id": out["arm_row_id"],
                    "module": module,
                    "condition": condition,
                    "chunk": chunk,
                    "interview_type": interview_type,
                    "response": out[ARM_RESPONSE_FIELD],
                    "raw_response": out.get(ARM_RAW_RESPONSE_FIELD, ""),
                    "row_timestamp": out.get(ARM_TIMESTAMP_COL, ""),
                    "sent_system_hash": out[f"{interview_type}_system_prompt"].map(
                        sha16
                    ),
                    "sent_user_hash": out[f"{interview_type}_user_prompt"].map(sha16),
                    "expected_user_hash": battery["user_hash"],
                    "output_mtime_utc": datetime.fromtimestamp(
                        os.path.getmtime(path), timezone.utc
                    ),
                }
            )
            frames.append(calls)
    if not frames:
        return pd.DataFrame()

    calls = pd.concat(frames, ignore_index=True)
    calls = calls.merge(
        plan["rows"].drop(columns=[ARM_SYSTEM_PROMPT_COL, "condition"]),
        on=["module", "arm_row_id"],
        how="left",
    )
    calls = _with_raw_metadata(calls, "raw_response", plan["model"])

    # Call time: the API's timestamp, else the row-mode call time, else the batch output time
    row_timestamps = pd.to_datetime(calls["row_timestamp"], utc=True, errors="coerce")
    calls["call_timestamp_utc"] = (
        pd.to_datetime(calls["response_created_utc"], utc=True)
        .fillna(row_timestamps)
        .fillna(calls["output_mtime_utc"])
    )
    no_raw = calls["call_path"].isna()
    calls.loc[no_raw, "call_path"] = np.where(
        row_timestamps[no_raw].notna(), "row", "batch"
    )
    calls["timing_flag"] = [
        timing_flag(ts, clock) if web_search else "n/a"
        for ts, web_search in zip(calls["call_timestamp_utc"], calls["web_search"])
    ]
    production_times = {
        module: pd.to_datetime(info["production_datetime"], utc=True, errors="coerce")
        for module, info in plan["manifest"]["modules"].items()
    }
    calls["production_datetime_utc"] = calls["module"].map(production_times)
    calls["delta_t_hours"] = (
        calls["call_timestamp_utc"] - calls["production_datetime_utc"]
    ).dt.total_seconds() / 3600
    calls["response_ok"] = ~calls["response"].isin(
        ["", LLM_ERROR_RESPONSE, LLM_SKIPPED_RESPONSE]
    )
    return calls


def write_arm_call_records(
    calls: pd.DataFrame, project_name: str, execution_date: str
) -> None:
    """
    Writes baseline_responses_{d}.jsonl (one record per call, with the raw
    response) and x_baseline_search_traces_{d}.csv (search-on calls).

    Args:
        calls (pd.DataFrame): Output of collect_arm_calls.
        project_name (str): Name of the project directory.
        execution_date (str): Drop date in DD-MM-YYYY format.

    Returns:
        None
    """
    out_dir = data_path(project_name, baseline_execution_date(execution_date))
    with open(
        os.path.join(out_dir, f"baseline_responses_{execution_date}.jsonl"), "w"
    ) as f:
        for record in calls.drop(columns=["row_timestamp", "output_mtime_utc"]).to_dict(
            orient="records"
        ):
            raw = record.pop("raw_response")
            try:
                record["raw_response"] = json.loads(raw) if raw else None
            except ValueError:
                record["raw_response"] = raw
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    traces = calls[calls["web_search"].astype(bool)][
        [
            "module",
            "chunk",
            "cell",
            "account_id",
            "donor_account",
            "replicate",
            "call_path",
            "call_timestamp_utc",
            "timing_flag",
            "n_search_calls",
            "search_queries",
            "search_sources",
            "cited_urls",
        ]
    ].copy()
    for col in ("search_queries", "search_sources", "cited_urls"):
        traces[col] = traces[col].map(lambda v: json.dumps(v, ensure_ascii=False))
    traces.to_csv(
        os.path.join(out_dir, f"x_baseline_search_traces_{execution_date}.csv"),
        index=False,
    )


def extract_production_search_traces(plan: dict, project_name: str) -> tuple:
    """
    Extracts the production (twin_yes) search traces and cost from production's raw
    batch outputs in the drop's batch-files folder, and writes
    x_twin_yes_search_traces_{d}.csv.

    Args:
        plan (dict): Output of build_arm_plan.
        project_name (str): Name of the project directory.

    Returns:
        tuple: One row per production call (empty if no batch outputs were found), and
            the list of missing batch output files.
    """
    execution_date = plan["execution_date"]
    frames, missing = [], []
    for module in MODULES:
        spec = BASELINE_MODULES_X[module]
        drop = plan["drops"][module]
        account_by_custom_id = (
            dict(zip(drop["custom_id"], drop["account_id"]))
            if "custom_id" in drop.columns
            else {}
        )
        for chunk in module_chunks(module):
            path = data_path(
                project_name,
                execution_date,
                "batch-files",
                spec["production_batch_output"].format(i=chunk),
            )
            if not os.path.exists(path):
                missing.append(path)
                continue
            with open(path) as f:
                lines = [line.strip() for line in f if line.strip()]
            custom_ids = [str(json.loads(line).get("custom_id")) for line in lines]
            frames.append(
                pd.DataFrame(
                    {
                        "module": module,
                        "chunk": chunk,
                        "custom_id": custom_ids,
                        "account_id": [account_by_custom_id.get(c) for c in custom_ids],
                        "raw_response": lines,
                    }
                )
            )
    if not frames:
        return pd.DataFrame(), missing
    traces = _with_raw_metadata(
        pd.concat(frames, ignore_index=True), "raw_response", plan["model"]
    )
    traces = traces.drop(columns=["raw_response"])
    traces["tx81"] = traces["account_id"].isin(set(BASELINE_TX81_ROSTER_X))
    for col in ("search_queries", "search_sources", "cited_urls"):
        traces[col] = traces[col].map(lambda v: json.dumps(v, ensure_ascii=False))
    out_dir = data_path(project_name, baseline_execution_date(execution_date))
    traces.to_csv(
        os.path.join(out_dir, f"x_twin_yes_search_traces_{execution_date}.csv"),
        index=False,
    )
    return traces, missing


def consolidate_module_results(
    plan: dict, calls: pd.DataFrame, module: str, project_name: str
) -> pd.DataFrame:
    """
    Combines a module's calls into one row per arm row and writes
    x_baseline_{module}_{d}.csv. Each row has the arm metadata, the response of
    each chunk, the parsed answers (same parser and column coalescing as
    production), usage, cost and timing. Chunks are joined on arm_row_id.

    Args:
        plan (dict): Output of build_arm_plan.
        calls (pd.DataFrame): Output of collect_arm_calls.
        module (str): "post_interview" or "daily_stock_pick".
        project_name (str): Name of the project directory.

    Returns:
        pd.DataFrame: One row per arm row, or an empty DataFrame if the module has no calls.
    """
    rows = plan["rows"][plan["rows"]["module"] == module].drop(
        columns=[ARM_SYSTEM_PROMPT_COL]
    )
    module_calls = calls[calls["module"] == module] if not calls.empty else calls
    if module_calls.empty:
        return pd.DataFrame()

    parsed = None
    for chunk, chunk_calls in module_calls.groupby("chunk"):
        extracted = pd.DataFrame(
            [extract_llm_responses(text) for text in chunk_calls["response"]],
            index=chunk_calls["arm_row_id"].values,
        )
        parsed = extracted if parsed is None else parsed.combine_first(extracted)
    patterns = (
        FINFLUENCER_DAILY_STOCK_PICK_REGEX_PATTERNS
        if module == "daily_stock_pick"
        else FINFLUENCER_INTERVIEW_REGEX_PATTERNS
    )
    parsed = coalesce_columns_by_regex(parsed, patterns)

    responses = module_calls.pivot(
        index="arm_row_id", columns="chunk", values="response"
    )
    responses.columns = [f"response_chunk_{c}" for c in responses.columns]
    summary = module_calls.groupby("arm_row_id").agg(
        n_calls=("chunk", "size"),
        n_calls_ok=("response_ok", "sum"),
        call_paths=("call_path", lambda v: ",".join(sorted(set(v.dropna())))),
        first_call_utc=("call_timestamp_utc", "min"),
        last_call_utc=("call_timestamp_utc", "max"),
        delta_t_hours=("delta_t_hours", "median"),
        input_tokens=("input_tokens", "sum"),
        cached_input_tokens=("cached_input_tokens", "sum"),
        output_tokens=("output_tokens", "sum"),
        n_search_calls=("n_search_calls", "sum"),
        cost_usd=("cost_usd", lambda v: v.sum(min_count=1)),
        timing_flag=("timing_flag", _worst_timing_flag),
    )
    results = (
        rows.set_index("arm_row_id")
        .join(summary, how="inner")
        .join(responses)
        .join(parsed)
        .reset_index()
    )
    results["model"] = plan["model"]
    out_dir = data_path(project_name, baseline_execution_date(plan["execution_date"]))
    results.to_csv(
        os.path.join(out_dir, f"x_baseline_{module}_{plan['execution_date']}.csv"),
        index=False,
    )
    return results


def _worst_timing_flag(flags) -> str:
    """
    Gets the most severe timing flag.

    Args:
        flags (Iterable[str]): Timing flags of a set of calls.

    Returns:
        str: The most severe flag, or "n/a" if none is a timing flag.
    """
    flags = [f for f in flags if f in TIMING_SEVERITY]
    if not flags:
        return "n/a"
    return max(flags, key=TIMING_SEVERITY.get)


def parsed_field_key(module: str, column: str):
    """
    Maps a parsed-answer column to a key that identifies its question and field,
    so production and arm answers can be compared even if the question text differs.

    Args:
        module (str): "post_interview" or "daily_stock_pick".
        column (str): Column name produced by extract_llm_responses.

    Returns:
        tuple: The key, or None if the column is not a parsed answer.
    """
    match = re.search(r"\s-\s*([a-z ]+)$", column)
    if not match or match.group(1).strip() not in PARSED_FIELDS:
        return None
    field = match.group(1).strip()
    if module == "daily_stock_pick":
        top = re.search(r"top[-\s]+conviction (BUY|SELL)", column, re.I)
        if top:
            return ("top", top.group(1).upper(), field)
        ticker = re.search(r"\(([^()]+)\)[^()]*$", column[: match.start()])
        return ("ticker", ticker.group(1), field) if ticker else None
    for idx, pattern in enumerate(FINFLUENCER_INTERVIEW_REGEX_PATTERNS):
        if re.search(pattern, column, re.I):
            return ("question", idx)
    return ("column", column)


def _answered_keys(module: str, data: pd.DataFrame) -> pd.DataFrame:
    """
    Marks which parsed answers each row has.

    Args:
        module (str): "post_interview" or "daily_stock_pick".
        data (pd.DataFrame): Rows with parsed-answer columns.

    Returns:
        pd.DataFrame: Rows by answer key, True where the row has a non-empty answer.
    """
    keyed = {c: parsed_field_key(module, c) for c in data.columns}
    keyed = {c: k for c, k in keyed.items() if k is not None}
    if not keyed:
        return pd.DataFrame(index=data.index)
    cols = list(keyed)
    answered = data[cols].apply(lambda s: s.notna() & (s.astype(str).str.strip() != ""))
    answered.columns = [str(keyed[c]) for c in cols]
    return answered.T.groupby(level=0).any().T


def _speculation_scores(data: pd.DataFrame) -> pd.Series:
    """
    Collects the numeric speculation scores of a set of rows.

    Args:
        data (pd.DataFrame): Rows with parsed-answer columns.

    Returns:
        pd.Series: All speculation scores found.
    """
    cols = [c for c in data.columns if re.search(r"\s-\s*speculation$", c)]
    if not cols:
        return pd.Series(dtype=float)
    values = data[cols].stack().astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0]
    return pd.to_numeric(values, errors="coerce").dropna()


def qc_and_log(
    plan: dict,
    calls: pd.DataFrame,
    results: dict,
    production_traces: pd.DataFrame,
    missing_production_traces: list,
    project_name: str,
    run_start: datetime,
    yes_skipped: bool,
    ignore_arm_clock: bool,
    task_errors: list,
) -> int:
    """
    Runs the QC of each module and cell, appends it to the baseline run log and
    prints one line per cell plus an overall status line.

    Args:
        plan (dict): Output of build_arm_plan.
        calls (pd.DataFrame): Output of collect_arm_calls.
        results (dict): Output of consolidate_module_results by module.
        production_traces (pd.DataFrame): Output of extract_production_search_traces.
        missing_production_traces (list): Production batch output files that were not found.
        project_name (str): Name of the project directory.
        run_start (datetime): Tz-aware UTC start time of the arms.
        yes_skipped (bool): Whether the search-on cells were skipped by the arm clock.
        ignore_arm_clock (bool): Whether the arm clock was ignored (testing only).
        task_errors (list): Errors raised by arm tasks, as dicts with "module" and "condition".

    Returns:
        int: Exit status, 0 OK / 1 WARN / 2 FAIL.
    """
    execution_date = plan["execution_date"]
    manifest = plan["manifest"]
    thresholds = BASELINE_QC_THRESHOLDS_X
    run_end = datetime.now(timezone.utc)
    drop_arrival = max(info["drop_mtime_utc"] for info in manifest["modules"].values())

    # Expected answers are those production gave for at least half of its rows
    expected_keys = {}
    for module, drop in plan["drops"].items():
        production_answered = _answered_keys(module, drop)
        share = (
            production_answered.mean()
            if not production_answered.empty
            else pd.Series(dtype=float)
        )
        expected_keys[module] = share[share >= 0.5].index.tolist()

    twin_no_cost = 0.0
    if not calls.empty:
        twin_no_cost = calls.loc[calls["cell"] == "twin_no", "cost_usd"].sum()

    log_rows = []
    for (module, cell), rows in plan["rows"].groupby(["module", "cell"]):
        n_chunks = len(module_chunks(module))
        cell_calls = (
            calls[(calls["module"] == module) & (calls["cell"] == cell)]
            if not calls.empty
            else calls
        )
        web_search = bool(rows["web_search"].iloc[0])
        n_expected = len(rows) * n_chunks
        n_returned = int(cell_calls["response_ok"].sum()) if not cell_calls.empty else 0
        completeness = n_returned / n_expected if n_expected else np.nan
        notes, status = [], "OK"

        def escalate(level, note):
            """
            Records a QC note and raises the cell status to at least the given level.

            Args:
                level (str): "OK", "WARN" or "FAIL".
                note (str): Note added to the run log.
            """
            nonlocal status
            notes.append(note)
            if STATUS_CODES[level] > STATUS_CODES[status]:
                status = level

        skipped = web_search and yes_skipped
        if skipped:
            escalate("WARN", "skipped: arms started after the 09:00 ET cutoff")
        elif completeness < thresholds["fail_completeness"]:
            escalate("FAIL", f"completeness {completeness:.2f}")
        elif completeness < thresholds["warn_completeness"]:
            escalate("WARN", f"completeness {completeness:.2f}")

        n_battery_hashes = battery_match = system_hash_match = None
        if not cell_calls.empty:
            n_battery_hashes = int(
                cell_calls.groupby("chunk")["sent_user_hash"].nunique().max()
            )
            battery_match = bool(
                (cell_calls["sent_user_hash"] == cell_calls["expected_user_hash"]).all()
            )
            system_hash_match = bool(
                (cell_calls["sent_system_hash"] == cell_calls["system_hash"]).all()
            )
            if n_battery_hashes > 1 or not battery_match:
                escalate("FAIL", "battery hash mismatch")
            if not system_hash_match:
                escalate("FAIL", "system prompt hash mismatch")

        module_results = results.get(module, pd.DataFrame())
        cell_results = (
            module_results[module_results["cell"] == cell]
            if not module_results.empty
            else module_results
        )
        parse_yield = speculation_mean = speculation_share = np.nan
        n_refusals = None
        if not cell_results.empty and expected_keys[module]:
            answered = _answered_keys(module, cell_results.set_index("arm_row_id"))
            answered = answered.reindex(columns=expected_keys[module], fill_value=False)
            coverage = answered.mean(axis=1)
            parse_yield = float(coverage.median())
            n_refusals = int((answered.sum(axis=1) == 0).sum())
            if parse_yield < thresholds["warn_parse_yield"]:
                escalate("WARN", f"parse yield {parse_yield:.2f}")
            if cell.startswith("bare"):
                scores = _speculation_scores(cell_results)
                if not scores.empty:
                    speculation_mean = float(scores.mean())
                    speculation_share = float(((scores >= 81) & (scores <= 100)).mean())

        flags = cell_calls["timing_flag"].tolist() if not cell_calls.empty else []
        timing = (
            "skipped"
            if skipped
            else (_worst_timing_flag(flags) if web_search else "n/a")
        )
        if ignore_arm_clock and web_search:
            timing = "override"
        elif timing in ("contaminated", "lost"):
            escalate("WARN", f"timing {timing}")

        cost = (
            cell_calls["cost_usd"].sum(min_count=1) if not cell_calls.empty else np.nan
        )
        if n_returned and pd.isna(cost):
            escalate("WARN", "cost unavailable (model missing from pricing table?)")
        if cell == "twin_no" and twin_no_cost > BASELINE_TWIN_NO_COST_WARN_USD_X:
            escalate("WARN", f"twin_no daily cost ${twin_no_cost:.2f} above threshold")
        if (
            web_search
            and not cell_calls.empty
            and cell_calls["raw_response"].eq("").all()
        ):
            escalate("WARN", "search traces unavailable")
        if cell == "twin_yes_replicate" and not manifest["replicate_draw1_ok"]:
            escalate("WARN", "replicate prompt differs from production (draw #1)")
        errors = [
            e
            for e in task_errors
            if e["module"] == module and e["condition"] == rows["condition"].iloc[0]
        ]
        if errors:
            escalate(
                "FAIL" if completeness < thresholds["fail_completeness"] else "WARN",
                f"{len(errors)} task error(s)",
            )

        log_rows.append(
            {
                "date": execution_date,
                "module": module,
                "cell": cell,
                "alias": rows["alias"].iloc[0],
                "web_search": web_search,
                "model": plan["model"],
                "provider": plan["provider"],
                "status": status,
                "notes": "; ".join(notes),
                "n_rows": len(rows),
                "n_chunks": n_chunks,
                "n_expected_calls": n_expected,
                "n_returned": n_returned,
                "n_error": (
                    int(cell_calls["response"].eq(LLM_ERROR_RESPONSE).sum())
                    if not cell_calls.empty
                    else 0
                ),
                "n_skipped_deadline": (
                    int(cell_calls["response"].eq(LLM_SKIPPED_RESPONSE).sum())
                    if not cell_calls.empty
                    else 0
                ),
                "completeness": completeness,
                "parse_yield": parse_yield,
                "n_refusals": n_refusals,
                "n_battery_hashes": n_battery_hashes,
                "battery_match": battery_match,
                "system_hash_match": system_hash_match,
                "replicate_draw1_ok": (
                    manifest["replicate_draw1_ok"] if "replicate" in cell else None
                ),
                "speculation_mean": speculation_mean,
                "speculation_share_81_100": speculation_share,
                "search_call_share": (
                    float((cell_calls["n_search_calls"] > 0).mean())
                    if web_search and not cell_calls.empty
                    else np.nan
                ),
                "mean_searches_per_call": (
                    float(cell_calls["n_search_calls"].mean())
                    if web_search and not cell_calls.empty
                    else np.nan
                ),
                "input_tokens": (
                    cell_calls["input_tokens"].sum() if not cell_calls.empty else 0
                ),
                "cached_input_tokens": (
                    cell_calls["cached_input_tokens"].sum()
                    if not cell_calls.empty
                    else 0
                ),
                "output_tokens": (
                    cell_calls["output_tokens"].sum() if not cell_calls.empty else 0
                ),
                "cost_usd": cost,
                "call_paths": (
                    ",".join(sorted(set(cell_calls["call_path"].dropna())))
                    if not cell_calls.empty
                    else ""
                ),
                "timing_flag": timing,
                "n_clean": flags.count("clean"),
                "n_contaminated": flags.count("contaminated"),
                "n_lost": flags.count("lost"),
                "first_call_utc": (
                    cell_calls["call_timestamp_utc"].min()
                    if not cell_calls.empty
                    else None
                ),
                "last_call_utc": (
                    cell_calls["call_timestamp_utc"].max()
                    if not cell_calls.empty
                    else None
                ),
                "delta_t_median_hours": (
                    cell_calls["delta_t_hours"].median()
                    if not cell_calls.empty
                    else np.nan
                ),
                "tx81_rows": int(rows["tx81"].sum()),
                "drop_arrival_utc": drop_arrival,
                "run_start_utc": run_start.isoformat(),
                "run_end_utc": run_end.isoformat(),
                "run_start_et": run_start.astimezone(ET).isoformat(),
                "run_end_et": run_end.astimezone(ET).isoformat(),
                "config_hash": manifest["config_hash"],
                "tx81_roster_hash": manifest["tx81_roster_hash"],
                "battery_hashes": ",".join(
                    b["user_hash"] for b in plan["batteries"][module]
                ),
            }
        )

    # twin_yes (production) search usage and cost, when its batch outputs exist
    for module in MODULES:
        module_traces = (
            production_traces[production_traces["module"] == module]
            if not production_traces.empty
            else production_traces
        )
        row = {
            "date": execution_date,
            "module": module,
            "cell": "twin_yes",
            "alias": "production",
            "web_search": True,
            "model": plan["model"],
            "provider": plan["provider"],
            "status": "OK",
            "notes": "",
            "n_rows": manifest["modules"][module]["n_production_rows"],
            "n_chunks": len(module_chunks(module)),
            "tx81_rows": manifest["modules"][module]["n_tx81_in_drop"],
            "drop_arrival_utc": drop_arrival,
            "config_hash": manifest["config_hash"],
            "tx81_roster_hash": manifest["tx81_roster_hash"],
        }
        if module_traces.empty:
            row["status"] = "WARN"
            row["notes"] = (
                "production search traces unavailable (no batch output files)"
            )
        else:
            row.update(
                {
                    "n_returned": len(module_traces),
                    "search_call_share": float(
                        (module_traces["n_search_calls"] > 0).mean()
                    ),
                    "mean_searches_per_call": float(
                        module_traces["n_search_calls"].mean()
                    ),
                    "input_tokens": module_traces["input_tokens"].sum(),
                    "cached_input_tokens": module_traces["cached_input_tokens"].sum(),
                    "output_tokens": module_traces["output_tokens"].sum(),
                    "cost_usd": module_traces["cost_usd"].sum(min_count=1),
                }
            )
        log_rows.append(row)

    log = pd.DataFrame(log_rows)
    append_run_log(project_name, log)

    overall = max(log["status"], key=STATUS_CODES.get)
    for r in log.itertuples():
        cost = "" if pd.isna(r.cost_usd) else f" cost=${r.cost_usd:.2f}"
        n_returned = 0 if pd.isna(r.n_returned) else int(r.n_returned)
        n_expected = "-" if pd.isna(r.n_expected_calls) else int(r.n_expected_calls)
        timing = r.timing_flag if isinstance(r.timing_flag, str) else "-"
        print(
            f"  {r.status:4s} {r.module:16s} {r.cell:20s} n={n_returned}/{n_expected} "
            f"timing={timing}{cost}{' | ' + r.notes if r.notes else ''}"
        )
    total_cost = log.loc[log["cell"] != "twin_yes", "cost_usd"].sum(min_count=1)
    print(
        f"BASELINE {execution_date}: {overall} "
        f"(arm cost ${0 if pd.isna(total_cost) else total_cost:.2f}, "
        f"run log {data_path(project_name, BASELINE_RUN_LOG_FILE_X)})"
    )
    return STATUS_CODES[overall]


def append_run_log(project_name: str, log: pd.DataFrame) -> None:
    """
    Appends rows to the baseline run log, creating it if needed.

    Args:
        project_name (str): Name of the project directory.
        log (pd.DataFrame): Rows to append.

    Returns:
        None
    """
    path = data_path(project_name, BASELINE_RUN_LOG_FILE_X)
    if os.path.exists(path):
        log = pd.concat(
            [pd.read_csv(path, dtype=str, keep_default_na=False), log.astype(object)],
            ignore_index=True,
        )
    log.to_csv(path, index=False)


def remove_arm_batch_inputs(project_name: str, execution_date: str) -> int:
    """
    Deletes the batch input files (*_batch_input.jsonl) of a drop date's arms. The
    batch outputs are kept, and the inputs can be rebuilt from the prompt manifest.

    Args:
        project_name (str): Name of the project directory.
        execution_date (str): Drop date in DD-MM-YYYY format.

    Returns:
        int: Number of bytes freed.
    """
    batch_dir = data_path(
        project_name, baseline_execution_date(execution_date), "batch-files"
    )
    if not os.path.isdir(batch_dir):
        return 0
    freed = 0
    for name in os.listdir(batch_dir):
        if name.endswith("_batch_input.jsonl"):
            path = os.path.join(batch_dir, name)
            freed += os.path.getsize(path)
            os.remove(path)
    print(f"Removed batch request files from {batch_dir} ({freed / 1e6:.0f} MB).")
    return freed


def log_arm_failure(project_name: str, execution_date: str, message: str) -> None:
    """
    Records a FAIL in the baseline run log for a run that stopped before QC.

    Args:
        project_name (str): Name of the project directory.
        execution_date (str): Drop date in DD-MM-YYYY format.
        message (str): Failure message.

    Returns:
        None
    """
    append_run_log(
        project_name,
        pd.DataFrame(
            [
                {
                    "date": execution_date,
                    "module": "all",
                    "cell": "all",
                    "status": "FAIL",
                    "notes": message,
                    "run_end_utc": datetime.now(timezone.utc).isoformat(),
                }
            ]
        ),
    )

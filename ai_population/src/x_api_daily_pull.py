"""
X API v2 daily pull for the market-signals finfluencer panel.

Produces the data shapes the pipeline already consumes, so it slots into the
daily run as an additional step with no downstream changes:

  1. POSTS   -> <out>/<DD-MM-YYYY>/x_posts_<DD-MM-YYYY>.csv
     One row per post, in the pipeline's standard post schema plus the derived
     columns the profile-search step normally adds itself (account_id,
     hashtags, tagged_users). Drop-in compatible with
     perform_x_profile_search(local_file=...).

  2. PROFILES -> <out>/<DD-MM-YYYY>/x_profiles_<DD-MM-YYYY>.csv
     One row per handle, in the pipeline's standard profile-metadata schema.
     Written on Mondays by default (matching the weekly metadata cadence);
     see METADATA_MODE and the --metadata flag.

Schema columns X API v2 does not populate (left empty; none are consumed by
the interview pipeline): source (retired by X), coverPicture, canDm,
affiliatesHighlightedLabel, isAutomated, automatedBy.

Cost controls (per-post billing, $0.005/post returned; requests are free):
  - Per handle: paginate until PER_HANDLE_ORIGINALS original posts are
    collected. Page size shrinks to what is still needed so the API never
    returns (and bills) posts that would be thrown away.
  - Per run: hard stop once DAILY_BUDGET_POSTS posts have been returned
    (2,000 posts = $10.00/day ceiling). Handles are processed in descending
    influence+credibility order, so if the ceiling is ever hit, the lowest
    priority handles are the ones deferred (logged as one summary entry).
  - Every HTTP call has a timeout; exhausted retries and failed batches are
    written to <out>/<date>/x_failures_<date>.json, never dropped silently.

Setup:
  - pip install requests pandas
  - Set env var X_BEARER_TOKEN (or put X_BEARER_TOKEN=... in a .env next to
    this script).
  - Point PANEL_FILE at the 183-handle csv (account_id, inclusion_date,
    influence, credibility).
  - User-id lookups are cached in x_user_id_cache.csv next to this script
    (ids stored as strings; do not open/re-save this file in Excel).

Usage:
  python x_api_daily_pull.py                  # pulls yesterday (UTC)
  python x_api_daily_pull.py 22-07-2026      # pulls a specific UTC day
  python x_api_daily_pull.py 22-07-2026 --metadata  # force a profile refresh
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PANEL_FILE = BASE_DIR / "x_verified_finfluencer_profiles_sample_183.csv"
OUTPUT_DIR = BASE_DIR / "x_api_data"
UID_CACHE = BASE_DIR / "x_user_id_cache.csv"

PER_HANDLE_ORIGINALS = 20  # cap: latest N original posts per handle
DAILY_BUDGET_POSTS = 2000  # hard stop: 2,000 posts x $0.005 = $10.00/day
EXCLUDE = "retweets,replies"  # originals only; remove "replies" or
# "retweets" from this list to widen the pull
METADATA_MODE = "monday"  # "monday" | "daily" | "off"  (keyed on the
# run date, matching the weekly cache logic)
PRICE_PER_POST = 0.005

API = "https://api.x.com/2"
TWEET_FIELDS = (
    "created_at,public_metrics,entities,referenced_tweets,lang,"
    "conversation_id,in_reply_to_user_id,author_id"
)
USER_FIELDS = (
    "created_at,description,entities,location,name,pinned_tweet_id,"
    "profile_image_url,protected,public_metrics,url,username,"
    "verified,verified_type"
)
MAX_RETRIES = 3
HTTP_TIMEOUT = 30

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTHS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

POST_COLUMNS = [
    "type",
    "id",
    "url",
    "twitterUrl",
    "text",
    "source",
    "retweetCount",
    "replyCount",
    "likeCount",
    "quoteCount",
    "viewCount",
    "createdAt",
    "lang",
    "bookmarkCount",
    "isReply",
    "inReplyToId",
    "conversationId",
    "inReplyToUserId",
    "inReplyToUsername",
    "author",
    "entities",
    "account_id",
    "hashtags",
    "tagged_users",
]


def classic_datetime(iso_string: str) -> str:
    """ISO-8601 -> 'Tue Jul 22 14:33:57 +0000 2026', locale-independent."""
    dt = datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
    return (
        f"{DAYS[dt.weekday()]} {MONTHS[dt.month - 1]} {dt.day:02d} "
        f"{dt:%H:%M:%S} +0000 {dt.year}"
    )


def _read_token_from_env_file(env_file: Path):
    """Return the X_BEARER_TOKEN value from a .env-style file, or None."""
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("X_BEARER_TOKEN="):
                return line.split("=", 1)[1].strip()
    return None


def load_bearer() -> str:
    token = os.environ.get("X_BEARER_TOKEN")
    if not token:
        # Look next to this script first, then in the shared pipeline .env so the
        # same token works for standalone runs and the market-signals pipeline.
        for env_file in (BASE_DIR / ".env", BASE_DIR.parent / "config" / ".env"):
            token = _read_token_from_env_file(env_file)
            if token:
                break
    if not token:
        raise RuntimeError(
            "X_BEARER_TOKEN not set (env var, .env next to this script, or "
            "ai_population/config/.env)."
        )
    return token


def api_get(url: str, headers: dict, params: dict, failures: list, tag: str):
    """GET with timeout, 429 wait, retries on transport/5xx errors.

    Returns parsed json, or None after recording a failure entry."""
    last = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=HTTP_TIMEOUT)
        except requests.RequestException as exc:
            last = {"where": tag, "error": str(exc)}
            time.sleep(3 * attempt)
            continue
        if r.status_code == 429:
            reset = int(r.headers.get("x-rate-limit-reset", "0"))
            last = {"where": tag, "status": 429, "note": "rate limited"}
            time.sleep(min(max(1, reset - int(time.time()) + 2), 900))
            continue
        if r.status_code >= 500:
            last = {"where": tag, "status": r.status_code, "body": r.text[:200]}
            time.sleep(3 * attempt)
            continue
        if r.status_code != 200:
            failures.append(
                {"where": tag, "status": r.status_code, "body": r.text[:200]}
            )
            return None
        return r.json()
    failures.append(last or {"where": tag, "error": "retries exhausted"})
    return None


# ---------------------------------------------------------------------------
# User-id resolution (string-safe cache) + profile metadata
# ---------------------------------------------------------------------------
def load_uid_cache() -> dict:
    if not UID_CACHE.exists():
        return {}
    df = pd.read_csv(UID_CACHE, dtype=str, keep_default_na=False)
    return {h: u for h, u in zip(df["handle"], df["user_id"]) if h and u}


def save_uid_cache(cache: dict) -> None:
    pd.DataFrame(sorted(cache.items()), columns=["handle", "user_id"]).to_csv(
        UID_CACHE, index=False
    )


def lookup_users(handles, headers, failures, tag):
    """Batched /users/by lookup. Returns (user objects, per-batch misses)."""
    users = []
    for i in range(0, len(handles), 100):
        batch = handles[i : i + 100]
        body = api_get(
            f"{API}/users/by",
            headers,
            {"usernames": ",".join(batch), "user.fields": USER_FIELDS},
            failures,
            f"{tag} batch {i // 100}",
        )
        if body is None:
            for h in batch:
                failures.append(
                    {"where": tag, "handle": h, "note": "batch lookup failed"}
                )
            continue
        users.extend(body.get("data", []))
        for e in body.get("errors", []):
            failures.append(
                {"where": tag, "value": e.get("value"), "title": e.get("title")}
            )
        time.sleep(1)
    return users


def merge_into_cache(cache: dict, users: list, requested: list) -> None:
    by_lower = {u["username"].lower(): str(u["id"]) for u in users}
    for h in requested:
        uid = by_lower.get(h.lower())
        if uid:
            cache[h] = uid


def profile_row(u: dict) -> dict:
    """Map an X API v2 user object onto the pipeline's profile-metadata schema."""
    m = u.get("public_metrics", {})
    created = classic_datetime(u["created_at"]) if u.get("created_at") else ""
    verified_type = u.get("verified_type", "") or ""
    return {
        "id": str(u.get("id", "")),
        "name": u.get("name", ""),
        "account_id": u.get("username", ""),
        "location": u.get("location", ""),
        "url": u.get("url", ""),
        "description": u.get("description", ""),
        "entities": json.dumps(u.get("entities", {})),
        "protected": u.get("protected", False),
        "isVerified": verified_type in ("business", "government"),
        "isBlueVerified": bool(u.get("verified", False)),
        "verifiedType": verified_type,
        "followers": m.get("followers_count", 0),
        "following": m.get("following_count", 0),
        "favouritesCount": m.get("like_count", 0),
        "statusesCount": m.get("tweet_count", 0),
        "mediaCount": m.get("media_count", 0),
        "createdAt": created,
        "coverPicture": "",  # not exposed by X API v2
        "profilePicture": u.get("profile_image_url", ""),
        "canDm": "",  # not exposed by X API v2
        "affiliatesHighlightedLabel": "",  # not exposed by X API v2
        "isAutomated": "",  # not exposed by X API v2
        "automatedBy": "",  # not exposed by X API v2
        "pinnedTweetIds": (
            json.dumps([u["pinned_tweet_id"]]) if u.get("pinned_tweet_id") else "[]"
        ),
        "unavailable": False,
        "message": "",
        "unavailableReason": "",
    }


def fetch_x_api_profile_metadata(handles) -> pd.DataFrame:
    """Fetch X API v2 profile metadata for ``handles`` in the pipeline's
    profile-metadata schema (one row per handle, with an ``account_id`` column).

    Designed to be used as the ``fetch_fn`` for the pipeline's weekly metadata cache
    (``utils._get_weekly_cached_profile_metadata``), so Approach 2 metadata is refreshed
    on the same weekly cadence as Approach 1. Opportunistically refreshes the user-id
    cache so a subsequent posts pull can skip the lookups.
    """
    handles = list(handles)
    headers = {"Authorization": f"Bearer {load_bearer()}"}
    failures: list = []
    users = lookup_users(handles, headers, failures, "profiles")
    cache = load_uid_cache()
    merge_into_cache(cache, users, handles)
    save_uid_cache(cache)
    return pd.DataFrame([profile_row(u) for u in users])


# ---------------------------------------------------------------------------
# Posts
# ---------------------------------------------------------------------------
def entities_v1(t: dict) -> dict:
    """Map v2 entities onto the v1-style shapes the pipeline parsers expect."""
    ent = t.get("entities", {}) or {}
    return {
        "hashtags": [{"text": h.get("tag", "")} for h in ent.get("hashtags", [])],
        "user_mentions": [
            {"screen_name": m.get("username", ""), "id_str": str(m.get("id", ""))}
            for m in ent.get("mentions", [])
        ],
        "urls": [
            {
                "url": u.get("url", ""),
                "expanded_url": u.get("expanded_url", ""),
                "display_url": u.get("display_url", ""),
            }
            for u in ent.get("urls", [])
        ],
    }


def post_row(t: dict, handle: str, user_id: str) -> dict:
    m = t.get("public_metrics", {})
    refs = t.get("referenced_tweets", []) or []
    ents = entities_v1(t)
    return {
        "type": "tweet",
        "id": str(t["id"]),
        "url": f"https://x.com/{handle}/status/{t['id']}",
        "twitterUrl": f"https://twitter.com/{handle}/status/{t['id']}",
        "text": t.get("text", ""),
        "source": "",  # retired by X API v2
        "retweetCount": m.get("retweet_count", 0),
        "replyCount": m.get("reply_count", 0),
        "likeCount": m.get("like_count", 0),
        "quoteCount": m.get("quote_count", 0),
        "viewCount": m.get("impression_count", 0),
        "createdAt": classic_datetime(t["created_at"]),
        "lang": t.get("lang", ""),
        "bookmarkCount": m.get("bookmark_count", 0),
        "isReply": any(r.get("type") == "replied_to" for r in refs),
        "inReplyToId": next(
            (r["id"] for r in refs if r.get("type") == "replied_to"), ""
        ),
        "conversationId": str(t.get("conversation_id", "")),
        "inReplyToUserId": str(t.get("in_reply_to_user_id", "") or ""),
        "inReplyToUsername": "",
        "author": json.dumps({"userName": handle, "id": str(user_id)}),
        "entities": json.dumps(ents),
        # Derived columns the pipeline's API branch normally adds itself:
        "account_id": handle,
        "hashtags": json.dumps([h["text"] for h in ents["hashtags"]]),
        "tagged_users": json.dumps([u["screen_name"] for u in ents["user_mentions"]]),
    }


def pull_handle(
    handle: str,
    user_id: str,
    start: str,
    end: str,
    headers: dict,
    failures: list,
    budget_left: int,
):
    """Collect up to PER_HANDLE_ORIGINALS originals inside the window.

    Returns (rows, posts_billed). Page size is trimmed to what is still
    needed so the API never returns posts that would be discarded; every
    returned post therefore counts against the budget exactly once."""
    cap = min(PER_HANDLE_ORIGINALS, budget_left)
    rows = []
    billed = 0
    next_token = None
    while len(rows) < cap:
        page_size = max(5, min(100, cap - len(rows)))
        params = {
            "start_time": start,
            "end_time": end,
            "max_results": page_size,
            "exclude": EXCLUDE,
            "tweet.fields": TWEET_FIELDS,
        }
        if next_token:
            params["pagination_token"] = next_token
        body = api_get(
            f"{API}/users/{user_id}/tweets",
            headers,
            params,
            failures,
            f"timeline {handle}",
        )
        if body is None:
            break
        data = body.get("data", [])
        billed += len(data)
        for t in data:
            if len(rows) < cap:
                rows.append(post_row(t, handle, user_id))
        next_token = body.get("meta", {}).get("next_token")
        if not next_token:
            break
    return rows, billed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_daily_pull(
    date_tag: str = None,
    force_metadata: bool = False,
    panel_file=None,
    output_dir=None,
    metadata_mode: str = None,
):
    """Run the X API v2 daily pull for a single UTC day.

    Reusable entry point behind both the CLI (``main``) and the market-signals
    pipeline integration. Writes the same artefacts the CLI does (posts CSV,
    optional profiles CSV, run summary and failures JSON).

    Args:
        date_tag: Day to pull in DD-MM-YYYY (UTC). Defaults to yesterday (UTC).
        force_metadata: Force a profile-metadata refresh regardless of weekday.
        panel_file: Panel CSV of handles (account_id[, influence, credibility]).
            Defaults to PANEL_FILE next to this script.
        output_dir: Root output folder; a <DD-MM-YYYY> subfolder is created under
            it. Defaults to OUTPUT_DIR next to this script.
        metadata_mode: Overrides the module-level METADATA_MODE for this run
            ("monday" | "daily" | "off"). Pass "off" to run posts-only when profile
            metadata is handled elsewhere (e.g. the pipeline's weekly metadata cache).

    Returns:
        (posts_file, profiles_file) as Paths. profiles_file is None when no
        metadata refresh happened this run.
    """
    panel_file = Path(panel_file) if panel_file else PANEL_FILE
    output_root = Path(output_dir) if output_dir else OUTPUT_DIR
    mode = metadata_mode if metadata_mode is not None else METADATA_MODE

    if date_tag:
        day = datetime.strptime(date_tag, "%d-%m-%Y").date()
    else:
        day = (datetime.now(timezone.utc) - timedelta(days=1)).date()
    date_tag = day.strftime("%d-%m-%Y")
    start = f"{day.isoformat()}T00:00:00Z"
    end = f"{(day + timedelta(days=1)).isoformat()}T00:00:00Z"

    headers = {"Authorization": f"Bearer {load_bearer()}"}
    failures: list = []

    panel = pd.read_csv(panel_file)
    panel["priority"] = pd.to_numeric(panel.get("influence"), errors="coerce").fillna(
        0
    ) + pd.to_numeric(panel.get("credibility"), errors="coerce").fillna(0)
    panel = panel.sort_values("priority", ascending=False)
    handles = panel["account_id"].astype(str).tolist()

    out_dir = output_root / date_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = load_uid_cache()
    do_metadata = (
        force_metadata
        or mode == "daily"
        or (mode == "monday" and datetime.now(timezone.utc).weekday() == 0)
    )

    profiles_file = None
    if do_metadata:
        # One lookup pass serves both jobs: refresh profiles AND fill the
        # uid cache, so no handle is billed twice for lookups.
        users = lookup_users(handles, headers, failures, "profiles")
        merge_into_cache(cache, users, handles)
        profiles_file = out_dir / f"x_profiles_{date_tag}.csv"
        pd.DataFrame([profile_row(u) for u in users]).to_csv(profiles_file, index=False)
    else:
        missing = [h for h in handles if h not in cache]
        if missing:
            users = lookup_users(missing, headers, failures, "users/by")
            merge_into_cache(cache, users, missing)
    save_uid_cache(cache)

    unresolved = [h for h in handles if h not in cache]
    for h in unresolved:
        failures.append(
            {"where": "resolve", "handle": h, "note": "no user id (suspended/renamed?)"}
        )

    all_rows = []
    posts_billed = 0
    deferred = []
    for handle in handles:
        if handle in unresolved:
            continue
        budget_left = DAILY_BUDGET_POSTS - posts_billed
        if budget_left <= 0:
            deferred.append(handle)
            continue
        rows, billed = pull_handle(
            handle, cache[handle], start, end, headers, failures, budget_left
        )
        all_rows.extend(rows)
        posts_billed += billed
        time.sleep(0.2)
    if deferred:
        failures.append(
            {
                "where": "budget",
                "note": f"daily post budget reached; "
                f"{len(deferred)} handles deferred",
                "handles": deferred,
            }
        )

    posts = pd.DataFrame(all_rows, columns=POST_COLUMNS)
    posts_file = out_dir / f"x_posts_{date_tag}.csv"
    posts.to_csv(posts_file, index=False)
    (out_dir / f"x_failures_{date_tag}.json").write_text(
        json.dumps(failures, indent=2), encoding="utf-8"
    )

    summary = {
        "date": date_tag,
        "handles_requested": len(handles),
        "handles_unresolved": len(unresolved),
        "handles_deferred_by_budget": len(deferred),
        "posts_kept": len(posts),
        "posts_billed": posts_billed,
        "active_handles": int(posts["account_id"].nunique()) if len(posts) else 0,
        "post_cost_usd": round(posts_billed * PRICE_PER_POST, 2),
        "budget_ceiling_usd": round(DAILY_BUDGET_POSTS * PRICE_PER_POST, 2),
        "metadata_refreshed": bool(do_metadata),
        "failures": len(failures),
    }
    (out_dir / f"x_run_summary_{date_tag}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Posts: {posts_file}")
    return posts_file, profiles_file


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force_metadata = "--metadata" in sys.argv
    try:
        run_daily_pull(
            date_tag=args[0] if args else None,
            force_metadata=force_metadata,
        )
    except RuntimeError as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()

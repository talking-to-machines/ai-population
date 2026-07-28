# X API Daily Pull

Daily collection of the market-signals finfluencer panel's posts directly from
X API v2, optimized for cost. It pulls the same kind of data the abundance step
collects (latest posts per handle for the day, plus profile metadata), but it
bills only for posts that actually exist: quiet handles cost near zero, dead
handles cost nothing, and there is no padding. Measured on the 183-handle
panel this runs about $7 per day with a hard $10 daily ceiling, and captures
98%+ of the original posts the abundance pull delivers for the same day.

## How it is meant to be used

1. **Integrate after the abundance pull** in the daily run. This script is an
   additional step; it does not touch or replace anything by itself.

   ```
   python x_api_daily_pull.py            # pulls yesterday (UTC)
   python x_api_daily_pull.py 22-07-2026 # pulls a specific UTC day
   ```

2. **Parallel run for one week.** Let both collections run daily. The script
   writes to its own folder (`x_api_data/<date>/`) and produces a posts CSV
   and, on Mondays, a profiles CSV, in the same column schema the pipeline's
   profile-search step consumes. Compare the two datasets over the week.

3. **Downstream verification is on the pipeline side.** The posts CSV is
   designed to load through `perform_x_profile_search(local_file=...)` without
   changes (column names, date format, and the derived account_id, hashtags
   and tagged_users columns are all matched), but the downstream steps should
   be verified against it during the parallel week before any cutover.

4. **The toggle.** Switching sources is one argument in the daily run:

   ```python
   # X API as the source:
   perform_x_profile_search(..., local_file="x_api_data/<date>/x_posts_<date>.csv")

   # abundance as the source (current behaviour):
   perform_x_profile_search(...)   # no local_file argument
   ```

   Keep both paths available so either source can be enabled at any time.

## Setup

- `pip install requests pandas`
- Set the env var `X_BEARER_TOKEN`, or put `X_BEARER_TOKEN=...` in a `.env`
  file next to the script (the token is provided separately; it is never part
  of this repository).
- Place the panel file `x_verified_finfluencer_profiles_sample_183.csv`
  (columns: account_id, inclusion_date, influence, credibility) next to the
  script, or adjust `PANEL_FILE` in the config block.

## Cost controls

- Per handle: latest 20 original posts (no retweets, no replies), collected by
  paginating until 20 originals are found. Change `PER_HANDLE_ORIGINALS` or
  `EXCLUDE` in the config block to widen the pull.
- Per run: hard stop at 2,000 returned posts, which caps any single day at
  $10.00. Handles are processed in influence + credibility order, so if the
  ceiling is ever reached, the lowest-priority handles are the ones deferred.
- Profile metadata refreshes on Mondays only (`METADATA_MODE`), matching the
  weekly metadata cadence; `--metadata` forces a refresh on any day.

## Outputs per run

| File | Contents |
| --- | --- |
| `x_posts_<date>.csv` | one row per original post, pipeline schema |
| `x_profiles_<date>.csv` | one row per handle (Mondays or `--metadata`) |
| `x_run_summary_<date>.json` | counts, cost, and config for the run |
| `x_failures_<date>.json` | every lookup or fetch problem; nothing fails silently |

Columns the schema carries that X API v2 does not populate (left empty, not
used by the interview pipeline): source, coverPicture, canDm,
affiliatesHighlightedLabel, isAutomated, automatedBy.

"""
End-of-run cleanup and Dropbox uploads for the X market-signals pipeline.

Uploads go through the Dropbox API, so the same code runs on a laptop and on a
server without the Dropbox desktop client. Run once to get a refresh token:

    python -m ai_population.src.dropbox_sync --authorize
"""

import argparse
import hashlib
import os
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd

from ai_population.config.base_config import (
    DROPBOX_APP_KEY,
    DROPBOX_APP_SECRET,
    DROPBOX_REFRESH_TOKEN,
)
from ai_population.config.market_signals_config import (
    INTERMEDIATE_FILES_X,
    DROPBOX_TRADING_APP_FOLDER_X,
    DROPBOX_TRADING_APP_FILES_X,
    DROPBOX_PROJECT_FOLDER_X,
    DROPBOX_SYNC_LOG_FILE_X,
)
from ai_population.src.baseline_arms import (
    STATUS_CODES,
    baseline_execution_date,
    data_path,
    module_chunks,
)

CHUNK_SIZE = 8 * 1024 * 1024
HASH_BLOCK_SIZE = 4 * 1024 * 1024
UPLOAD_WORKERS = 4
UPLOAD_ATTEMPTS = 4
RETRY_DELAYS_SECONDS = (10, 30, 90)
DATE_PATTERN = re.compile(r"^\d{2}-\d{2}-\d{4}$")
AUTH_ERROR_HELP = (
    "Dropbox rejected the credentials in ai_population/config/.env. Run "
    "`python -m ai_population.src.dropbox_sync --authorize`, paste the code into its "
    "prompt and put the refresh token it prints in DROPBOX_REFRESH_TOKEN."
)

_thread_state = threading.local()


def get_dropbox_client():
    """
    Creates a Dropbox client from the refresh token in .env; the client renews its
    access token automatically. Each thread gets its own client.

    Returns:
        dropbox.Dropbox: The client.

    Raises:
        RuntimeError: If a Dropbox credential is missing from .env.
    """
    if getattr(_thread_state, "client", None) is None:
        missing = [
            name
            for name, value in (
                ("DROPBOX_APP_KEY", DROPBOX_APP_KEY),
                ("DROPBOX_APP_SECRET", DROPBOX_APP_SECRET),
                ("DROPBOX_REFRESH_TOKEN", DROPBOX_REFRESH_TOKEN),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                f"Missing {', '.join(missing)} in ai_population/config/.env "
                f"(run `python -m ai_population.src.dropbox_sync --authorize`)."
            )
        import dropbox

        _thread_state.client = dropbox.Dropbox(
            oauth2_refresh_token=DROPBOX_REFRESH_TOKEN,
            app_key=DROPBOX_APP_KEY,
            app_secret=DROPBOX_APP_SECRET,
            max_retries_on_error=4,
            max_retries_on_rate_limit=8,
            timeout=300,
        )
    return _thread_state.client


def dropbox_content_hash(path: str) -> str:
    """
    Computes a file's Dropbox content hash (SHA-256 of the SHA-256 digests of its
    4 MB blocks), which Dropbox reports for every stored file.

    Args:
        path (str): Local file path.

    Returns:
        str: The hex content hash.
    """
    block_hashes = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(HASH_BLOCK_SIZE)
            if not block:
                break
            block_hashes.update(hashlib.sha256(block).digest())
    return block_hashes.hexdigest()


def _remote_content_hash(client, dropbox_path: str):
    """
    Gets the content hash of a file in Dropbox.

    Args:
        client (dropbox.Dropbox): Dropbox client.
        dropbox_path (str): Path in Dropbox.

    Returns:
        str: The content hash, or None if the file does not exist.
    """
    from dropbox.exceptions import ApiError

    try:
        return getattr(client.files_get_metadata(dropbox_path), "content_hash", None)
    except ApiError as e:
        error = e.error
        if error.is_path() and error.get_path().is_not_found():
            return None
        raise


def _upload_once(client, local_path: str, dropbox_path: str, size: int):
    """
    Uploads a file in one call, or in 8 MB chunks through an upload session when it
    is larger than one chunk. The file appears in Dropbox only once fully uploaded.

    Args:
        client (dropbox.Dropbox): Dropbox client.
        local_path (str): Local file path.
        dropbox_path (str): Destination path in Dropbox.
        size (int): File size in bytes.

    Returns:
        dropbox.files.FileMetadata: Metadata of the uploaded file.
    """
    from dropbox.files import CommitInfo, UploadSessionCursor, WriteMode

    with open(local_path, "rb") as f:
        if size <= CHUNK_SIZE:
            return client.files_upload(
                f.read(), dropbox_path, mode=WriteMode.overwrite, mute=True
            )
        session = client.files_upload_session_start(f.read(CHUNK_SIZE))
        cursor = UploadSessionCursor(session_id=session.session_id, offset=f.tell())
        commit = CommitInfo(path=dropbox_path, mode=WriteMode.overwrite, mute=True)
        while size - f.tell() > CHUNK_SIZE:
            client.files_upload_session_append_v2(f.read(CHUNK_SIZE), cursor)
            cursor.offset = f.tell()
        return client.files_upload_session_finish(f.read(), cursor, commit)


def upload_file(local_path: str, dropbox_path: str, dry_run: bool = False) -> dict:
    """
    Uploads a file to Dropbox, overwriting any existing file, unless Dropbox already
    has an identical copy. Failed uploads are retried with back-off and every upload
    is checked against the local content hash.

    Args:
        local_path (str): Local file path.
        dropbox_path (str): Destination path in Dropbox.
        dry_run (bool, optional): Only report what would be uploaded. Defaults to False.

    Returns:
        dict: Log row with local_path, dropbox_path, bytes, status ("uploaded",
            "unchanged", "would upload" or "failed"), seconds and error.
    """
    started = time.monotonic()
    row = {
        "action": "upload",
        "local_path": local_path,
        "dropbox_path": dropbox_path,
        "bytes": os.path.getsize(local_path),
        "status": "would upload",
        "error": "",
    }
    if dry_run:
        row["seconds"] = 0.0
        return row
    try:
        client = get_dropbox_client()
        local_hash = dropbox_content_hash(local_path)
        if _remote_content_hash(client, dropbox_path) == local_hash:
            row["status"] = "unchanged"
        else:
            for attempt in range(1, UPLOAD_ATTEMPTS + 1):
                try:
                    metadata = _upload_once(
                        client, local_path, dropbox_path, row["bytes"]
                    )
                    break
                except Exception as e:
                    if attempt == UPLOAD_ATTEMPTS or _is_auth_error(e):
                        raise
                    delay = RETRY_DELAYS_SECONDS[
                        min(attempt, len(RETRY_DELAYS_SECONDS)) - 1
                    ]
                    print(
                        f"Upload of {dropbox_path} failed ({e}); retrying in {delay}s."
                    )
                    time.sleep(delay)
            if metadata.content_hash != local_hash:
                raise RuntimeError(
                    "content hash in Dropbox does not match the local file"
                )
            row["status"] = "uploaded"
    except Exception as e:
        row["status"] = "failed"
        row["error"] = _error_text(e)
    row["seconds"] = round(time.monotonic() - started, 1)
    return row


def _is_auth_error(e: Exception) -> bool:
    """
    Checks whether an exception is a Dropbox authentication error.

    Args:
        e (Exception): The exception.

    Returns:
        bool: True for dropbox.exceptions.AuthError.
    """
    return type(e).__name__ == "AuthError"


def _error_text(e: Exception) -> str:
    """
    Describes an exception for the sync log, with setup guidance for rejected credentials.

    Args:
        e (Exception): The exception.

    Returns:
        str: The description.
    """
    return AUTH_ERROR_HELP if _is_auth_error(e) else str(e)


def upload_folder(local_dir: str, dropbox_dir: str, dry_run: bool = False) -> list:
    """
    Uploads every non-hidden file in a folder and its subfolders, keeping the folder
    structure.

    Args:
        local_dir (str): Local folder.
        dropbox_dir (str): Destination folder in Dropbox.
        dry_run (bool, optional): Only report what would be uploaded. Defaults to False.

    Returns:
        list: One log row per file (see upload_file).
    """
    jobs = []
    for root, dirs, files in os.walk(local_dir):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(files):
            if name.startswith("."):
                continue
            local_path = os.path.join(root, name)
            relative = os.path.relpath(local_path, local_dir).replace(os.sep, "/")
            jobs.append((local_path, f"{dropbox_dir}/{relative}"))
    with ThreadPoolExecutor(max_workers=UPLOAD_WORKERS) as executor:
        return list(
            executor.map(lambda job: upload_file(job[0], job[1], dry_run), jobs)
        )


def delete_intermediate_files(
    project_name: str, execution_date: str, dry_run: bool = False
) -> list:
    """
    Deletes the run's intermediate files (INTERMEDIATE_FILES_X) from the drop folder.

    Args:
        project_name (str): Name of the project directory.
        execution_date (str): Drop date in DD-MM-YYYY format.
        dry_run (bool, optional): Only report what would be deleted. Defaults to False.

    Returns:
        list: One log row per target with status "deleted", "would delete" or "missing".

    Raises:
        ValueError: If execution_date is not in DD-MM-YYYY format.
    """
    if not DATE_PATTERN.match(execution_date):
        raise ValueError(f"Invalid drop date {execution_date!r}; expected DD-MM-YYYY.")
    drop_dir = os.path.realpath(data_path(project_name, execution_date))
    names = []
    for pattern in INTERMEDIATE_FILES_X:
        if "{i}" in pattern:
            names += [
                pattern.format(d=execution_date, i=i)
                for i in module_chunks("daily_stock_pick")
            ]
        else:
            names.append(pattern.format(d=execution_date))

    rows = []
    for name in names:
        path = os.path.realpath(os.path.join(drop_dir, name))
        if os.path.dirname(path) != drop_dir:
            raise ValueError(f"Refusing to delete {path}: outside {drop_dir}.")
        row = {"action": "delete", "local_path": path, "dropbox_path": "", "error": ""}
        if not os.path.exists(path):
            row["status"] = "missing"
        elif dry_run:
            row["status"] = "would delete"
        else:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            row["status"] = "deleted"
        rows.append(row)
    return rows


def _log_and_report(
    project_name: str,
    execution_date: str,
    stage: str,
    rows: list,
    notes: list,
    dry_run: bool = False,
) -> int:
    """
    Appends the rows to the Dropbox sync log and prints one summary line. A dry run
    prints every row instead of logging it.

    Args:
        project_name (str): Name of the project directory.
        execution_date (str): Drop date in DD-MM-YYYY format.
        stage (str): Pipeline stage the rows belong to.
        rows (list): Log rows from the upload and delete functions.
        notes (list): Warnings to report, such as missing folders.
        dry_run (bool, optional): Whether the rows come from a dry run. Defaults to False.

    Returns:
        int: Exit status, 0 OK / 1 WARN / 2 FAIL.
    """
    failed = [r for r in rows if r["status"] == "failed"]
    status = "FAIL" if failed else ("WARN" if notes else "OK")
    if dry_run:
        for r in rows:
            size = f" ({r['bytes'] / 1e6:.1f} MB)" if "bytes" in r else ""
            target = f" -> {r['dropbox_path']}" if r["dropbox_path"] else ""
            print(f"  {r['status']}: {r['local_path']}{target}{size}")
    elif rows:
        log = pd.DataFrame(rows)
        log.insert(0, "stage", stage)
        log.insert(0, "date", execution_date)
        log["logged_at_utc"] = datetime.now(timezone.utc).isoformat()
        path = data_path(project_name, DROPBOX_SYNC_LOG_FILE_X)
        if os.path.exists(path):
            log = pd.concat(
                [pd.read_csv(path, dtype=str), log.astype(object)], ignore_index=True
            )
        log.to_csv(path, index=False)

    counts = pd.Series([r["status"] for r in rows], dtype=object).value_counts()
    uploaded_mb = (
        sum(r.get("bytes", 0) for r in rows if r["status"] == "uploaded") / 1e6
    )
    summary = ", ".join(f"{n} {s}" for s, n in counts.items()) or "nothing to do"
    print(
        f"DROPBOX {execution_date} {stage}: {status} ({summary}, {uploaded_mb:.0f} MB uploaded)"
    )
    for r in failed:
        print(f"  failed: {r['local_path']} -> {r['dropbox_path']}: {r['error']}")
    for note in notes:
        print(f"  warning: {note}")
    return STATUS_CODES[status]


def sync_after_step_9(
    project_name: str, execution_date: str, cleanup: bool, dry_run: bool = False
) -> int:
    """
    Uploads the daily stock pick file to the trading app folder and, when the
    baseline arms are not run, deletes the intermediate files.

    Args:
        project_name (str): Name of the project directory.
        execution_date (str): Drop date in DD-MM-YYYY format.
        cleanup (bool): Delete the intermediate files now.
        dry_run (bool, optional): Only report what would be done. Defaults to False.

    Returns:
        int: Exit status, 0 OK / 1 WARN / 2 FAIL.
    """
    rows, notes = [], []
    try:
        for pattern in DROPBOX_TRADING_APP_FILES_X:
            name = pattern.format(d=execution_date)
            local_path = data_path(project_name, execution_date, name)
            if os.path.exists(local_path):
                rows.append(
                    upload_file(
                        local_path, f"{DROPBOX_TRADING_APP_FOLDER_X}/{name}", dry_run
                    )
                )
            else:
                notes.append(f"{local_path} not found")
        if cleanup:
            rows += delete_intermediate_files(project_name, execution_date, dry_run)
    except Exception as e:
        rows.append(
            {
                "action": "sync",
                "local_path": "",
                "dropbox_path": "",
                "status": "failed",
                "error": _error_text(e),
            }
        )
    return _log_and_report(
        project_name, execution_date, "after step 9", rows, notes, dry_run
    )


def sync_after_step_10(
    project_name: str,
    execution_date: str,
    cleanup: bool,
    expect_baseline: bool,
    dry_run: bool = False,
) -> int:
    """
    Deletes the intermediate files (if not done after step 9), then uploads the
    baseline folder to the trading app folder, and the drop and baseline folders to
    the project folder.

    Args:
        project_name (str): Name of the project directory.
        execution_date (str): Drop date in DD-MM-YYYY format.
        cleanup (bool): Delete the intermediate files before uploading.
        expect_baseline (bool): Warn if the baseline folder does not exist.
        dry_run (bool, optional): Only report what would be done. Defaults to False.

    Returns:
        int: Exit status, 0 OK / 1 WARN / 2 FAIL.
    """
    rows, notes = [], []
    try:
        if cleanup:
            rows += delete_intermediate_files(project_name, execution_date, dry_run)
        drop_dir = data_path(project_name, execution_date)
        baseline_name = baseline_execution_date(execution_date)
        baseline_dir = data_path(project_name, baseline_name)
        if os.path.isdir(baseline_dir):
            rows += upload_folder(
                baseline_dir, f"{DROPBOX_TRADING_APP_FOLDER_X}/{baseline_name}", dry_run
            )
        elif expect_baseline:
            notes.append(f"{baseline_dir} not found")
        if os.path.isdir(drop_dir):
            rows += upload_folder(
                drop_dir, f"{DROPBOX_PROJECT_FOLDER_X}/{execution_date}", dry_run
            )
        else:
            notes.append(f"{drop_dir} not found")
        if os.path.isdir(baseline_dir):
            rows += upload_folder(
                baseline_dir, f"{DROPBOX_PROJECT_FOLDER_X}/{baseline_name}", dry_run
            )
    except Exception as e:
        rows.append(
            {
                "action": "sync",
                "local_path": "",
                "dropbox_path": "",
                "status": "failed",
                "error": _error_text(e),
            }
        )
    return _log_and_report(
        project_name, execution_date, "after step 10", rows, notes, dry_run
    )


def authorize() -> None:
    """
    Runs the one-time Dropbox authorization and prints the refresh token to add to
    ai_population/config/.env. Needs DROPBOX_APP_KEY and DROPBOX_APP_SECRET in .env.

    Returns:
        None
    """
    from dropbox import DropboxOAuth2FlowNoRedirect

    if not DROPBOX_APP_KEY or not DROPBOX_APP_SECRET:
        raise RuntimeError(
            "Add DROPBOX_APP_KEY and DROPBOX_APP_SECRET to ai_population/config/.env first."
        )
    flow = DropboxOAuth2FlowNoRedirect(
        DROPBOX_APP_KEY, consumer_secret=DROPBOX_APP_SECRET, token_access_type="offline"
    )
    print(f"1. Open this link and click Allow:\n   {flow.start()}")
    code = input("2. Paste the authorization code here: ").strip()
    result = flow.finish(code)
    print(
        f"3. Add this line to ai_population/config/.env:\nDROPBOX_REFRESH_TOKEN={result.refresh_token}"
    )


def check_connection() -> None:
    """
    Checks the Dropbox credentials and that both destination folders exist.

    Returns:
        None
    """
    client = get_dropbox_client()
    try:
        account = client.users_get_current_account()
    except Exception as e:
        if _is_auth_error(e):
            raise SystemExit(AUTH_ERROR_HELP)
        raise
    print(f"Connected to Dropbox as {account.email}")
    for folder in (DROPBOX_TRADING_APP_FOLDER_X, DROPBOX_PROJECT_FOLDER_X):
        client.files_get_metadata(folder)
        print(f"  found {folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dropbox setup for the X market signals pipeline."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--authorize", action="store_true", help="Get a Dropbox refresh token."
    )
    group.add_argument(
        "--check",
        action="store_true",
        help="Check the credentials and destination folders.",
    )
    args = parser.parse_args()
    if args.authorize:
        authorize()
    else:
        check_connection()

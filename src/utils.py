import pandas as pd
from datetime import datetime
from pathlib import Path
from config import (
    DATA_DIR, RESULTS_DIR, PUBMED_DATA_FILE_BASE, REDDIT_DATA_FILE_BASE,
    STUDY_START_DATE, STUDY_END_DATE, REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET
)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_latest_data_file(pattern):
    """Helper function to get the latest file matching a pattern"""
    files = list(DATA_DIR.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda x: x.stat().st_ctime)

def get_latest_data_filepath(base_name: str) -> Path or None:
    """Finds the most recently created file matching the base_name pattern."""
    pattern = f"{base_name}*.csv"
    files = list(DATA_DIR.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda x: x.stat().st_ctime)

def load_data(filename_or_base: str, is_timestamped=False, **kwargs) -> pd.DataFrame or None:
    """Loads data, handling both fixed and timestamped files."""
    if is_timestamped:
        path = get_latest_data_filepath(filename_or_base)
    else:
        path = DATA_DIR / filename_or_base

    if path and path.exists():
        print(f"  > Loading data from: {path.name}")
        try:
            return pd.read_csv(path, **kwargs)
        except Exception as e:
            print(f"[Error] Failed to read CSV {path.name}: {e}")
            return None
    else:
        print(f"[Error] Data file not found for: {filename_or_base}")
        return None

def validate_config():
    """Validate that essential configuration is present"""
    errors = []

    # Check required directories
    if not DATA_DIR.exists():
        errors.append(f"DATA_DIR does not exist: {DATA_DIR}")
    if not RESULTS_DIR.exists():
        errors.append(f"RESULTS_DIR does not exist: {RESULTS_DIR}")

    # Check Reddit credentials (warn but don't fail if missing)
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        print("Warning: Reddit API credentials not found. Reddit data collection will fail.")

    return errors

# ==============================================================================
# DATA PROCESSING FUNCTIONS
# ==============================================================================

def process_submission(submission, search_term=None):
    """Processes a PRAW submission object into a dictionary row."""
    # Create the main post data
    post_data = {
        'subreddit': submission.subreddit.display_name,
        'search_term': search_term,
        'post_id': submission.id,
        'post_title': submission.title,
        'post_text': submission.selftext,
        'timestamp': datetime.fromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
        'type': 'post',
        'comment_text': None
    }

    # Initialize list with the post data
    all_data = [post_data]

    # Get comments if they exist
    try:
        submission.comments.replace_more(limit=0)

        for comment in submission.comments.list():
            # Only include top-level comments for simplicity
            if comment.parent_id == submission.fullname:
                comment_data = {
                    'subreddit': submission.subreddit.display_name,
                    'search_term': search_term,
                    'post_id': submission.id,
                    'post_title': submission.title,
                    'post_text': None,  # Post text is only on the post row
                    'timestamp': datetime.fromtimestamp(comment.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                    'type': 'comment',
                    'comment_text': comment.body
                }
                all_data.append(comment_data)

    except Exception as e:
        print(f"  Warning: Could not process comments for {submission.id}: {e}")

    return all_data

# ==============================================================================
# FILE MANAGEMENT FUNCTIONS
# ==============================================================================

def get_latest_timestamp_filepath(base_name: str) -> Path:
    """Generates a timestamped filepath for saving data using consistent naming."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # Use consistent naming: base_name + timestamp for ALL files
    filename = f"{base_name}{timestamp}.csv"

    return DATA_DIR / filename




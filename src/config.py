from pathlib import Path
from dotenv import load_dotenv
import os

# ==============================================================================
# 1. PATH CONFIGURATION
# ==============================================================================
# Resolve the project root (up one level from src/config.py)
PROJECT_ROOT = Path(__file__).parent.parent

# Define main project directories using Path objects
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Ensure data and results folders exist locally
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ==============================================================================
# 2. SECRET CONFIGURATION (.env)
# ==============================================================================
# Load environment variables from .env located in the SAME folder (src/)
load_dotenv(dotenv_path=Path(__file__).parent / '.env')

# API Keys/Secrets - Access these from other scripts using import config
REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.environ.get("REDDIT_USER_AGENT", "masld_research_bot_v1.0")

# ==============================================================================
# 3. PROJECT DATASETS & RESULTS CONFIGURATION
# ==============================================================================

# --- File Names for Data Directory ---
GOOGLE_TRENDS_FILE = "google_trends_initial_data.csv"
STOCK_DATA_FILE = "stock_prices.csv"
REDDIT_DATA_FILE_BASE = "reddit_data_"
PUBMED_DATA_FILE_BASE = "pubmed_masld_articles_"

# File patterns that match my actual file naming conventions
REDDIT_DATA_PATTERN = "reddit_data_2023_2025_*.csv"  # Matches: reddit_data_2023_2025_YYYYMMDD_HHMM.csv
PUBMED_DATA_PATTERN = "pubmed_masld_articles_*.csv"  # Matches: pubmed_masld_articles_YYYYMMDD_HHMMSS.csv

# --- Analysis Subdirectories ---
GOOGLE_TRENDS_ANALYSIS_SUBDIR = "google_trends"
PUBMED_ANALYSIS_SUBDIR = "pubmed"
MEDIA_CLOUD_ANALYSIS_SUBDIR = "media_cloud"
REDDIT_ANALYSIS_SUBDIR = "reddit_sentiment"
STOCK_ANALYSIS_SUBDIR = "stock_analysis"

# --- Media Cloud Configuration ---
MEDIA_CLOUD_DATASETS = {
    'disease': 'disease_focused',
    'resmetirom': 'resmetirom_focused', 
    'glp1': 'glp1_focused'
}

# ==============================================================================
# 4. KEY DATES & EVENTS (Centralized for all analyses)
# ==============================================================================
FDA_EVENT_DATES = {
    'Resmetirom Approval': '2024-03-14',
    'GLP-1 Agonists Approval': '2025-08-15'
}

# PubMed-specific date constants (for backward compatibility)
RESMETIROM_APPROVAL_DATE = '2024-03'
SEMAGLUTIDE_APPROVAL_DATE = '2025-08'

# ==============================================================================
# 5. ANALYSIS PARAMETERS
# ==============================================================================

# Google Trends parameters
GOOGLE_TRENDS_KEYWORDS = ['MASLD', 'NAFLD', 'Rezdiffra', 'Wegovy', 'Ozempic']
GOOGLE_TRENDS_TIMEFRAME = '2023-01-01 2025-10-28'

# Google Trends analysis parameters (regardless of timeframe)
GOOGLE_TRENDS_ANALYSIS = {
    'keywords': ['MASLD', 'NAFLD', 'Rezdiffra', 'Wegovy', 'Ozempic'],
    'timeframe': '2023-01-01 2025-10-28',
    'validation': {
        'required_non_zero_points': 10,  # MASLD had 13 non-zero points
        'value_range': (0, 100),
        'adf_test_alpha': 0.05,
    },
    'decomposition': {
        'model': 'additive',
        'period': 12
    }
}

# Add validation thresholds
VALIDATION_THRESHOLDS = {
    'min_non_zero_points': 10,
    'value_ranges': {
        'MASLD': (0, 1),
        'NAFLD': (0, 1),
        'Rezdiffra': (0, 1),
        'Wegovy': (15, 36),
        'Ozempic': (39, 100)
    }
}

# Reddit analysis parameters
REDDIT_SUBREDDIT_CONFIG = [
    # Core MASLD subreddits (direct collection)
    ("NAFLD", None),
    ("MASH", None),
    ("NASH", None),
    ("MASLD", None),
    ("obesity", None),
    # Medication subreddits - BOTH direct collection AND MASLD search
    ("Ozempic", ["NAFLD", "NASH", "MASLD", "fatty liver", "liver", "Semaglutide"]),
    ("Wegovy", ["NAFLD", "NASH", "MASLD", "fatty liver", "liver", "Semaglutide", "Novo Nordisk"]),
    ("semaglutide", ["NAFLD", "NASH", "MASLD", "fatty liver", "liver"]),
    # Resmetirom focused searches
    ("Supplements", ["NAFLD", "fatty liver", "NASH", "MASLD", "Resmetirom", "Rezdiffra", "Madrigal"]),
    ("AskDocs", ["NAFLD", "fatty liver", "NASH", "MASLD", "Resmetirom", "Rezdiffra", "new FDA liver drug"]),
    ("medical", ["NAFLD", "fatty liver", "NASH", "MASLD", "Resmetirom", "Rezdiffra", "new FDA liver drug"]),
    ("liver", ["NAFLD", "fatty liver", "NASH", "MASLD", "Resmetirom", "Rezdiffra", "new FDA liver drug", "Madrigal"]),
    # Medical professional forums
    ("medicine", ["NAFLD", "fatty liver", "NASH", "MASLD", "Resmetirom", "Rezdiffra", "Semaglutide", "Wegovy", "Ozempic"]),
    ("pharmacy", ["Resmetirom", "Rezdiffra", "Semaglutide", "Wegovy", "Ozempic", "new FDA liver drug"]),
]

# Study timeframe (consistent across all data sources)
STUDY_START_DATE = '2023-01-01'
STUDY_END_DATE = '2025-10-28'

# ==============================================================================
# 6. VISUALIZATION SETTINGS
# ==============================================================================
PLOT_STYLE = 'whitegrid'
COLOR_PALETTE = "husl"

# Media Cloud specific colors
MEDIA_CLOUD_COLORS = {
    'disease': '#440154',
    'resmetirom': '#21908d', 
    'glp1': '#fde725'
}

# ==============================================================================
# 7. HELPER FUNCTIONS
# ==============================================================================
def get_latest_data_file(pattern):
    """Helper function to get the latest file matching a pattern"""
    files = list(DATA_DIR.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda x: x.stat().st_ctime)

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
# 8. GOOGLE DRIVE/PRE-RETRIEVED DATA LINKS (For User Testing)
# ==============================================================================
GDRIVE_GOOGLE_TRENDS_URL = "https://drive.google.com/file/d/1lrov39Ww1Zp2kJTu4zb1yr3Q2j69H1rX/view?usp=drive_link"
GDRIVE_STOCK_DATA_URL = "https://drive.google.com/file/d/1hHWJ85BtGBP0aJBkhgr0wjjbPWvWqZYX/view?usp=drive_link"
GDRIVE_REDDIT_DATA_URL = "https://drive.google.com/file/d/1atMK_8axChUJMtzw8e7iv46tEehPTSsK/view?usp=drive_link"
GDRIVE_PUBMED_DATA_URL = "https://drive.google.com/file/d/1HyNZl8yxF3U9ooj9reF48MlRHheFTPEt/view?usp=drive_link"
GDRIVE_MEDIA_CLOUD_URL = "https://drive.google.com/file/d/1MfP0OizpCSUrqjZmMLwrCj3vS4KI0KA3/view?usp=drive_link"

# ==============================================================================
# 9. GOOGLE DRIVE DATA LOADING FUNCTIONS (Fallback when APIs fail)
# ==============================================================================

def load_google_trends_from_drive():
    """Load Google Trends data from Google Drive"""
    try:
        import gdown
        import pandas as pd
        import tempfile
        import os

        # Extract file ID from Google Drive URL
        file_id = GDRIVE_GOOGLE_TRENDS_URL.split('/d/')[1].split('/')[0]
        download_url = f"https://drive.google.com/uc?id={file_id}"

        # Download to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_path = tmp_file.name

        # Download outside the context manager
        gdown.download(download_url, tmp_path, quiet=False)

        # Read the file
        df = pd.read_csv(tmp_path)

        # Close and delete the file
        try:
            os.unlink(tmp_path)
        except:
            pass  # Ignore cleanup errors

        print("Google Trends data loaded from Google Drive")
        return df

    except Exception as e:
        print(f"Failed to load Google Trends from Google Drive: {e}")
        return None


def load_reddit_data_from_drive():
    """Load Reddit data from Google Drive"""
    try:
        import gdown
        import pandas as pd
        import tempfile
        import os

        file_id = GDRIVE_REDDIT_DATA_URL.split('/d/')[1].split('/')[0]
        download_url = f"https://drive.google.com/uc?id={file_id}"

        # Download to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_path = tmp_file.name

        # Download outside the context manager
        gdown.download(download_url, tmp_path, quiet=False)

        # Read the file
        df = pd.read_csv(tmp_path)

        # Close and delete the file
        try:
            os.unlink(tmp_path)
        except:
            pass  # Ignore cleanup errors

        print("Reddit data loaded from Google Drive")
        return df

    except Exception as e:
        print(f"Failed to load Reddit data from Google Drive: {e}")
        return None


def load_pubmed_data_from_drive():
    """Load PubMed data from Google Drive"""
    try:
        import gdown
        import pandas as pd
        import tempfile
        import os

        file_id = GDRIVE_PUBMED_DATA_URL.split('/d/')[1].split('/')[0]
        download_url = f"https://drive.google.com/uc?id={file_id}"

        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            gdown.download(download_url, tmp_file.name, quiet=False)
            df = pd.read_csv(tmp_file.name)
            os.unlink(tmp_file.name)

        print("PubMed data loaded from Google Drive")
        return df

    except Exception as e:
        print(f"Failed to load PubMed data from Google Drive: {e}")
        return None


def load_stock_data_from_drive():
    """Load stock data from Google Drive"""
    try:
        import gdown
        import pandas as pd
        import tempfile
        import os

        file_id = GDRIVE_STOCK_DATA_URL.split('/d/')[1].split('/')[0]
        download_url = f"https://drive.google.com/uc?id={file_id}"

        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            gdown.download(download_url, tmp_file.name, quiet=False)
            df = pd.read_csv(tmp_file.name)
            os.unlink(tmp_file.name)

        print("Stock data loaded from Google Drive")
        return df

    except Exception as e:
        print(f"Failed to load stock data from Google Drive: {e}")
        return None


def load_media_cloud_from_drive():
    """Load Media Cloud data from Google Drive"""
    try:
        import gdown
        import zipfile
        import tempfile
        import os
        from pathlib import Path

        file_id = GDRIVE_MEDIA_CLOUD_URL.split('/d/')[1].split('/')[0]
        download_url = f"https://drive.google.com/uc?id={file_id}"

        # Download and extract zip file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            gdown.download(download_url, tmp_file.name, quiet=False)

            # Extract to media_cloud directory
            with zipfile.ZipFile(tmp_file.name, 'r') as zip_ref:
                zip_ref.extractall(DATA_DIR / "media_cloud")

            os.unlink(tmp_file.name)

        print("Media Cloud data loaded from Google Drive")
        return True

    except Exception as e:
        print(f"Failed to load Media Cloud data from Google Drive: {e}")
        return False


# Optional: Auto-validate on import
if __name__ != "__main__":
    config_errors = validate_config()
    if config_errors:
        print("Configuration warnings:")
        for error in config_errors:
            print(f"  - {error}")
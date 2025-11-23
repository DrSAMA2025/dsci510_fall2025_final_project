from pathlib import Path
from dotenv import load_dotenv
import os
from datetime import datetime

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
REDDIT_USER_AGENT = os.environ.get("REDDIT_USER_AGENT", "dsci510_final_project_v1.0")

# ==============================================================================
# 3. PROJECT DATASETS & RESULTS CONFIGURATION
# ==============================================================================

# --- File Names for Data Directory ---
GOOGLE_TRENDS_FILE = "google_trends_initial_data.csv"
STOCK_DATA_FILE = "stock_prices.csv"
REDDIT_DATA_FILE_BASE = "reddit_data_"
PUBMED_DATA_FILE_BASE = "pubmed_masld_articles_"

# File patterns that match my actual file naming conventions
REDDIT_DATA_PATTERN = "reddit_data_*.csv"  # Matches: reddit_data_YYYYMMDD_HHMM.csv
PUBMED_DATA_PATTERN = "pubmed_masld_articles_*.csv"  # Matches: pubmed_masld_articles_YYYYMMDD_HHMM.csv

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
STUDY_START_DATE = datetime(2023, 1, 1)
STUDY_END_DATE = datetime(2025, 10, 28)

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
# 7. GOOGLE DRIVE/PRE-RETRIEVED DATA LINKS (For User Testing)
# ==============================================================================
GDRIVE_GOOGLE_TRENDS_URL = "https://drive.google.com/file/d/1lrov39Ww1Zp2kJTu4zb1yr3Q2j69H1rX/view?usp=drive_link"
GDRIVE_STOCK_DATA_URL = "https://drive.google.com/file/d/1hHWJ85BtGBP0aJBkhgr0wjjbPWvWqZYX/view?usp=drive_link"
GDRIVE_REDDIT_DATA_URL = "https://drive.google.com/file/d/1atMK_8axChUJMtzw8e7iv46tEehPTSsK/view?usp=drive_link"
GDRIVE_PUBMED_DATA_URL = "https://drive.google.com/file/d/1HyNZl8yxF3U9ooj9reF48MlRHheFTPEt/view?usp=drive_link"
GDRIVE_MEDIA_CLOUD_URL = "https://drive.google.com/file/d/1MfP0OizpCSUrqjZmMLwrCj3vS4KI0KA3/view?usp=drive_link"

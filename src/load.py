import pandas as pd
import praw
import yfinance as yf
from pytrends.request import TrendReq
from Bio import Entrez
import requests
from datetime import datetime
from pathlib import Path

# Import configuration constants
from config import (
    DATA_DIR,
    REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT, REDDIT_SUBREDDIT_CONFIG,
    STUDY_START_DATE, STUDY_END_DATE,
    GOOGLE_TRENDS_KEYWORDS, GOOGLE_TRENDS_TIMEFRAME, GOOGLE_TRENDS_FILE,
    STOCK_DATA_FILE,
    PUBMED_DATA_FILE_BASE,
    REDDIT_DATA_FILE_BASE
)


# --- Utility Functions ---

def get_latest_timestamp_filepath(base_name: str) -> Path:
    """Generates a timestamped filepath for saving data."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"{base_name}{STUDY_START_DATE.replace('-', '')}_{STUDY_END_DATE.replace('-', '')}_{timestamp}.csv"
    # Ensure all file names are correctly standardized for saving
    if base_name == PUBMED_DATA_FILE_BASE:
        filename = f"{PUBMED_DATA_FILE_BASE}{timestamp}.csv"
    elif base_name == REDDIT_DATA_FILE_BASE:
        filename = f"{REDDIT_DATA_FILE_BASE}{STUDY_START_DATE.replace('-', '')}_{STUDY_END_DATE.replace('-', '')}_{timestamp}.csv"
    else:
        # For non-timestamped files like stock/trends, return the configured name
        return DATA_DIR / base_name

    return DATA_DIR / filename


# --- Data Source Functions ---

def get_google_trends_data():
    """Fetches and saves Google Trends data for key terms."""
    print(f"\n[Loading] Fetching Google Trends data for: {GOOGLE_TRENDS_KEYWORDS}")
    try:
        pytrend = TrendReq(hl='en-US', tz=360)
        pytrend.build_payload(
            kw_list=GOOGLE_TRENDS_KEYWORDS,
            cat=0,  # All categories
            timeframe=GOOGLE_TRENDS_TIMEFRAME,
            geo='US',
            gprop=''  # Web search
        )
        df_trends = pytrend.interest_over_time()

        # Cleanup
        df_trends = df_trends.drop(columns=['isPartial'], errors='ignore')
        df_trends = df_trends.reset_index().rename(columns={'date': 'date'})

        save_path = DATA_DIR / GOOGLE_TRENDS_FILE
        df_trends.to_csv(save_path, index=False)
        print(f"[Success] Google Trends data saved to: {save_path.name}")
        return df_trends
    except Exception as e:
        print(f"[ERROR] Failed to fetch Google Trends data: {e}")
        return None


def get_reddit_data():
    """Fetches Reddit posts and comments using PRAW."""
    print("\n[Loading] Connecting to Reddit API and fetching data...")
    if not all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT]):
        print("[ERROR] Reddit credentials missing. Skipping Reddit data collection.")
        return None

    try:
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )

        all_data = []

        # Use the config list for subreddits and search terms
        for subreddit_name, search_terms in REDDIT_SUBREDDIT_CONFIG:
            subreddit = reddit.subreddit(subreddit_name)

            # 1. Collect direct posts from subreddit (no search)
            if not search_terms:
                print(f"  > Collecting hot posts from r/{subreddit_name}...")
                for submission in subreddit.hot(limit=100):  # Limit to 100 posts for a quick run
                    all_data.append(process_submission(submission, search_term=None))

            # 2. Collect posts matching search terms within the subreddit
            else:
                for term in search_terms:
                    print(f"  > Searching r/{subreddit_name} for term: '{term}'...")
                    for submission in subreddit.search(
                            query=term,
                            limit=50,  # Limit search results
                            time_filter='all'
                    ):
                        all_data.append(process_submission(submission, search_term=term))

        df_reddit = pd.DataFrame([d for d in all_data if d is not None])

        save_path = get_latest_timestamp_filepath(REDDIT_DATA_FILE_BASE)
        df_reddit.to_csv(save_path, index=False)
        print(f"[Success] Reddit data saved to: {save_path.name} ({len(df_reddit)} records)")
        return df_reddit

    except Exception as e:
        print(f"[ERROR] Failed to fetch Reddit data: {e}")
        return None


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


def get_pubmed_data(email="test@example.com"):
    """Fetches PubMed article metadata using Biopython's Entrez."""
    print("\n[Loading] Fetching PubMed article metadata...")
    Entrez.email = email

    # Search terms focusing on MASLD/MASH and the key drug
    search_query = "(MASLD OR MASH OR NAFLD) AND (Rezdiffra OR Resmetirom OR semaglutide OR GLP-1)"

    try:
        handle = Entrez.esearch(db="pubmed", term=search_query, retmax="1000")  # Limit to 1000
        record = Entrez.read(handle)
        id_list = record["IdList"]
        handle.close()

        print(f"  > Found {len(id_list)} articles. Fetching details...")

        article_data = []
        if id_list:
            fetch_handle = Entrez.efetch(db="pubmed", id=id_list, rettype="xml", retmode="xml")
            articles = Entrez.read(fetch_handle)
            fetch_handle.close()

            for pubmed_article in articles['PubmedArticle']:
                medline_citation = pubmed_article['MedlineCitation']
                article = medline_citation['Article']

                # Extracting details
                title = article.get('ArticleTitle', 'No Title')
                abstract = article.get('Abstract', {}).get('AbstractText', ['No Abstract'])
                abstract = ' '.join(abstract) if isinstance(abstract, list) else abstract

                # Date extraction logic can be complex; simplified here
                pub_date = article.get('Journal', {}).get('JournalIssue', {}).get('PubDate', {})
                year = pub_date.get('Year', 'N/A')
                month = pub_date.get('Month', 'N/A')

                article_data.append({
                    'pubmed_id': medline_citation['PMID'],
                    'title': title,
                    'abstract': abstract,
                    'publication_year': year,
                    'publication_month': month,
                    'journal': article.get('Journal', {}).get('Title', 'N/A'),
                })

        df_pubmed = pd.DataFrame(article_data)

        save_path = get_latest_timestamp_filepath(PUBMED_DATA_FILE_BASE)
        df_pubmed.to_csv(save_path, index=False)
        print(f"[Success] PubMed data saved to: {save_path.name} ({len(df_pubmed)} records)")
        return df_pubmed

    except Exception as e:
        print(f"[ERROR] Failed to fetch PubMed data: {e}")
        return None

def get_stock_data():
    """Fetches and saves historical stock data for NVO and MDGL."""
    TICKERS = ['NVO', 'MDGL']
    print(f"\n[Loading] Fetching stock data for: {TICKERS}")
    try:
        df_stocks = yf.download(
            TICKERS,
            start=STUDY_START_DATE,
            end=STUDY_END_DATE,
            interval="1d",
            group_by='ticker'
        )

        # Flatten column names for easier saving/loading (MultiIndex fix)
        df_stocks.columns = pd.MultiIndex.from_tuples([(col[0], col[1]) for col in df_stocks.columns])

        save_path = DATA_DIR / STOCK_DATA_FILE
        df_stocks.to_csv(save_path)
        print(f"[Success] Stock data saved to: {save_path.name}")
        return df_stocks
    except Exception as e:
        print(f"[ERROR] Failed to fetch stock data: {e}")
        return None


def get_media_cloud_data():
    """
    Ensure Media Cloud data is available - download if missing from Google Drive.
    """
    print("\n[Media Cloud] Checking Media Cloud data availability...")

    if download_media_cloud_data():
        print("[Media Cloud] Data ready for analysis")
        return True
    else:
        print("[Media Cloud] Data not available. Analysis will be skipped.")
        return False


def download_from_gdrive(file_url, save_path):
    """Download file from Google Drive if local file doesn't exist"""
    if save_path.exists():
        print(f"  Using existing local file: {save_path.name}")
        return save_path

    try:
        import gdown
        print(f"  Downloading from Google Drive: {save_path.name}")

        # gdown works with Google Drive share links
        gdown.download(file_url, str(save_path), fuzzy=True)

        if save_path.exists():
            print(f"  Successfully downloaded: {save_path.name}")
            return save_path
        else:
            print(f"  Download failed for: {save_path.name}")
            return None

    except ImportError:
        print("  gdown not installed. Please run: pip install gdown")
        return None
    except Exception as e:
        print(f"  Download error: {e}")
        return None


def download_media_cloud_data():
    """Download and extract Media Cloud data from Google Drive"""
    from config import DATA_DIR, GDRIVE_MEDIA_CLOUD_URL, MEDIA_CLOUD_DATASETS

    media_cloud_dir = DATA_DIR / "media_cloud"
    zip_path = DATA_DIR / "media_cloud_data.zip"

    # Check if data already exists
    if media_cloud_dir.exists() and any(media_cloud_dir.iterdir()):
        print(f"[Media Cloud] Using existing data in: {media_cloud_dir.name}")
        return True

    # Download from Google Drive
    if not GDRIVE_MEDIA_CLOUD_URL or GDRIVE_MEDIA_CLOUD_URL == "your_media_cloud_zip_link_here":
        print("[Media Cloud] No Google Drive URL configured. Skipping download.")
        return False

    try:
        import gdown
        import zipfile

        print(f"[Media Cloud] Downloading data from Google Drive...")
        gdown.download(GDRIVE_MEDIA_CLOUD_URL, str(zip_path), fuzzy=True)

        if zip_path.exists():
            print(f"[Media Cloud] Extracting data...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(DATA_DIR)

            # Verify extraction
            if media_cloud_dir.exists():
                print(f"[Media Cloud] Successfully downloaded and extracted")
                # Clean up zip file
                zip_path.unlink()
                return True
            else:
                print(f"[Media Cloud] Extraction failed - folder not created")
                return False
        else:
            print(f"[Media Cloud] Download failed")
            return False

    except ImportError:
        print("[Media Cloud] gdown not installed. Please run: pip install gdown")
        return False
    except Exception as e:
        print(f"[Media Cloud] Error: {e}")
        return False


def ensure_data_available():
    """Ensure all required data files are available (download if missing)"""
    from config import (
        DATA_DIR, GDRIVE_GOOGLE_TRENDS_URL, GDRIVE_STOCK_DATA_URL,
        GDRIVE_REDDIT_DATA_URL, GDRIVE_PUBMED_DATA_URL, GDRIVE_MEDIA_CLOUD_URL,
        GOOGLE_TRENDS_FILE, STOCK_DATA_FILE
    )

    print("\n" + "=" * 60)
    print("DATA AVAILABILITY CHECK & DOWNLOAD")
    print("=" * 60)

    # Download individual CSV files
    download_from_gdrive(GDRIVE_GOOGLE_TRENDS_URL, DATA_DIR / GOOGLE_TRENDS_FILE)
    download_from_gdrive(GDRIVE_STOCK_DATA_URL, DATA_DIR / STOCK_DATA_FILE)
    download_from_gdrive(GDRIVE_REDDIT_DATA_URL, DATA_DIR / "reddit_data_latest.csv")
    download_from_gdrive(GDRIVE_PUBMED_DATA_URL, DATA_DIR / "pubmed_data_latest.csv")

    # Download and extract Media Cloud data
    download_media_cloud_data()

    print("\n✓ Data availability check complete")


# Combine all functions into a single entry point
def run_all_data_loaders():
    """Runs all data acquisition functions."""
    print("=" * 60)
    print("STARTING DATA ACQUISITION (load.py)")
    print("=" * 60)

    get_google_trends_data()
    get_stock_data()
    # Note: Reddit and PubMed often require more time/auth, hence their output is important
    get_reddit_data()
    get_pubmed_data()
    get_media_cloud_data()

    print("\nDATA ACQUISITION COMPLETE.")


if __name__ == "__main__":
    run_all_data_loaders()
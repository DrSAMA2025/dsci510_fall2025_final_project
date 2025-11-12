import pandas as pd
from pathlib import Path
import os
import glob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon if not present (required by VADER)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    print("[nltk] Downloading vader_lexicon...")
    nltk.download('vader_lexicon', quiet=True)

# Import configuration constants
from config import (
    DATA_DIR, RESULTS_DIR,
    GOOGLE_TRENDS_FILE, STOCK_DATA_FILE,
    REDDIT_DATA_FILE_BASE, PUBMED_DATA_FILE_BASE,
    STUDY_START_DATE, STUDY_END_DATE
)


# --- Utility Functions ---

def get_latest_data_filepath(base_name: str) -> Path or None:
    """Finds the most recently created file matching the base_name pattern."""
    search_pattern = str(DATA_DIR / f"{base_name}*.csv")
    list_of_files = glob.glob(search_pattern)

    if not list_of_files:
        print(f"[Error] No files found matching pattern: {base_name}*.csv")
        return None

    latest_file = max(list_of_files, key=os.path.getctime)
    return Path(latest_file)


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


# --- Processing Functions ---

def process_google_trends(df_or_path):
    """Cleans and prepares Google Trends data."""
    print("[Processing] Cleaning Google Trends data...")

    # Handle both DataFrame and file path inputs
    if isinstance(df_or_path, Path):
        # It's a file path - load the data
        df = pd.read_csv(df_or_path)
    else:
        # It's already a DataFrame
        df = df_or_path

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()

    # Normalize data if needed, but for initial analysis, raw scores are fine.
    # We will simply ensure all trend columns are integer type.
    trend_cols = [col for col in df.columns if col not in ['isPartial']]
    df[trend_cols] = df[trend_cols].fillna(0).astype(int)
    print(f"  > Trends data ready. Shape: {df.shape}")
    return df


def process_stock_data(df):
    """Cleans and prepares Stock data, renaming columns for clarity."""
    print("[Processing] Cleaning Stock data...")

    # The data still has MultiIndex columns, so let's flatten them first
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]

    # Now the columns are flattened like: 'MDGL_Close', 'NVO_Close', etc.
    print(f"Flattened columns: {df.columns.tolist()}")

    # Reset index to get Date as a column
    df = df.reset_index()
    df = df.rename(columns={'index': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()

    # Select the close price columns (they're now 'MDGL_Close' and 'NVO_Close')
    df_adj_close = pd.DataFrame({
        'NVO_Close': df['NVO_Close'],
        'MDGL_Close': df['MDGL_Close']
    })

    df_adj_close = df_adj_close.dropna()
    print(f"  > Stock data ready. Shape: {df_adj_close.shape}")
    return df_adj_close


def process_reddit_data(df: pd.DataFrame) -> pd.DataFrame:
    """Performs VADER sentiment analysis on Reddit data."""
    print("[Processing] Performing Sentiment Analysis on Reddit data...")
    analyzer = SentimentIntensityAnalyzer()

    # Concatenate post text and comment text for analysis
    df['text_to_analyze'] = df['post_text'].fillna('') + ' ' + df['comment_text'].fillna('')
    df['text_to_analyze'] = df['text_to_analyze'].str.strip()

    # Filter out rows where text is empty (e.g., deleted comments, pure link posts)
    df = df[df['text_to_analyze'].str.len() > 0].copy()

    # Apply sentiment analysis
    df['sentiment_score'] = df['text_to_analyze'].apply(
        lambda x: analyzer.polarity_scores(x)['compound']
    )

    # Add sentiment category
    df['sentiment_category'] = pd.cut(
        df['sentiment_score'],
        bins=[-1, -0.05, 0.05, 1],
        labels=['Negative', 'Neutral', 'Positive'],
        right=True
    )

    # Final cleanup and date formatting
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])

    print(f"  > Reddit data with sentiment ready. Shape: {df.shape}")
    return df[['subreddit', 'search_term', 'timestamp', 'type', 'sentiment_score', 'sentiment_category']]


def process_pubmed_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans and filters PubMed data."""
    print("[Processing] Cleaning PubMed data...")

    # Convert month names to numbers
    month_map = {
        'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04', 'may': '05', 'jun': '06',
        'jul': '07', 'aug': '08', 'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
    }

    # Clean year and month data
    df = df[df['publication_year'] != 'N/A'].copy()  # Remove rows with N/A years
    df['month_num'] = df['publication_month'].str.lower().str[:3].map(month_map)

    # Create publication date
    df['publication_date'] = pd.to_datetime(
        df['publication_year'].astype(str) + '-' + df['month_num'] + '-01',
        format='%Y-%m-%d',
        errors='coerce'
    )

    # Filter to study period
    df = df[(df['publication_date'] >= STUDY_START_DATE) &
            (df['publication_date'] <= STUDY_END_DATE)].copy()

    df = df.dropna(subset=['publication_date'])
    df = df.sort_values(by='publication_date')

    print(f"  > PubMed data ready. Shape: {df.shape}")
    return df[['pubmed_id', 'title', 'abstract', 'publication_date', 'journal']]


def run_all_data_processors():
    """Loads raw data and runs all processing steps."""
    print("=" * 60)
    print("STARTING DATA PROCESSING (process.py)")
    print("=" * 60)

    processed_data = {}

    # 1. Google Trends
    df_trends_raw = load_data(GOOGLE_TRENDS_FILE)
    if df_trends_raw is not None:
        processed_data['trends'] = process_google_trends(df_trends_raw)

    # 2. Stock Data
    # Stock data has a multi-level header, requiring special loading
    df_stocks_raw = load_data(STOCK_DATA_FILE, header=[0, 1], index_col=0)
    if df_stocks_raw is not None:
        processed_data['stocks'] = process_stock_data(df_stocks_raw)

    # 3. Reddit Data (Timestamped file)
    df_reddit_raw = load_data(REDDIT_DATA_FILE_BASE, is_timestamped=True)
    if df_reddit_raw is not None:
        processed_data['reddit'] = process_reddit_data(df_reddit_raw)

    # 4. PubMed Data (Timestamped file)
    df_pubmed_raw = load_data(PUBMED_DATA_FILE_BASE, is_timestamped=True)
    if df_pubmed_raw is not None:
        processed_data['pubmed'] = process_pubmed_data(df_pubmed_raw)

    # 5. Media Cloud Data (This is complex and usually handled in analyze.py)
    print("\n[Processing] Skipping Media Cloud file processing (Handled in analyze.py).")

    print("\nDATA PROCESSING COMPLETE.")
    return processed_data


if __name__ == "__main__":
    processed_data_frames = run_all_data_processors()
    # Example: Save processed data (optional, but good for debugging)
    # for name, df in processed_data_frames.items():
    #     df.to_csv(DATA_DIR / f'processed_{name}.csv')
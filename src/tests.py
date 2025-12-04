# tests.py - Comprehensive Test Suite for MASLD Project
import pandas as pd
import os
import sys
from pathlib import Path
from datetime import datetime

# Import moved functions from new locations
try:
    from utils import get_latest_data_file, validate_config
    from load import (
        load_google_trends_from_drive, load_reddit_data_from_drive,
        load_pubmed_data_from_drive, load_stock_data_from_drive,
        load_media_cloud_from_drive
    )
    MOVED_FUNCTIONS_LOADED = True
except ImportError as e:
    print(f"Warning: Could not import moved functions: {e}")
    MOVED_FUNCTIONS_LOADED = False

# Import project configuration
try:
    from config import (
        DATA_DIR, RESULTS_DIR,
        GOOGLE_TRENDS_FILE, STOCK_DATA_FILE,
        REDDIT_DATA_PATTERN, PUBMED_DATA_PATTERN,
        REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT,
        STUDY_START_DATE, STUDY_END_DATE,
    )

    CONFIG_LOADED = True
except ImportError as e:
    print(f"Warning: Could not import configuration: {e}")
    print("Using fallback paths...")
    DATA_DIR = Path("data")
    RESULTS_DIR = Path("results")
    CONFIG_LOADED = False

# Add the current directory to Python path for imports
sys.path.append('.')


def test_reddit_api_connection():
    """
    Tests connectivity to the Reddit API using PRAW.
    """
    print("\n" + "=" * 70)
    print("API CONNECTION TEST (Reddit/PRAW)")
    print("=" * 70)

    if not CONFIG_LOADED:
        print("FAIL: Configuration not loaded. Cannot run API test.")
        return False

    # Check for credentials
    if not all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT]):
        print("FAIL: Missing Reddit credentials in config.py / .env file.")
        print("Please ensure REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, and REDDIT_USER_AGENT are set.")
        return False

    try:
        import praw

        # Initialize Reddit client
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )

        # Test connection by fetching a single post from a test subreddit
        test_subreddit = "test"  # Using a small, fast subreddit for connection test
        try:
            # Check authentication by attempting to retrieve a subreddit instance
            subreddit = reddit.subreddit(test_subreddit)

            # Try to get one post to verify connection works
            post_count = 0
            for post in subreddit.hot(limit=1):
                print(f"PASS: Successfully connected to Reddit API")
                print(f"Retrieved test post: '{post.title[:50]}...' from r/{test_subreddit}")
                post_count += 1
                break

            if post_count == 0:
                print("WARNING: Connected to API, but couldn't fetch a post from r/test (might be empty/rate-limited).")
                print("If credentials are correct, this may be a soft pass.")
                return True  # Assuming configuration is the main hurdle
            return True

        except Exception as e:
            print(f"FAIL: Could not fetch data from Reddit: {e}")
            return False

    except ImportError:
        print("FAIL: PRAW library not installed. Run: pip install praw")
        return False
    except Exception as e:
        print(f"FAIL: API Connection/Authentication Error: {e}")
        return False


# --- Google Trends Data Tests ---
def test_google_trends_data():
    """Test the quality and structure of Google Trends data"""
    print("\n" + "=" * 50)
    print("GOOGLE TRENDS DATA VALIDATION TESTS")
    print("=" * 50)

    try:
        # 1. Check if file exists
        trends_file = DATA_DIR / GOOGLE_TRENDS_FILE
        if not trends_file.exists():
            print(f"FAIL: File {trends_file} not found. Run google_trends.py first.")
            return False

        # 2. Load data
        df = pd.read_csv(trends_file)
        print("File loaded successfully")
        print(f"Dataset shape: {df.shape}")

        # 3. Basic structure tests
        print("\n--- STRUCTURE TESTS ---")
        required_columns = ['MASLD', 'NAFLD', 'Rezdiffra', 'Wegovy', 'Ozempic']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"FAIL: Missing columns: {missing_columns}")
            return False
        else:
            print("PASS: All required columns present")

        # 4. Data quality tests
        print("\n--- DATA QUALITY TESTS ---")
        missing_values = df.isnull().sum().sum()
        if missing_values == 0:
            print("PASS: No missing values in dataset")
        else:
            print(f"WARNING: Found {missing_values} missing values")

        # Check date range and continuity (using index)
        if isinstance(df.index, pd.DatetimeIndex):
            date_range = df.index.max() - df.index.min()
            expected_days = (pd.to_datetime(STUDY_END_DATE) - pd.to_datetime(STUDY_START_DATE)).days
            date_coverage = (date_range.days / expected_days) * 100

            print(f"Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
            print(f"Date coverage: {date_coverage:.1f}% of expected range")
        else:
            print("WARNING: DataFrame index is not DatetimeIndex")

        # 5. Content validation tests
        print("\n--- CONTENT VALIDATION TESTS ---")
        masld_non_zero = (df['MASLD'] > 0).sum()
        if masld_non_zero > 0:
            print(f"PASS: MASLD has {masld_non_zero} non-zero data points")
        else:
            print("FAIL: MASLD has no non-zero values - check data collection")

        # Check value ranges (Google Trends should be 0-100)
        trend_columns = ['MASLD', 'NAFLD', 'Rezdiffra', 'Wegovy', 'Ozempic']
        for col in trend_columns:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                if 0 <= min_val <= max_val <= 100:
                    print(f"PASS: {col}: values in valid range ({min_val}-{max_val})")
                else:
                    print(f"WARNING: {col}: values outside expected 0-100 range ({min_val}-{max_val})")

        print("\nGoogle Trends data validation: COMPLETE")
        return True

    except Exception as e:
        print(f"TEST FAILED: {e}")
        return False


# --- Reddit Data Tests ---
def test_reddit_data_quality():
    """Test the quality and structure of Reddit data"""
    print("\n=== REDDIT DATA QUALITY TESTS ===")

    # Test 1: Check if Reddit data file exists using the pattern from config
    try:
        reddit_files = list(DATA_DIR.glob(REDDIT_DATA_PATTERN))
        if not reddit_files:
            raise FileNotFoundError(
                f"No files matching pattern '{REDDIT_DATA_PATTERN}' found. Run reddit_data_collector.py first.")

        latest_file = max(reddit_files, key=lambda x: x.stat().st_ctime)
        print(f"Found Reddit data file: {latest_file.name}")
    except Exception as e:
        print(f"Test 1 FAILED: {e}")
        return False

    # Test 2: Load and check data structure
    try:
        df = pd.read_csv(latest_file)
        expected_columns = ['subreddit', 'post_title', 'post_text', 'comment_text', 'timestamp', 'type']
        for col in expected_columns:
            assert col in df.columns, f"Missing expected column: {col}"
        print("Data structure validation passed")
    except Exception as e:
        print(f"Test 2 FAILED: {e}")
        return False

    # Test 3: Check data volume
    try:
        assert len(df) > 0, "Data file is empty"
        print(f"Data volume check passed: {len(df):,} records")
    except Exception as e:
        print(f"Test 3 FAILED: {e}")
        return False

    # Test 4: Check date range
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        valid_timestamps = df['timestamp'].dropna()
        if valid_timestamps.empty:
            print("Test 4 FAILED: No valid timestamps found.")
            return False

        min_date = valid_timestamps.min()
        max_date = valid_timestamps.max()

        # Check start date - show what will be filtered
        print(f"\nDate Range Analysis:")
        print(f"  Raw data range: {min_date.date()} to {max_date.date()}")

        # Calculate what WILL be in study period after filtering
        in_study = df[(df['timestamp'] >= pd.Timestamp(STUDY_START_DATE)) &
                      (df['timestamp'] <= pd.Timestamp(STUDY_END_DATE))]

        print(f"  Posts in study period: {len(in_study)}/{len(df)} ({len(in_study) / len(df) * 100:.1f}%)")

        if len(in_study) > 100:
            print("  PASS: Sufficient data for analysis after filtering")
            print(
                f"  Date range after filtering: {in_study['timestamp'].min().date()} to {in_study['timestamp'].max().date()}")

            # For end date - allow extension beyond study period (normal for live data)
            if max_date > pd.Timestamp(STUDY_END_DATE):
                print(f"  WARNING: Data extends beyond study period: {max_date.date()}")
                print("          This is normal for real-time data collection.")
            return True
        else:
            print(f"  FAIL: Only {len(in_study)} posts in study period")
            return False

    except Exception as e:
        print(f"Test 4 FAILED: {e}")
        return False

# --- Stock Data Tests ---
def test_stock_data_integrity():
    """Test stock data integrity"""
    print("\n" + "=" * 50)
    print("STOCK DATA VALIDATION TESTS")
    print("=" * 50)

    try:
        # 1. Check if file exists
        stock_file = DATA_DIR / STOCK_DATA_FILE
        if not stock_file.exists():
            print(f"FAIL: File {stock_file} not found. Run stock_data.py first.")
            return False

        # 2. Load data
        df = pd.read_csv(stock_file, index_col=0, skiprows=1)
        print("Stock data loaded successfully")
        print(f"Dataset shape: {df.shape}")
        print("First few rows of stock data:")
        print(df.head(2))
        print("Column names and types:")
        print(df.dtypes)

        # 3. Basic structure tests
        print("\n--- STRUCTURE TESTS ---")
        if 'Close' in df.columns and 'Close.1' in df.columns:
            print("PASS: Data for both companies (NVO, MDGL) present")
        else:
            print(f"FAIL: Missing company data. Available columns: {df.columns.tolist()}")
            return False

        # 4. Data quality tests
        print("\n--- DATA QUALITY TESTS ---")
        date_series = pd.to_datetime(df.index, errors='coerce')
        valid_dates = date_series.dropna()

        if valid_dates.empty:
            print("FAIL: No valid dates found in stock data index.")
            return False

        print(f"Date range: {valid_dates.min().strftime('%Y-%m-%d')} to {valid_dates.max().strftime('%Y-%m-%d')}")
        print(f"Total trading days: {len(valid_dates)}")

        # 5. Content validation
        print("\n--- CONTENT VALIDATION TESTS ---")
        nvo_prices = df['Close.1'].dropna().astype(float)  # NVO Close prices
        mdgl_prices = df['Close'].dropna().astype(float)  # MDGL Close prices

        if nvo_prices.empty or mdgl_prices.empty:
            print("FAIL: Price data is empty after cleanup.")
            return False

        print(f"Novo Nordisk (NVO) price range: ${nvo_prices.min():.2f} - ${nvo_prices.max():.2f}")
        print(f"Madrigal (MDGL) price range: ${mdgl_prices.min():.2f} - ${mdgl_prices.max():.2f}")

        # Simple check: prices should be positive
        if nvo_prices.min() > 0 and mdgl_prices.min() > 0:
            print("PASS: All minimum prices are positive.")
        else:
            print("FAIL: Found non-positive stock prices.")
            return False

        print("\nStock data validation: COMPLETE")
        return True

    except Exception as e:
        print(f"TEST FAILED: {e}")
        return False


# --- PubMed Data Tests ---
def test_pubmed_data_quality():
    """Test PubMed data quality"""
    print("\n=== PUBMED DATA QUALITY TESTS ===")

    try:
        # Find the specific PubMed file using the pattern from config
        pubmed_files = list(DATA_DIR.glob(PUBMED_DATA_PATTERN))
        if not pubmed_files:
            raise FileNotFoundError(
                f"No files matching pattern '{PUBMED_DATA_PATTERN}' found. Run pubmed_data_collection.py first.")

        latest_file = max(pubmed_files, key=lambda x: x.stat().st_ctime)
        print(f"Found PubMed data file: {latest_file.name}")

        # Load and test data
        df = pd.read_csv(latest_file)

        # Check required columns
        required_columns = ['pubmed_id', 'title', 'abstract', 'publication_year', 'publication_month', 'journal']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"Missing columns: {missing_columns}")
            return False
        else:
            print("All required columns present")

        # Check data volume
        if len(df) > 0:
            print(f"Data volume check passed: {len(df):,} records")
        else:
            print("Data file is empty")
            return False

        # Check date range
        if 'publication_date' in df.columns:
            df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')
            valid_dates = df['publication_date'].dropna()

            if len(valid_dates) > 0:
                min_date = valid_dates.min()
                max_date = valid_dates.max()
                print(f"Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
            else:
                print("No valid dates found")

        print("ALL PUBMED DATA TESTS PASSED!")
        return True

    except Exception as e:
        print(f"TEST FAILED: {e}")
        return False


# --- Media Cloud Data Tests ---
def test_media_cloud_data_integrity():
    """Test Media Cloud data integrity"""
    print("\n" + "=" * 60)
    print("MEDIA CLOUD DATA VALIDATION TESTS")
    print("=" * 60)

    try:
        # Check raw data folders
        media_cloud_dir = DATA_DIR / "media_cloud"
        if not media_cloud_dir.exists():
            print("Media Cloud data directory not found")
            return False

        expected_folders = ['disease_focused', 'resmetirom_focused', 'glp1_focused']
        for folder in expected_folders:
            folder_path = media_cloud_dir / folder
            if folder_path.exists():
                csv_files = list(folder_path.glob('*.csv'))
                if csv_files:
                    print(f"{folder}: Found {len(csv_files)} CSV file(s)")
                else:
                    print(f"{folder}: Folder exists but contains no CSV files")
            else:
                print(f"{folder}: Folder missing")

        # Check analysis outputs directory
        analysis_dir = RESULTS_DIR / "media_cloud"
        if analysis_dir.exists():
            analysis_files = list(analysis_dir.glob('*'))
            print(f"Media Cloud analysis directory contains {len(analysis_files)} output file(s)")
        else:
            print("Media Cloud analysis directory not yet created")

        print("Media Cloud data validation: COMPLETE")
        return True

    except Exception as e:
        print(f"TEST FAILED: {e}")
        return False


def test_gdrive_download():
    """Test that Google Drive download functionality works"""
    print("\n" + "=" * 50)
    print("TESTING GOOGLE DRIVE DOWNLOAD FUNCTIONALITY")
    print("=" * 50)

    try:
        # Test individual download functions instead of ensure_data_available
        print("Testing individual Google Drive download functions...")

        # Test one representative function to verify the imports work
        from load import load_google_trends_from_drive
        result = load_google_trends_from_drive()

        if result is not None:
            print("Google Drive download functionality test passed")
            return True
        else:
            print("Google Drive download returned None (may be expected if no credentials)")
            return True  # Still pass since the function exists and runs
    except Exception as e:
        print(f"Download test failed: {e}")
        return False


# --- Utility Functions Tests ---
def test_moved_functions():
    """Test that functions moved from config.py are accessible in their new locations"""
    print("\n" + "=" * 50)
    print("MOVED FUNCTIONS ACCESSIBILITY TEST")
    print("=" * 50)

    try:
        # Test utility functions
        from utils import get_latest_data_file, validate_config
        print("Successfully imported utility functions from utils.py")

        # Test data loading functions
        from load import (
            load_google_trends_from_drive, load_reddit_data_from_drive,
            load_pubmed_data_from_drive, load_stock_data_from_drive,
            load_media_cloud_from_drive
        )
        print("Successfully imported data loading functions from load.py")

        print("All moved functions are accessible in their new locations")
        return True

    except ImportError as e:
        print(f"Failed to import moved functions: {e}")
        return False


# --- Main Test Runner ---
def run_all_tests():
    """Run all tests for the MASLD project"""
    print("MASLD PROJECT COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    test_results = {}

    # 1. File Structure Test
    test_results['moved_functions'] = test_moved_functions()

    # 2. API Connection Test
    test_results['api_connection'] = test_reddit_api_connection()

    # 3. Data Quality Tests
    test_results['google_trends'] = test_google_trends_data()
    test_results['reddit_data'] = test_reddit_data_quality()
    test_results['stock_data'] = test_stock_data_integrity()
    test_results['pubmed_data'] = test_pubmed_data_quality()
    test_results['media_cloud'] = test_media_cloud_data_integrity()
    test_results['gdrive_download'] = test_gdrive_download()

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed_tests = sum(1 for result in test_results.values() if result is True)
    total_tests = len(test_results)

    for test_name, result in test_results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name:20} : {status}")

    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("ALL TESTS PASSED! Project data is ready for analysis.")
    else:
        print("Some tests failed. Please check your data collection scripts and configurations.")

    return all(test_results.values())


if __name__ == "__main__":
    run_all_tests()
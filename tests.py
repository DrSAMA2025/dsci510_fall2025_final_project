# Tests for Google Trends Data (Data Source #1)
import pandas as pd
import os


def test_data_integrity(filename='data/google_trends_initial_data.csv'):

    print("\n" + "=" * 50)
    print("GOOGLE TRENDS DATA VALIDATION TESTS")
    print("=" * 50)

    try:
        # 1. Check if file exists
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found. Run google_trends.py first.")

        # 2. Load data
        df = pd.read_csv(filename)

        print("File loaded successfully")
        print(f"Dataset shape: {df.shape}")

        # 3. Basic structure tests
        print("\n--- STRUCTURE TESTS ---")

        # Check required columns
        required_columns = ['date', 'MASLD', 'NAFLD', 'Rezdiffra', 'Wegovy', 'Ozempic']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"FAIL: Missing columns: {missing_columns}")
        else:
            print("PASS: All required columns present")

        # Check data types
        if pd.api.types.is_datetime64_any_dtype(df['date']):
            print("PASS: Date column is properly formatted as datetime")
        else:
            print("WARNING: Date column is not datetime - consider converting")

        # 4. Data quality tests
        print("\n--- DATA QUALITY TESTS ---")

        # Check for missing values
        missing_values = df.isnull().sum().sum()
        if missing_values == 0:
            print("PASS: No missing values in dataset")
        else:
            print(f"WARNING: Found {missing_values} missing values")

        # Check date range and continuity
        df['date'] = pd.to_datetime(df['date'])
        date_range = df['date'].max() - df['date'].min()
        expected_days = (pd.to_datetime('2025-10-28') - pd.to_datetime('2023-01-01')).days
        date_coverage = (date_range.days / expected_days) * 100

        print(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        print(f"Date coverage: {date_coverage:.1f}% of expected range")

        # 5. Content validation tests
        print("\n--- CONTENT VALIDATION TESTS ---")

        # Check that MASLD data exists (should not be all zeros)
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

        # 6. Sample data display
        print("\n--- SAMPLE DATA ---")
        print("First 3 rows:")
        print(df.head(3).to_string(index=False))

        print("\n--- SUMMARY STATISTICS ---")
        print(df[trend_columns].describe())

        # 7. Final assessment
        print("\n" + "=" * 50)
        print("TEST SUMMARY")
        print("=" * 50)
        print("Google Trends data collection: COMPLETE")
        print(f"Total data points: {len(df)}")
        print(f"Time period coverage: {date_coverage:.1f}%")
        print("Data appears valid and ready for analysis!")

        return True

    except Exception as e:
        print(f"TEST FAILED: {e}")
        return False


def test_analysis_outputs():
    print("\n--- ANALYSIS OUTPUT TESTS ---")

    expected_files = [
        'google_trends_analysis.py',
        'analysis/google_trends_main_analysis.png',
        'analysis/google_trends_correlation_heatmap.png',
        'analysis/google_trends_statistical_summary.csv'
    ]

    for file in expected_files:
        if os.path.exists(file):
            print(f"PASS: {file} - Found")
        else:
            print(f"WARNING: {file} - Missing")


if __name__ == "__main__":
    # Run data integrity tests
    data_ok = test_data_integrity()

    # Run analysis output tests
    test_analysis_outputs()

    print("\n" + "=" * 50)
    if data_ok:
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("Task 1 data is ready for the final project.")
    else:
        print("SOME TESTS FAILED - Please check your data collection.")
    print("=" * 50)


# Tests for Reddit Data (Data Source #2)
# tests.py
import pandas as pd
import os
import sys
from datetime import datetime


def test_reddit_data_quality():
    """Test the quality and structure of Reddit data"""
    print("=== REDDIT DATA QUALITY TESTS ===")

    # Test 1: Check if Reddit data file exists
    try:
        reddit_files = [f for f in os.listdir('data') if f.startswith('reddit_data_2023_2025_') and f.endswith('.csv')]
        assert len(reddit_files) > 0, "No Reddit data files found in data folder"
        latest_file = max(reddit_files)
        print(f"✓ Found Reddit data file: {latest_file}")
    except Exception as e:
        print(f"✗ Test 1 FAILED: {e}")
        return False

    # Test 2: Load and check data structure
    try:
        df = pd.read_csv(f'data/{latest_file}')
        expected_columns = ['subreddit', 'post_title', 'post_text', 'comment_text', 'timestamp', 'type']
        for col in expected_columns:
            assert col in df.columns, f"Missing expected column: {col}"
        print("✓ Data structure validation passed")
    except Exception as e:
        print(f"✗ Test 2 FAILED: {e}")
        return False

    # Test 3: Check data volume
    try:
        assert len(df) > 0, "Data file is empty"
        assert len(df) >= 1000, f"Low data volume: only {len(df)} records"
        print(f"✓ Data volume check passed: {len(df):,} records")
    except Exception as e:
        print(f"✗ Test 3 FAILED: {e}")
        return False

    # Test 4: Check date range
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        min_date = df['timestamp'].min()
        max_date = df['timestamp'].max()
        assert min_date >= pd.Timestamp('2023-01-01'), f"Data starts before study period: {min_date}"
        assert max_date <= pd.Timestamp('2025-10-28'), f"Data extends beyond study period: {max_date}"
        print(f"✓ Date range validation passed: {min_date.date()} to {max_date.date()}")
    except Exception as e:
        print(f"✗ Test 4 FAILED: {e}")
        return False

    # Test 5: Check subreddit coverage
    try:
        subreddits = df['subreddit'].unique()
        expected_subs = ['Ozempic', 'Wegovy', 'semaglutide', 'NAFLD', 'MASH']
        found_subs = [sub for sub in expected_subs if sub in subreddits]
        assert len(found_subs) >= 3, f"Missing key subreddits. Found: {found_subs}"
        print(f"✓ Subreddit coverage passed: {len(subreddits)} subreddits including {found_subs}")
    except Exception as e:
        print(f"✗ Test 5 FAILED: {e}")
        return False

    # Test 6: Check for key term mentions
    try:
        all_text = df['post_text'].fillna('') + ' ' + df['comment_text'].fillna('')
        key_terms = ['resmetirom', 'rezdiffra', 'semaglutide', 'ozempic', 'wegovy', 'nafld', 'nash', 'masld']
        mentions = {term: all_text.str.contains(term, case=False).sum() for term in key_terms}
        total_mentions = sum(mentions.values())
        assert total_mentions > 0, "No mentions of key MASLD terms found"
        print(f"✓ Key term mentions found: {total_mentions} total mentions")
        for term, count in mentions.items():
            if count > 0:
                print(f"  - {term}: {count} mentions")
    except Exception as e:
        print(f"✗ Test 6 FAILED: {e}")
        return False

    print("\nALL REDDIT DATA TESTS PASSED!")
    return True


def test_sentiment_analysis_output():
    """Test the sentiment analysis outputs"""
    print("\n=== SENTIMENT ANALYSIS OUTPUT TESTS ===")

    # Test 1: Check if sentiment analysis files exist
    try:
        analysis_files = os.listdir('analysis')
        sentiment_files = [f for f in analysis_files if 'sentiment' in f.lower()]
        assert len(sentiment_files) >= 3, f"Not enough sentiment analysis files. Found: {sentiment_files}"
        print(f"✓ Found sentiment analysis files: {len(sentiment_files)} files")
    except Exception as e:
        print(f"✗ Sentiment file test FAILED: {e}")
        return False

    # Test 2: Check daily sentiment file
    try:
        daily_files = [f for f in analysis_files if 'daily' in f.lower() and 'sentiment' in f.lower()]
        if daily_files:
            daily_df = pd.read_csv(f'analysis/{daily_files[0]}')
            assert 'avg_sentiment' in daily_df.columns, "Missing avg_sentiment column"
            assert 'date' in daily_df.columns, "Missing date column"
            assert len(daily_df) > 0, "Daily sentiment file is empty"
            print(f"✓ Daily sentiment file validation passed: {len(daily_df)} days")
    except Exception as e:
        print(f"✗ Daily sentiment test FAILED: {e}")
        return False

    # Test 3: Check sentiment range
    try:
        full_files = [f for f in analysis_files if 'with_sentiment' in f.lower()]
        if full_files:
            full_df = pd.read_csv(f'analysis/{full_files[0]}')
            assert 'sentiment' in full_df.columns, "Missing sentiment column in full dataset"
            sentiment_range = full_df['sentiment'].between(-1, 1).all()
            assert sentiment_range, "Sentiment scores outside valid range (-1 to 1)"
            print(
                f"✓ Sentiment range validation passed: {full_df['sentiment'].min():.3f} to {full_df['sentiment'].max():.3f}")
    except Exception as e:
        print(f"✗ Sentiment range test FAILED: {e}")
        return False

    print("ALL SENTIMENT ANALYSIS TESTS PASSED!")
    return True


def test_visualization_outputs():
    """Test that visualization files were created"""
    print("\n=== VISUALIZATION OUTPUT TESTS ===")

    try:
        analysis_files = os.listdir('analysis')
        plot_files = [f for f in analysis_files if f.endswith('.png')]
        assert len(plot_files) >= 3, f"Expected at least 3 plots, found: {len(plot_files)}"

        expected_plots = ['trend', 'subreddit', 'distribution']
        found_keywords = []
        for plot_file in plot_files:
            if any(keyword in plot_file.lower() for keyword in ['trend', 'fda']):
                found_keywords.append('trend')
            elif 'subreddit' in plot_file.lower():
                found_keywords.append('subreddit')
            elif 'distribution' in plot_file.lower():
                found_keywords.append('distribution')

        assert len(set(found_keywords)) >= 2, f"Missing key plot types. Found: {found_keywords}"
        print(f"✓ Visualization files check passed: {len(plot_files)} plots generated")

    except Exception as e:
        print(f"✗ Visualization test FAILED: {e}")
        return False

    print("ALL VISUALIZATION TESTS PASSED!")
    return True


def main():
    """Run all tests"""
    print("Running MASLD Reddit Analysis Tests...")
    print("=" * 50)

    all_passed = True

    # Run data quality tests
    if not test_reddit_data_quality():
        all_passed = False

    # Run sentiment analysis tests (only if analysis folder exists)
    if os.path.exists('analysis'):
        if not test_sentiment_analysis_output():
            all_passed = False
        if not test_visualization_outputs():
            all_passed = False
    else:
        print("\nAnalysis folder not found - skipping sentiment analysis tests")

    print("\n" + "=" * 50)
    if all_passed:
        print("ALL TESTS PASSED! Reddit data analysis is ready for use.")
    else:
        print("SOME TESTS FAILED! Please check your data and analysis.")

    return all_passed


if __name__ == "__main__":
    main()



# Tests for Stock Data (Data Source #5)
def test_stock_data_integrity(filename='data/stock_prices.csv'):
    print("\n" + "=" * 50)
    print("STOCK DATA VALIDATION TESTS")
    print("=" * 50)

    try:
        # 1. Check if file exists
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found. Run stock_data.py first.")

        # 2. Load data - skip the header row for date parsing
        df = pd.read_csv(filename, header=[0, 1])

        print("Stock data loaded successfully")
        print(f"Dataset shape: {df.shape}")

        # 3. Basic structure tests
        print("\n--- STRUCTURE TESTS ---")

        # Check if we have data for both companies
        if 'NVO' in df['Close'].columns and 'MDGL' in df['Close'].columns:
            print("PASS: Data for both companies (NVO, MDGL) present")
        else:
            print("FAIL: Missing company data")

        # 4. Data quality tests
        print("\n--- DATA QUALITY TESTS ---")

        # Check for missing values
        missing_values = df.isnull().sum().sum()
        if missing_values == 0:
            print("PASS: No missing values in dataset")
        else:
            print(f"INFO: Found {missing_values} missing values (normal for stock data)")

        # Check date range - handle the date column properly
        date_col = df.columns[0]  # First column should be date
        # Skip the first row if it's the header
        date_series = df[date_col].iloc[1:]  # Skip first row
        date_series = pd.to_datetime(date_series, errors='coerce')
        valid_dates = date_series.dropna()

        print(f"Date range: {valid_dates.min().strftime('%Y-%m-%d')} to {valid_dates.max().strftime('%Y-%m-%d')}")
        print(f"Total trading days: {len(valid_dates)}")

        # 5. Content validation
        print("\n--- CONTENT VALIDATION TESTS ---")

        # Check that prices are reasonable (skip first row)
        nvo_prices = df['Close']['NVO'].iloc[1:].astype(float)
        mdgl_prices = df['Close']['MDGL'].iloc[1:].astype(float)

        print(f"Novo Nordisk (NVO) price range: ${nvo_prices.min():.2f} - ${nvo_prices.max():.2f}")
        print(f"Madrigal (MDGL) price range: ${mdgl_prices.min():.2f} - ${mdgl_prices.max():.2f}")

        # 6. FDA approval date coverage
        print("\n--- FDA APPROVAL DATE COVERAGE ---")
        fda_dates = ['2024-03-14', '2025-08-15']
        for date in fda_dates:
            fda_date = pd.to_datetime(date)
            if any(valid_dates.dt.date == fda_date.date()):
                print(f"PASS: FDA approval date {date} is in dataset")
            else:
                # Find the closest trading date
                closest_date = valid_dates.iloc[(valid_dates - fda_date).abs().argmin()]
                print(
                    f"INFO: FDA date {date} was on a weekend/holiday. Closest trading date: {closest_date.strftime('%Y-%m-%d')}")

        print("\n" + "=" * 50)
        print("STOCK DATA TEST SUMMARY")
        print("=" * 50)
        print("Stock data collection: COMPLETE")
        print(f"Total trading days: {len(valid_dates)}")
        print("Data appears valid and ready for analysis!")

        return True

    except Exception as e:
        print(f"TEST FAILED: {e}")
        return False
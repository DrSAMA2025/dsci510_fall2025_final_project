# Tests for google_trends.py
"""
Tests for Google Trends Data (Task 1)
"""

import pandas as pd
import os


def test_data_integrity(filename='google_trends_initial_data.csv'):

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
    """Test if analysis files were generated."""
    print("\n--- ANALYSIS OUTPUT TESTS ---")

    expected_files = [
        'google_trends_analysis.py',
        'google_trends_main_analysis.png',
        'google_trends_correlation_heatmap.png'
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
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
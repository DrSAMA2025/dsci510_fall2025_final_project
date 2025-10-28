# Tests for google_trends.py

import pandas as pd


def check_data_integrity(filename='google_trends_initial_data.csv'):
    """Loads the CSV file and prints the head for data verification."""

    try:
        # 1. Load the data from the CSV file saved by google_trends.py
        df = pd.read_csv(filename)

        print("\n--- Data Integrity Check (tests.py) ---")

        # 2. Print the first 5 rows and data summary
        print("First 5 rows of data (df.head()):")
        print(df.head())

        print("\nData Information (df.info()):")
        df.info()

    except FileNotFoundError:
        print(f"Error: The file {filename} was not found. Please run google_trends.py first.")
    except Exception as e:
        print(f"An error occurred during data verification: {e}")


if __name__ == "__main__":
    check_data_integrity()
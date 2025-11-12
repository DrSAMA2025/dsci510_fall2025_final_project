import yfinance as yf
import pandas as pd
import os
from config import DATA_DIR  # Import the standard data directory path

def collect_stock_data():
    """
    Collects stock price data for specified pharmaceutical companies using yfinance,
    covering the project's defined timeframe, and saves the data to a CSV file.

    :return: A pandas DataFrame of the collected stock data, or None on failure.
    """
    print("--- Collecting Stock Data for MASLD Project ---")

    # Define companies and their tickers
    # NVO = Novo Nordisk (Wegovy, Ozempic)
    # MDGL = Madrigal Pharmaceuticals (Resmetirom/Rezdiffra)
    tickers = ['NVO', 'MDGL']

    # Date range matching the Google Trends data Timeframe
    start_date = '2023-01-01'
    # Note: Using a future date for end_date means yfinance will pull data up to the current date.
    end_date = '2025-10-28'

    print(f"Downloading data for {tickers} from {start_date} to {end_date}")

    try:
        # Download stock data
        stock_data = yf.download(tickers, start=start_date, end=end_date)

        # Construct the full file path using the DATA_DIR from config
        save_path = os.path.join(DATA_DIR, 'stock_prices.csv')

        # Ensure the data directory exists before saving
        os.makedirs(DATA_DIR, exist_ok=True)

        # Save to CSV
        stock_data.to_csv(save_path)
        print(f"Stock data successfully saved to {save_path}")
        print(f"Downloaded {len(stock_data)} days of data")

        return stock_data

    except Exception as e:
        print(f"Error collecting stock data: {e}")
        return None


if __name__ == "__main__":
    data = collect_stock_data()
    if data is not None:
        print("\nSuccess! Data collection complete.")
        print("Data Head:")
        print(data.head())
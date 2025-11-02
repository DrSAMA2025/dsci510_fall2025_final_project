# stock_data.py
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


def collect_stock_data():
    print("Collecting Stock Data for MASLD Project...")

    # Define companies and their tickers
    # NVO = Novo Nordisk (Wegovy, Ozempic)
    # MDGL = Madrigal Pharmaceuticals (Resmetirom/Rezdiffra)
    tickers = ['NVO', 'MDGL']

    # Date range matching the Google Trends data Timeframe
    start_date = '2023-01-01'
    end_date = '2025-10-28'

    print(f"Downloading data for {tickers} from {start_date} to {end_date}")

    try:
        # Download stock data
        stock_data = yf.download(tickers, start=start_date, end=end_date)

        # Save to CSV
        stock_data.to_csv('data/stock_prices.csv')
        print("Stock data saved to data/stock_prices.csv")
        print(f"Downloaded {len(stock_data)} days of data")

        return stock_data

    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    data = collect_stock_data()
    if data is not None:
        print("Success! Data collection complete.")
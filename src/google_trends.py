import pandas as pd
import time, random
from pytrends.request import TrendReq
from pytrends.exceptions import TooManyRequestsError
import config # <-- NEW: Import the central configuration

# --- Headers ---
custom_request_args = {
    'headers': {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/91.0.4472.124 Safari/537.36'
    }
}

# --- Initialize PyTrends ---
pytrend = TrendReq(hl='en-US', tz=360, requests_args=custom_request_args)

# --- Parameters ---
all_keywords = ['MASLD', 'NAFLD', 'Rezdiffra', 'Wegovy', 'Ozempic']
timeframe = '2023-01-01 2025-10-28'
geo = ''

# --- Container for all data ---
all_data = pd.DataFrame()

# --- Split into small batches ---
for i in range(0, len(all_keywords), 2):  # 2 keywords at a time
    batch = all_keywords[i:i+2]
    print(f"\nFetching data for: {batch}")
    success = False

    while not success:
        try:
            # Build payload and fetch data
            pytrend.build_payload(batch, cat=0, timeframe=timeframe, geo=geo)
            df = pytrend.interest_over_time()

            if 'isPartial' in df.columns:
                df = df.drop(columns=['isPartial'])

            all_data = pd.concat([all_data, df], axis=1)
            success = True
            print("Success.")

        except TooManyRequestsError:
            wait = random.randint(60, 120)  # wait 1–2 minutes
            print(f"Too many requests. Waiting {wait}s before retry...")
            time.sleep(wait)
        except Exception as e:
            print(f"Unexpected error: {e}")
            break

    time.sleep(random.uniform(5, 10))  # wait before next batch

# --- Save merged results ---
# Path Fix: Using config.DATA_DIR
output_path = config.DATA_DIR / 'google_trends_initial_data.csv'
all_data.to_csv(output_path, index=True) # <-- Fixed path
print(f"\nAll data saved to '{output_path}'.")


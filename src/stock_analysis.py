import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from config import DATA_DIR, RESULTS_DIR

# Define a subdirectory for the specific analysis results
STOCK_ANALYSIS_DIR = os.path.join(RESULTS_DIR, 'stock_analysis')

def analyze_stock_impact():
    """
    Performs stock market impact analysis for NVO and MDGL related to MASLD events,
    including normalized price trends, correlation with Google Trends data, and
    immediate FDA approval impact. Saves charts and data results to the RESULTS_DIR.
    """
    print("--- STOCK MARKET IMPACT ANALYSIS - FDA APPROVALS ---")
    print("=" * 50)

    # Define file paths for input data
    stock_data_path = os.path.join(DATA_DIR, 'stock_prices.csv')
    trends_data_path = os.path.join(DATA_DIR, 'google_trends_initial_data.csv')

    # 1. Load data and check for existence
    try:
        # Load stock data (using the multi-level header structure from yfinance)
        df = pd.read_csv(stock_data_path, header=[0, 1], index_col=0)
        df.index = pd.to_datetime(df.index)

        # Load Google Trends data
        trends_df = pd.read_csv(trends_data_path)
        trends_df['date'] = pd.to_datetime(trends_df['date'])
        trends_df.set_index('date', inplace=True)

    except FileNotFoundError as e:
        print(f"Error: Required input file not found. Please run data collection scripts first: {e}")
        return
    except Exception as e:
        print(f"Error loading or processing initial data: {e}")
        return

    print(f"Stock Data loaded: {len(df)} trading days")
    print(f"Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")

    # Ensure the output directory exists
    os.makedirs(STOCK_ANALYSIS_DIR, exist_ok=True)

    # FDA approval dates (placeholders for the Semaglutide approval)
    fda_dates = {
        'Resmetirom (Madrigal)': '2024-03-14', # Actual approval date for Rezdiffra
        'Semaglutide (Novo Nordisk)': '2025-08-15' # Placeholder date
    }

    # Set professional style for plots
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    # 1. STOCK PRICE TREND CHART
    print("\nCreating presentation-ready stock price chart...")
    plt.figure(figsize=(14, 8))

    # Normalize prices to percentage change for better comparison (Base price is the first closing price)
    nvo_normalized = (df['Close']['NVO'] / df['Close']['NVO'].iloc[0] - 1) * 100
    mdgl_normalized = (df['Close']['MDGL'] / df['Close']['MDGL'].iloc[0] - 1) * 100

    plt.plot(df.index, nvo_normalized, label='Novo Nordisk (NVO)', linewidth=3, color='#2E86AB')
    plt.plot(df.index, mdgl_normalized, label='Madrigal (MDGL)', linewidth=3, color='#A23B72')

    # Add FDA approval events
    colors = ['#F18F01', '#C73E1D']
    for i, (event, date) in enumerate(fda_dates.items()):
        event_date = pd.to_datetime(date)
        plt.axvline(event_date, color=colors[i], linestyle='--', linewidth=2, alpha=0.8, label=f'{event}')
        # Add text label for the event
        plt.text(event_date, plt.ylim()[1] * 0.85, event.split(' ')[0],
                 rotation=90, verticalalignment='top', fontweight='bold', fontsize=10)

    plt.title(
        'Stock Performance: MASLD Pharmaceutical Companies (2023-2025)\nNormalized Price Trends with FDA Approval Events',
        fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Price Change (%) from Jan 2023', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 0.95, 1]) # Adjust for legend
    plt.savefig(os.path.join(STOCK_ANALYSIS_DIR, 'stock_price_trend_presentation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(STOCK_ANALYSIS_DIR, 'stock_price_trend_presentation.png')}")


    # 2. CORRELATION MATRIX SETUP
    print("\nCreating correlation matrix...")

    # Merge datasets (resample to monthly averages for smoother correlation)
    stock_monthly = df['Close'].resample('ME').mean()
    trends_monthly = trends_df.resample('ME').mean()

    # Align dataframes and combine
    merged_data = pd.concat([stock_monthly, trends_monthly], axis=1)
    merged_data.dropna(inplace=True)

    # Select key variables for correlation
    correlation_vars = ['NVO', 'MDGL', 'MASLD', 'NAFLD', 'Wegovy', 'Ozempic']
    # Filter for columns that actually exist after merging/cleaning
    valid_correlation_vars = [col for col in correlation_vars if col in merged_data.columns]
    correlation_data = merged_data[valid_correlation_vars]

    # Create correlation matrix
    correlation_matrix = correlation_data.corr()

    # 3. CORRELATION MATRIX PLOT
    plt.figure(figsize=(10, 8))
    # Use upper triangle mask to avoid redundancy
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r',
                center=0, square=True, fmt='.3f', cbar_kws={"shrink": .8},
                annot_kws={'size': 10, 'weight': 'bold'})

    plt.title('Correlation Matrix: Stock Prices vs MASLD Search Interest (Monthly Averages)',
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(STOCK_ANALYSIS_DIR, 'correlation_matrix_presentation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(STOCK_ANALYSIS_DIR, 'correlation_matrix_presentation.png')}")


    # 4. FDA APPROVAL IMPACT ANALYSIS
    print("\n--- FDA APPROVAL IMPACT ANALYSIS (Immediate Event Impact) ---")

    results = []
    for event, date in fda_dates.items():
        event_date = pd.to_datetime(date)
        company = 'MDGL' if 'Resmetirom' in event else 'NVO'

        # Get the closing price on the last trading day *before* the event
        pre_period = df[event_date - pd.Timedelta(days=30):event_date - pd.Timedelta(days=1)]
        post_period = df[event_date:event_date + pd.Timedelta(days=30)]

        if len(pre_period) > 0:
            pre_price = pre_period['Close'][company].iloc[-1]
            try:
                # Use the 'Open' price on the event day to capture the immediate overnight/weekend change
                post_price = df['Open'][company].loc[event_date:].iloc[0]
            except IndexError:
                 # Fallback: Use the first available Close price after the event
                if len(post_period) > 0:
                     post_price = post_period['Close'][company].iloc[0]
                else:
                    post_price = pre_price # No post data available

            # Calculate immediate change (overnight/weekend)
            change_pct = (post_price - pre_price) / pre_price * 100

            print(f"{event} ({company}):")
            print(f"  Pre-approval Close: ${pre_price:.2f}")
            print(f"  Post-approval Open: ${post_price:.2f}")
            print(f"  Immediate change: {change_pct:+.2f}%")

            results.append({
                'Event': event,
                'Company': company,
                'Event_Date': date,
                'Pre_Approval_Price': f"{pre_price:.2f}",
                'Post_Approval_Price': f"{post_price:.2f}",
                'Immediate_Percent_Change': f"{change_pct:+.2f}%"
            })
        else:
             print(f"Warning: Insufficient pre-event data to analyze immediate impact for {event}.")


    # 5. CORRELATION RESULTS
    print("\n--- CORRELATION ANALYSIS (Results) ---")

    if 'MASLD' in correlation_matrix.index and 'NVO' in correlation_matrix.columns:
        correlation_nvo_masld = correlation_matrix.loc['NVO', 'MASLD']
        correlation_mdgl_masld = correlation_matrix.loc['MDGL', 'MASLD']

        print(f"Novo Nordisk vs MASLD searches: r = {correlation_nvo_masld:.3f}")
        print(f"Madrigal vs MASLD searches: r = {correlation_mdgl_masld:.3f}")

        # Save correlation results
        correlation_results = pd.DataFrame({
            'Correlation_Type': ['NVO_vs_MASLD', 'MDGL_vs_MASLD'],
            'Correlation_Value': [correlation_nvo_masld, correlation_mdgl_masld]
        })
        correlation_results_path = os.path.join(STOCK_ANALYSIS_DIR, 'stock_correlation_results.csv')
        correlation_results.to_csv(correlation_results_path, index=False)
        print(f"Saved: {correlation_results_path}")

    else:
        print("Could not compute correlations due to missing columns in merged data.")
        correlation_results = pd.DataFrame()


    # Save FDA impact results
    results_df = pd.DataFrame(results)
    impact_results_path = os.path.join(STOCK_ANALYSIS_DIR, 'stock_impact_results.csv')
    results_df.to_csv(impact_results_path, index=False)
    print(f"Saved: {impact_results_path}")

    print("\nAnalysis complete! All charts and data saved to the results directory.")


if __name__ == "__main__":
    analyze_stock_impact()
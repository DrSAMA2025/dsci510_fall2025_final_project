import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def analyze_stock_impact():
    print("STOCK MARKET IMPACT ANALYSIS - FDA APPROVALS")
    print("=" * 50)

    # Load stock data
    df = pd.read_csv('data/stock_prices.csv', header=[0, 1], index_col=0)
    df.index = pd.to_datetime(df.index)

    print(f"Data loaded: {len(df)} trading days")
    print(f"Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")

    # FDA approval dates
    fda_dates = {
        'Resmetirom (Madrigal)': '2024-03-14',
        'Semaglutide (Novo Nordisk)': '2025-08-15'
    }

    # 1. BASIC PRICE TRENDS
    plt.figure(figsize=(14, 10))

    # Plot 1: Stock prices with FDA events
    plt.subplot(2, 2, 1)
    plt.plot(df.index, df['Close']['NVO'], label='Novo Nordisk (NVO)', linewidth=2, color='blue')
    plt.plot(df.index, df['Close']['MDGL'], label='Madrigal (MDGL)', linewidth=2, color='red')

    for event, date in fda_dates.items():
        plt.axvline(pd.to_datetime(date), color='black', linestyle='--', alpha=0.7)
        plt.text(pd.to_datetime(date), plt.ylim()[1] * 0.9, event.split(' ')[0],
                 rotation=90, verticalalignment='top')

    plt.title('Stock Prices: FDA Approval Impact')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. PRICE CHANGES AROUND FDA DATES
    print("\n--- FDA APPROVAL IMPACT ANALYSIS ---")

    results = []
    for event, date in fda_dates.items():
        event_date = pd.to_datetime(date)
        company = 'MDGL' if 'Resmetirom' in event else 'NVO'

        # 30-day window around event
        pre_period = df[event_date - pd.Timedelta(days=30):event_date - pd.Timedelta(days=1)]
        post_period = df[event_date:event_date + pd.Timedelta(days=30)]

        if len(pre_period) > 0 and len(post_period) > 0:
            pre_price = pre_period['Close'][company].iloc[-1]
            post_price = post_period['Close'][company].iloc[0]
            change_pct = (post_price - pre_price) / pre_price * 100

            print(f"{event}:")
            print(f"  Pre-approval: ${pre_price:.2f}")
            print(f"  Post-approval: ${post_price:.2f}")
            print(f"  Immediate change: {change_pct:+.1f}%")

            results.append({
                'Event': event,
                'Company': company,
                'Pre_Approval_Price': pre_price,
                'Post_Approval_Price': post_price,
                'Percent_Change': change_pct
            })

    # 3. CORRELATION WITH GOOGLE TRENDS
    print("\n--- CORRELATION ANALYSIS ---")

    # Load Google Trends data for comparison
    trends_df = pd.read_csv('data/google_trends_initial_data.csv')
    trends_df['date'] = pd.to_datetime(trends_df['date'])
    trends_df.set_index('date', inplace=True)

    # Merge datasets (monthly averages for comparison)
    stock_monthly = df['Close'].resample('M').mean()
    trends_monthly = trends_df.resample('M').mean()

    merged_data = pd.concat([stock_monthly, trends_monthly], axis=1)

    # Calculate correlations
    correlation_nvo_masld = merged_data['NVO'].corr(merged_data['MASLD'])
    correlation_mdgl_masld = merged_data['MDGL'].corr(merged_data['MASLD'])

    print(f"Novo Nordisk vs MASLD searches: r = {correlation_nvo_masld:.3f}")
    print(f"Madrigal vs MASLD searches: r = {correlation_mdgl_masld:.3f}")

    # Save correlation results
    correlation_results = pd.DataFrame({
        'Correlation_Type': ['NVO_vs_MASLD', 'MDGL_vs_MASLD'],
        'Correlation_Value': [correlation_nvo_masld, correlation_mdgl_masld]
    })

    plt.tight_layout()
    plt.savefig('analysis/stock_analysis.png', dpi=300, bbox_inches='tight')

    # Save all results to analysis folder
    results_df = pd.DataFrame(results)
    results_df.to_csv('analysis/stock_analysis_results.csv', index=False)
    correlation_results.to_csv('analysis/stock_correlation_results.csv', index=False)

    print(f"\nAnalysis complete! Files saved to analysis/ folder:")
    print(f"  • analysis/stock_analysis.png")
    print(f"  • analysis/stock_analysis_results.csv")
    print(f"  • analysis/stock_correlation_results.csv")

    # REMOVED plt.show() - this was blocking the file saves!


if __name__ == "__main__":
    analyze_stock_impact()
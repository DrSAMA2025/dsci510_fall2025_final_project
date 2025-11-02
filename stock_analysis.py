# stock_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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

    # Set professional style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    # 1. STOCK PRICE TREND CHART
    print("\nCreating presentation-ready stock price chart...")
    plt.figure(figsize=(14, 8))

    # Normalize prices to percentage change for better comparison
    nvo_normalized = (df['Close']['NVO'] / df['Close']['NVO'].iloc[0] - 1) * 100
    mdgl_normalized = (df['Close']['MDGL'] / df['Close']['MDGL'].iloc[0] - 1) * 100

    plt.plot(df.index, nvo_normalized, label='Novo Nordisk (NVO)', linewidth=3, color='#2E86AB')
    plt.plot(df.index, mdgl_normalized, label='Madrigal (MDGL)', linewidth=3, color='#A23B72')

    # Add FDA approval events
    colors = ['#F18F01', '#C73E1D']
    for i, (event, date) in enumerate(fda_dates.items()):
        event_date = pd.to_datetime(date)
        plt.axvline(event_date, color=colors[i], linestyle='--', linewidth=2, alpha=0.8, label=f'{event}')
        plt.text(event_date, plt.ylim()[1] * 0.85, event.split(' ')[0],
                 rotation=90, verticalalignment='top', fontweight='bold', fontsize=10)

    plt.title(
        'Stock Performance: MASLD Pharmaceutical Companies (2023-2025)\nNormalized Price Trends with FDA Approval Events',
        fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Price Change (%) from Jan 2023', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('analysis/stock/stock_price_trend_presentation.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. CORRELATION MATRIX
    print("Creating correlation matrix...")

    # Load Google Trends data
    trends_df = pd.read_csv('data/google_trends_initial_data.csv')
    trends_df['date'] = pd.to_datetime(trends_df['date'])
    trends_df.set_index('date', inplace=True)

    # Merge datasets (monthly averages)
    stock_monthly = df['Close'].resample('ME').mean()
    trends_monthly = trends_df.resample('ME').mean()

    merged_data = pd.concat([stock_monthly, trends_monthly], axis=1)

    # Select key variables for correlation
    correlation_vars = ['NVO', 'MDGL', 'MASLD', 'NAFLD', 'Wegovy', 'Ozempic']
    correlation_data = merged_data[correlation_vars]

    # Create correlation matrix
    correlation_matrix = correlation_data.corr()

    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    heatmap = sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r',
                          center=0, square=True, fmt='.3f', cbar_kws={"shrink": .8},
                          annot_kws={'size': 10, 'weight': 'bold'})

    plt.title('Correlation Matrix: Stock Prices vs MASLD Search Interest',
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('analysis/stock/correlation_matrix_presentation.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. FDA APPROVAL IMPACT ANALYSIS
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

    # 4. CORRELATION ANALYSIS
    print("\n--- CORRELATION ANALYSIS ---")

    correlation_nvo_masld = correlation_matrix.loc['NVO', 'MASLD']
    correlation_mdgl_masld = correlation_matrix.loc['MDGL', 'MASLD']

    print(f"Novo Nordisk vs MASLD searches: r = {correlation_nvo_masld:.3f}")
    print(f"Madrigal vs MASLD searches: r = {correlation_mdgl_masld:.3f}")

    # Save results
    correlation_results = pd.DataFrame({
        'Correlation_Type': ['NVO_vs_MASLD', 'MDGL_vs_MASLD'],
        'Correlation_Value': [correlation_nvo_masld, correlation_mdgl_masld]
    })

    results_df = pd.DataFrame(results)
    results_df.to_csv('analysis/stock/stock_analysis_results.csv', index=False)
    correlation_results.to_csv('analysis/stock/stock_correlation_results.csv', index=False)

    print(f"\nAnalysis complete! Presentation-ready files saved:")
    print(f"   • analysis/stock/stock_price_trend_presentation.png")
    print(f"   • analysis/stock/correlation_matrix_presentation.png")
    print(f"   • analysis/stock/stock_analysis_results.csv")
    print(f"   • analysis/stock/stock_correlation_results.csv")


if __name__ == "__main__":
    analyze_stock_impact()
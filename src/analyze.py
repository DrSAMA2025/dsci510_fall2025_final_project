import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Import configuration constants
from config import (
    RESULTS_DIR, DATA_DIR,
    FDA_EVENT_DATES,
    PLOT_STYLE, COLOR_PALETTE, MEDIA_CLOUD_COLORS,
    GOOGLE_TRENDS_ANALYSIS_SUBDIR, STOCK_ANALYSIS_SUBDIR,
    REDDIT_ANALYSIS_SUBDIR, PUBMED_ANALYSIS_SUBDIR,
    MEDIA_CLOUD_ANALYSIS_SUBDIR, MEDIA_CLOUD_DATASETS
)

# Set plotting style globally
sns.set_theme(style=PLOT_STYLE)
plt.rcParams['figure.figsize'] = (12, 6)


def ensure_result_subdirs():
    """Ensures all necessary subdirectories in the results folder exist."""
    subdirs = [
        GOOGLE_TRENDS_ANALYSIS_SUBDIR,
        STOCK_ANALYSIS_SUBDIR,
        REDDIT_ANALYSIS_SUBDIR,
        PUBMED_ANALYSIS_SUBDIR,
        'media_cloud'
    ]
    for subdir in subdirs:
        (RESULTS_DIR / subdir).mkdir(exist_ok=True, parents=True)


# --- Analysis Functions ---

def analyze_google_trends(df_trends: pd.DataFrame, notebook_plot=False):
    """Generates a time series plot of all search terms."""
    print("\n[Analysis] Analyzing Google Trends...")
    save_dir = RESULTS_DIR / GOOGLE_TRENDS_ANALYSIS_SUBDIR

    plt.figure()
    ax = df_trends.plot(kind='line', colormap='viridis', linewidth=2)
    plt.title('Google Search Interest Over Time (2023-2025)')
    plt.xlabel('Date')
    plt.ylabel('Relative Search Volume Index')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Add ALL FDA approval dates as vertical lines
    for label, date_str in FDA_EVENT_DATES.items():
        plt.axvline(pd.to_datetime(date_str), color='red', linestyle='--', alpha=0.7)

    # Add custom legend for FDA events without overwriting the main legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='red', linestyle='--', linewidth=2)]

    # Get existing legend and combine with FDA legend
    existing_legend = ax.legend_
    if existing_legend is not None:
        # Combine both legends
        handles = existing_legend.legend_handles + custom_lines
        labels = [t.get_text() for t in existing_legend.get_texts()] + ['FDA Approvals']
        ax.legend(handles, labels, title='Search Term & Events')
    else:
        ax.legend(custom_lines, ['FDA Approvals'])

    if not notebook_plot:
        save_path = save_dir / "google_trends_timeline.png"
        plt.savefig(save_path)
        plt.close()
        print(f"  > Saved trend plot to: {save_path.name}")
    else:
        plt.show()
        print("  > Displayed Google Trends plot in notebook")


def analyze_reddit_sentiment(df_reddit: pd.DataFrame, notebook_plot=False):
    """Analyzes and visualizes Reddit sentiment over time."""
    print("\n[Analysis] Analyzing Reddit Sentiment...")
    save_dir = RESULTS_DIR / REDDIT_ANALYSIS_SUBDIR

    # Resample sentiment data weekly (or monthly for less noise)
    df_weekly_sentiment = df_reddit.set_index('timestamp')['sentiment_score'].resample('W').mean().dropna()

    plt.figure()
    df_weekly_sentiment.plot(kind='line', color='skyblue', linewidth=2)

    plt.title('Average Weekly Reddit Sentiment Score')
    plt.xlabel('Date')
    plt.ylabel('Average Compound Sentiment Score (VADER)')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Add FDA Approval Lines
    for label, date_str in FDA_EVENT_DATES.items():
        plt.axvline(pd.to_datetime(date_str), color='red', linestyle='--', alpha=0.7, label=f'{label}')

    # Add reference line for neutral sentiment
    plt.axhline(0.05, color='gray', linestyle=':', label='Slightly Positive Threshold')
    plt.axhline(-0.05, color='gray', linestyle=':', label='Slightly Negative Threshold')

    # Add legend
    plt.legend()

    if not notebook_plot:
        save_path = save_dir / "reddit_sentiment_timeline.png"
        plt.savefig(save_path)
        plt.close()
        print(f"  > Saved Reddit sentiment plot to: {save_path.name}")
    else:
        plt.show()
        print("  > Displayed Reddit sentiment plot in notebook")


def analyze_pubmed_publication_rate(df_pubmed: pd.DataFrame, notebook_plot=False):
    """Analyzes and visualizes PubMed publication rates for MASLD+drug combinations."""
    print("\n[Analysis] Analyzing PubMed Publications...")
    save_dir = RESULTS_DIR / PUBMED_ANALYSIS_SUBDIR

    # Create boolean columns
    df_pubmed['mentions_masld_resmetirom'] = (
                                                     df_pubmed['title'].str.contains('masld|mafld', case=False,
                                                                                     na=False) |
                                                     df_pubmed['abstract'].str.contains('masld|mafld', case=False,
                                                                                        na=False)
                                             ) & (
                                                     df_pubmed['title'].str.contains('resmetirom|rezdiffra', case=False,
                                                                                     na=False) |
                                                     df_pubmed['abstract'].str.contains('resmetirom|rezdiffra',
                                                                                        case=False, na=False)
                                             )

    df_pubmed['mentions_masld_glp1'] = (
                                               df_pubmed['title'].str.contains('masld|mafld', case=False, na=False) |
                                               df_pubmed['abstract'].str.contains('masld|mafld', case=False, na=False)
                                       ) & (
                                               df_pubmed['title'].str.contains('semaglutide|ozempic|wegovy|glp-1|glp1',
                                                                               case=False, na=False) |
                                               df_pubmed['abstract'].str.contains(
                                                   'semaglutide|ozempic|wegovy|glp-1|glp1', case=False, na=False)
                                       )

    # Group by month and plot
    monthly_data = df_pubmed.groupby(pd.Grouper(key='publication_date', freq='ME')).agg({
        'mentions_masld_resmetirom': 'sum',
        'mentions_masld_glp1': 'sum'
    })

    plt.figure(figsize=(12, 6))
    plt.plot(monthly_data.index, monthly_data['mentions_masld_resmetirom'],
             label='MASLD + Resmetirom', linewidth=2.5, color='#1f77b4')
    plt.plot(monthly_data.index, monthly_data['mentions_masld_glp1'],
             label='MASLD + GLP-1', linewidth=2.5, color='#ff7f0e')

    plt.title('Combined Disease+Drug Mentions in PubMed Publications Over Time')
    plt.xlabel('Publication Date')
    plt.ylabel('Monthly Publication Count')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Add FDA approval dates
    for label, date_str in FDA_EVENT_DATES.items():
        date = pd.to_datetime(date_str)
        plt.axvline(date, color='red', linestyle='--', alpha=0.7, linewidth=1.5)

    # Add Labels
    for label, date_str in FDA_EVENT_DATES.items():
        date = pd.to_datetime(date_str)
        plt.text(date, plt.ylim()[1] * 0.9, label, rotation=90,
                 verticalalignment='top', fontsize=9, color='red', backgroundcolor='white')

    if not notebook_plot:
        save_path = save_dir / "pubmed_drug_comparison_timeline.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  > Saved PubMed comparison plot to: {save_path.name}")
    else:
        plt.show()
        print("  > Displayed PubMed comparison plot in notebook")

    return monthly_data


def analyze_stock_and_events(df_stocks: pd.DataFrame, notebook_plot=False):
    """Analyzes stock movement relative to the key FDA event."""
    print("\n[Analysis] Analyzing Stock Data...")
    save_dir = RESULTS_DIR / STOCK_ANALYSIS_SUBDIR

    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot MDGL on primary axis
    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('MDGL Closing Price (USD)', color=color)
    ax1.plot(df_stocks.index, df_stocks['MDGL_Close'], color=color, linewidth=2, label='MDGL')
    ax1.tick_params(axis='y', labelcolor=color)

    # Create secondary axis for NVO
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('NVO Closing Price (USD)', color=color)
    ax2.plot(df_stocks.index, df_stocks['NVO_Close'], color=color, linewidth=2, label='NVO')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Stock Price Movement: MDGL vs. NVO')

    # Add FDA approval dates as vertical lines
    resmetirom_date = pd.to_datetime(FDA_EVENT_DATES['Resmetirom Approval'])
    glp1_date = pd.to_datetime(FDA_EVENT_DATES['GLP-1 Agonists Approval'])

    ax1.axvline(resmetirom_date, color='green', linestyle='-', linewidth=2, label='Rezdiffra Approval')
    ax1.axvline(glp1_date, color='orange', linestyle='-', linewidth=2, label='Wegovy Approval')

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    if not notebook_plot:
        save_path = save_dir / "stock_vs_events_timeline.png"
        plt.savefig(save_path)
        plt.close()
        print(f"  > Saved stock plot to: {save_path.name}")
    else:
        plt.show()
        print("  > Displayed stock plot in notebook")

    # Calculate impact for BOTH FDA events
    # Resmetirom (MDGL) impact
    try:
        pre_resmetirom = resmetirom_date - pd.Timedelta(days=5)
        post_resmetirom = resmetirom_date + pd.Timedelta(days=5)

        pre_price_mdgl = df_stocks.loc[df_stocks.index <= pre_resmetirom, 'MDGL_Close'].iloc[-1]
        post_price_mdgl = df_stocks.loc[df_stocks.index >= post_resmetirom, 'MDGL_Close'].iloc[0]

        mdgl_change = ((post_price_mdgl - pre_price_mdgl) / pre_price_mdgl) * 100
        print(f"  > MDGL Price Change around Resmetirom FDA: {mdgl_change:.2f}%")

    except IndexError:
        print("  > Warning: Could not calculate MDGL price change for Resmetirom approval.")

    # GLP-1 (NVO) impact
    try:
        pre_glp1 = glp1_date - pd.Timedelta(days=5)
        post_glp1 = glp1_date + pd.Timedelta(days=5)

        pre_price_nvo = df_stocks.loc[df_stocks.index <= pre_glp1, 'NVO_Close'].iloc[-1]
        post_price_nvo = df_stocks.loc[df_stocks.index >= post_glp1, 'NVO_Close'].iloc[0]

        nvo_change = ((post_price_nvo - pre_price_nvo) / pre_price_nvo) * 100
        print(f"  > NVO Price Change around GLP-1 FDA: {nvo_change:.2f}%")

    except IndexError:
        print("  > Warning: Could not calculate NVO price change for GLP-1 approval.")


def analyze_media_cloud_timeline(notebook_plot=False):
    """Create comparative timeline plot showing Media Cloud coverage trends."""
    print("\n[Analysis] Analyzing Media Cloud Timeline...")
    save_dir = RESULTS_DIR / MEDIA_CLOUD_ANALYSIS_SUBDIR

    try:
        from load import get_media_cloud_data
        media_data_available = get_media_cloud_data()

        if not media_data_available:
            print("  > Media Cloud data not available for analysis")
            return None

        # Load datasets
        datasets = load_media_cloud_datasets()
        if not datasets:
            print("  > No Media Cloud datasets loaded successfully")
            return None

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Plot 1: Individual monthly trends
        for name, data in datasets.items():
            if 'counts' in data:
                counts_df = data['counts'].copy()
                counts_df['date'] = pd.to_datetime(counts_df['date'])
                monthly = counts_df.set_index('date').sort_index().resample('ME').sum()

                ax1.plot(monthly.index, monthly['count'],
                         label=name.title(), linewidth=2.5, alpha=0.9,
                         color=MEDIA_CLOUD_COLORS.get(name))

        # Add FDA approval markers
        for label, date_str in FDA_EVENT_DATES.items():
            date = pd.to_datetime(date_str)
            ax1.axvline(date, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
            ax1.text(date + pd.Timedelta(days=5), ax1.get_ylim()[1] * 0.8, label, rotation=0,
                     horizontalalignment='left', verticalalignment='top',
                     fontsize=9, color='red', backgroundcolor='white', alpha=0.7)

        ax1.set_title('Media Coverage Timeline: MASLD and Related Drugs (Monthly Article Count)',
                      fontsize=14, fontweight='bold', pad=15)
        ax1.set_ylabel('Monthly Article Count', fontsize=11)
        ax1.legend(title='Search Query Type', loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        ax1.margins(x=0)

        # Plot 2: Cumulative coverage comparison
        for name, data in datasets.items():
            if 'counts' in data:
                counts_df = data['counts'].copy()
                counts_df['date'] = pd.to_datetime(counts_df['date'])
                counts_df = counts_df.sort_values('date')
                counts_df['cumulative'] = counts_df['count'].cumsum()

                ax2.plot(counts_df['date'], counts_df['cumulative'],
                         label=name.title(), linewidth=2, alpha=0.9,
                         color=MEDIA_CLOUD_COLORS.get(name))

        ax2.set_title('Cumulative Media Coverage Over Time', fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('Date', fontsize=11)
        ax2.set_ylabel('Cumulative Article Count', fontsize=11)
        ax2.legend(title='Search Query Type', loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        ax2.margins(x=0)

        plt.tight_layout(rect=[0, 0.03, 1, 0.98])

        if not notebook_plot:
            save_path = save_dir / "comparative_timeline.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  > Saved Media Cloud timeline to: {save_path.name}")
        else:
            plt.show()
            print("  > Displayed Media Cloud timeline in notebook")

        return datasets

    except Exception as e:
        print(f"  > Media Cloud timeline analysis error: {e}")
        return None


def analyze_media_cloud_sources(notebook_plot=False):
    """Analyze top media sources across Media Cloud datasets."""
    print("\n[Analysis] Analyzing Media Cloud Sources...")
    save_dir = RESULTS_DIR / MEDIA_CLOUD_ANALYSIS_SUBDIR

    try:
        datasets = load_media_cloud_datasets()
        if not datasets:
            print("  > No Media Cloud datasets available for source analysis")
            return None

        source_analysis = []
        plot_data = []

        for name, data in datasets.items():
            if 'sources' in data:
                sources_df = data['sources'].copy()
                top_sources = sources_df.head(10)

                for rank, (_, row) in enumerate(top_sources.iterrows(), 1):
                    source_analysis.append({
                        'Dataset': name,
                        'Source': row['source'],
                        'Article_Count': row['count'],
                        'Rank': rank
                    })
                    plot_data.append({
                        'Dataset': name.title(),
                        'Source': row['source'],
                        'Article_Count': row['count']
                    })

        # Create source comparison visualization
        plot_df = pd.DataFrame(plot_data)
        datasets_list = list(datasets.keys())
        n_datasets = len(datasets_list)

        fig, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 6), sharey=False)
        if n_datasets == 1:
            axes = [axes]

        for i, name in enumerate(datasets_list):
            subset = plot_df[plot_df['Dataset'] == name.title()].nlargest(10, 'Article_Count').sort_values(
                'Article_Count', ascending=True)
            axes[i].barh(subset['Source'], subset['Article_Count'], color=MEDIA_CLOUD_COLORS.get(name))
            axes[i].set_title(f'Top 10 Sources: {name.title()}', fontweight='bold')
            axes[i].set_xlabel('Article Count')
            axes[i].grid(axis='x', alpha=0.3)

        plt.suptitle('Comparison of Top Media Sources Across Search Queries', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if not notebook_plot:
            save_path = save_dir / "top_sources_comparison.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  > Saved Media Cloud sources plot to: {save_path.name}")
        else:
            plt.show()
            print("  > Displayed Media Cloud sources plot in notebook")

        return pd.DataFrame(source_analysis)

    except Exception as e:
        print(f"  > Media Cloud sources analysis error: {e}")
        return None


def load_media_cloud_datasets():
    media_cloud_data = DATA_DIR / "media_cloud"
    datasets = {}

    for dataset_name, folder_name in MEDIA_CLOUD_DATASETS.items():
        folder_path = media_cloud_data / folder_name

        if folder_path.exists():
            dataset_data = {}

            # Load counts data
            counts_files = list(folder_path.glob("*counts*.csv"))
            if counts_files:
                try:
                    dataset_data['counts'] = pd.read_csv(counts_files[0])
                except Exception as e:
                    print(f"Error loading counts for {dataset_name}: {e}")

            # Load sources data
            sources_files = list(folder_path.glob("*sources*.csv"))
            if sources_files:
                try:
                    dataset_data['sources'] = pd.read_csv(sources_files[0])
                except Exception as e:
                    print(f"Error loading sources for {dataset_name}: {e}")

            if dataset_data:
                datasets[dataset_name] = dataset_data

    return datasets


def run_all_analysis(processed_data: dict):
    """Runs all analysis and visualization functions."""
    print("=" * 60)
    print("STARTING DATA ANALYSIS & VISUALIZATION (analyze.py)")
    print("=" * 60)

    ensure_result_subdirs()

    if 'trends' in processed_data:
        analyze_google_trends(processed_data['trends'])

    if 'stocks' in processed_data:
        analyze_stock_and_events(processed_data['stocks'])

    if 'reddit' in processed_data:
        analyze_reddit_sentiment(processed_data['reddit'])

    if 'pubmed' in processed_data:
        analyze_pubmed_publication_rate(processed_data['pubmed'])

    # Add Media Cloud analysis
    analyze_media_cloud_timeline()
    analyze_media_cloud_sources()

    print("\nDATA ANALYSIS & VISUALIZATION COMPLETE.")


if __name__ == "__main__":
    # In a standalone run, we assume process.py saves processed files
    print("WARNING: Running analyze.py standalone requires processed files to be present.")

    # Placeholder for loading processed data if run directly (best practice is via main.py)
    # This requires a dedicated function to load processed files
    # For now, we only run if a dictionary is manually passed.
    pass
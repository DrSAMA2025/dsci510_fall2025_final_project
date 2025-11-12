import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from datetime import datetime
# Import configuration variables for path consistency and data structure
from config import DATA_DIR, RESULTS_DIR, MEDIA_CLOUD_DATASETS, FDA_EVENT_DATES

# Define specific subdirectories using configuration constants
MEDIA_CLOUD_RESULTS = os.path.join(RESULTS_DIR, 'media_cloud')
MEDIA_CLOUD_DATA = os.path.join(DATA_DIR, 'media_cloud')

print("Starting Media Cloud Analysis for MASLD Awareness Project")

# Set up plotting style and colors for a professional look
plt.style.use('seaborn-v0_8')
sns.set_palette("viridis", n_colors=3)
# Define specific colors for the three datasets (Viridis colormap selection)
COLORS = {'disease': '#440154', 'resmetirom': '#21908d', 'glp1': '#fde725'}


def setup_folders():
    """Create necessary folders for analysis output and check for data folder existence."""
    os.makedirs(MEDIA_CLOUD_RESULTS, exist_ok=True)
    print(f"Created results folder: {MEDIA_CLOUD_RESULTS}")

    if not os.path.exists(MEDIA_CLOUD_DATA):
        print(f"CRITICAL: Data folder not found: {MEDIA_CLOUD_DATA}")
        print("Please ensure your 'data/media_cloud' structure is correct.")
        return False
    return True


def load_media_cloud_datasets():
    """
    Load all Media Cloud datasets (counts, words, sources) using configuration from config.py.

    Returns:
        dict: A dictionary of loaded DataFrames, keyed by dataset type ('disease', 'resmetirom', 'glp1').
    """
    print("\nLoading Media Cloud datasets...")

    datasets = {}

    for dataset_name, folder_name in MEDIA_CLOUD_DATASETS.items():
        folder_path = os.path.join(MEDIA_CLOUD_DATA, folder_name)

        if os.path.exists(folder_path):
            print(f"\nProcessing {dataset_name} dataset from {folder_name}")

            dataset_data = {}
            loaded_files = 0

            # Use glob to find files based on common naming patterns
            for file_type in ['counts', 'top-words', 'top-sources']:
                pattern = os.path.join(folder_path, f"*-{file_type}.csv")
                files = glob.glob(pattern)

                if files:
                    # Load the first matching file
                    try:
                        df = pd.read_csv(files[0])
                        key = file_type.split('-')[-1] # 'counts', 'words', 'sources'
                        dataset_data[key] = df
                        print(f"  Loaded {key} data: {os.path.basename(files[0])}")
                        loaded_files += 1
                    except Exception as e:
                        print(f"  Error loading file {files[0]}: {e}")
                        continue
                else:
                    print(f"  No {file_type} file found in {folder_name}")

            if 'counts' in dataset_data: # Ensure counts data is available for analysis
                 datasets[dataset_name] = dataset_data
            elif loaded_files > 0:
                 print(f"Skipping {dataset_name}: 'counts' data is mandatory for analysis.")


        else:
            print(f"Folder not found: {folder_path}")

    return datasets


def analyze_basic_statistics(datasets):
    """
    Generate basic statistics (total articles, date range, peaks) for each dataset
    and save the results to a CSV file.
    """
    print("\n" + "=" * 70)
    print("BASIC DATASET STATISTICS")
    print("=" * 70)

    stats_data = []

    for name, data in datasets.items():
        print(f"\n{name.upper()} DATASET ANALYSIS:")

        if 'counts' in data:
            counts_df = data['counts'].copy()

            # Ensure date conversion is robust
            try:
                counts_df['date'] = pd.to_datetime(counts_df['date'])
            except Exception as e:
                print(f"Error converting dates in {name} dataset: {e}. Skipping analysis for this dataset.")
                continue

            total_articles = counts_df['count'].sum()
            date_range = f"{counts_df['date'].min().strftime('%Y-%m-%d')} to {counts_df['date'].max().strftime('%Y-%m-%d')}"
            num_days = len(counts_df)

            # Monthly analysis
            monthly = counts_df.set_index('date').sort_index().resample('M').sum()
            peak_month_idx = monthly['count'].idxmax() if not monthly.empty else None
            peak_month = peak_month_idx.strftime('%Y-%m') if peak_month_idx else 'N/A'
            peak_articles = monthly['count'].max() if not monthly.empty else 0
            avg_daily = counts_df['count'].mean() if num_days > 0 else 0

            print(f"  Timeline: {date_range}")
            print(f"  Total articles: {total_articles:,}")
            print(f"  Number of days: {num_days}")
            print(f"  Average daily articles: {avg_daily:.2f}")
            print(f"  Peak month: {peak_month} ({peak_articles:,} articles)")

            stats_data.append({
                'Dataset': name,
                'Total_Articles': total_articles,
                'Date_Range': date_range,
                'Days_Covered': num_days,
                'Avg_Daily_Articles': avg_daily,
                'Peak_Month': peak_month,
                'Peak_Month_Articles': peak_articles
            })

        if 'words' in data:
            words_df = data['words']
            print(f"  Unique terms: {len(words_df):,}")
            if len(words_df) > 0:
                top_5_words = words_df.head(5)[['term', 'term_count']].to_string(index=False)
                print(f"  Top 5 terms:\n{top_5_words}")

        if 'sources' in data:
            sources_df = data['sources']
            print(f"  Unique sources: {len(sources_df):,}")
            if len(sources_df) > 0:
                top_5_sources = sources_df.head(5)[['source', 'count']].to_string(index=False)
                print(f"  Top 5 sources:\n{top_5_sources}")

    # Save statistics to CSV
    stats_df = pd.DataFrame(stats_data)
    output_path = os.path.join(MEDIA_CLOUD_RESULTS, 'dataset_statistics.csv')
    stats_df.to_csv(output_path, index=False)
    print(f"\nSaved dataset statistics to: {output_path}")

    return stats_df


def create_comparative_timeline(datasets):
    """
    Create comparative timeline plot showing individual monthly trends and cumulative coverage.
    """
    print("\nCreating comparative timeline analysis...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Individual monthly trends
    for name, data in datasets.items():
        if 'counts' in data:
            counts_df = data['counts'].copy()
            counts_df['date'] = pd.to_datetime(counts_df['date'])
            # Resample to monthly
            monthly = counts_df.set_index('date').sort_index().resample('M').sum()

            ax1.plot(monthly.index, monthly['count'],
                     label=name.title(), linewidth=2.5, alpha=0.9,
                     color=COLORS.get(name))

    # Add FDA approval markers using FDA_EVENT_DATES from config
    for label, date_str in FDA_EVENT_DATES.items():
        date = pd.to_datetime(date_str)
        ax1.axvline(date, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        # Add text annotation
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
                     label=name.title(), linewidth=2, alpha=0.9, color=COLORS.get(name))

    ax2.set_title('Cumulative Media Coverage Over Time', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Cumulative Article Count', fontsize=11)
    ax2.legend(title='Search Query Type', loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    ax2.margins(x=0)

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    timeline_path = os.path.join(MEDIA_CLOUD_RESULTS, 'comparative_timeline.png')
    plt.savefig(timeline_path, dpi=300, bbox_inches='tight')
    plt.close() # Close figure after saving to free memory

    print(f"Saved comparative timeline to: {timeline_path}")


def analyze_fda_impact(datasets):
    """
    Analyze media coverage impact around FDA approval dates (30 days before vs. 30 days after)
    using dates defined in config.py.
    """
    print("\n" + "=" * 70)
    print("FDA APPROVAL IMPACT ANALYSIS")
    print("=" * 70)

    results = []

    for name, data in datasets.items():
        if 'counts' in data:
            counts_df = data['counts'].copy()
            counts_df['date'] = pd.to_datetime(counts_df['date'])

            for event_name, date_str in FDA_EVENT_DATES.items():
                fda_dt = pd.to_datetime(date_str)

                # 30 days before and after FDA event (excluding the day of the event)
                pre_period = counts_df[
                    (counts_df['date'] >= fda_dt - pd.Timedelta(days=30)) &
                    (counts_df['date'] < fda_dt)
                    ]

                post_period = counts_df[
                    (counts_df['date'] > fda_dt) &
                    (counts_df['date'] <= fda_dt + pd.Timedelta(days=30))
                    ]

                pre_count = pre_period['count'].sum()
                post_count = post_period['count'].sum()

                # Calculate change safely (handles division by zero)
                if pre_count > 0:
                    percent_change = ((post_count - pre_count) / pre_count) * 100
                    change_direction = "increase" if percent_change > 0 else "decrease"
                else:
                    percent_change = post_count * 100 # Represents growth from 0 to post_count
                    change_direction = "jump" if post_count > 0 else "none"

                results.append({
                    'Dataset': name,
                    'FDA_Event': event_name,
                    'Pre_FDA_Articles': pre_count,
                    'Post_FDA_Articles': post_count,
                    'Percent_Change': percent_change,
                    'Change_Direction': change_direction,
                    'Analysis_Period': '30_days'
                })

                print(f"\n{name.upper()} - {event_name} Event:")
                print(f"  30 days before: {pre_count:,} articles")
                print(f"  30 days after:  {post_count:,} articles")

                if pre_count > 0:
                    print(f"  Change: {percent_change:+.1f}% {change_direction}")
                else:
                    print(f"  Change: Baseline was 0. Post-event articles: {post_count:,}")

    # Save FDA impact results
    results_df = pd.DataFrame(results)
    output_path = os.path.join(MEDIA_CLOUD_RESULTS, 'fda_impact_analysis.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved FDA impact analysis to: {output_path}")

    return results_df


def analyze_media_sources(datasets):
    """
    Analyze and compare top 10 media sources across datasets and create a multi-panel visualization.

    Returns:
        pd.DataFrame: DataFrame containing top 10 source data for all datasets.
    """
    print("\n" + "=" * 70)
    print("MEDIA SOURCE ANALYSIS")
    print("=" * 70)

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

            print(f"\n{name.upper()} - Top 5 Media Sources:")
            print(top_sources.head(5)[['source', 'count']].to_string(index=False))

    # Save source analysis (all top 10 per category)
    source_df = pd.DataFrame(source_analysis)
    output_path = os.path.join(MEDIA_CLOUD_RESULTS, 'source_analysis.csv')
    source_df.to_csv(output_path, index=False)
    print(f"\nSaved source analysis to: {output_path}")

    # Create source comparison visualization (using subplots for clarity)
    plot_df = pd.DataFrame(plot_data)
    datasets_list = list(datasets.keys())
    n_datasets = len(datasets_list)
    fig, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 6), sharey=False)

    # Ensure axes is iterable even if only one subplot is created
    if n_datasets == 1:
        axes = [axes]

    for i, name in enumerate(datasets_list):
        # Filter for the current dataset, take the top 10, and sort
        subset = plot_df[plot_df['Dataset'] == name.title()].nlargest(10, 'Article_Count').sort_values('Article_Count', ascending=True)

        # Create horizontal bar plot
        axes[i].barh(subset['Source'], subset['Article_Count'], color=COLORS.get(name))
        axes[i].set_title(f'Top 10 Sources: {name.title()}', fontweight='bold')
        axes[i].set_xlabel('Article Count')
        axes[i].grid(axis='x', alpha=0.3)

    plt.suptitle('Comparison of Top Media Sources Across Search Queries', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    image_path = os.path.join(MEDIA_CLOUD_RESULTS, 'top_sources_comparison.png')
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved source comparison to: {image_path}")

    return source_df


def generate_summary_report(datasets, stats_df, fda_results, source_analysis):
    """Generate a comprehensive summary report based on all analysis steps."""
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY REPORT")
    print("=" * 70)

    total_articles = stats_df['Total_Articles'].sum()

    print(f"\nOVERVIEW:")
    print(f"Total articles across all queries: {total_articles:,}")
    try:
        start_date = stats_df['Date_Range'].apply(lambda x: x.split(' to ')[0]).min()
        end_date = stats_df['Date_Range'].apply(lambda x: x.split(' to ')[1]).max()
        print(f"Overall Time Period Covered: {start_date} to {end_date}")
    except Exception:
        print("Overall Time Period Covered: N/A (Error parsing date range)")

    print(f"Number of search queries analyzed: {len(datasets)}")

    print(f"\nCOVERAGE BY QUERY TYPE:")
    for _, row in stats_df.iterrows():
        percentage = (row['Total_Articles'] / total_articles) * 100
        print(f"  - {row['Dataset'].title()}: {row['Total_Articles']:,} articles ({percentage:.1f}%)")

    print(f"\nKEY FINDINGS:")

    # Most covered topic
    if not stats_df.empty and stats_df['Total_Articles'].max() > 0:
        most_covered = stats_df.loc[stats_df['Total_Articles'].idxmax()]
        print(f"  - Most Covered Topic: {most_covered['Dataset'].title()} ({most_covered['Total_Articles']:,} articles)")
        print(f"  - Peak Activity: {most_covered['Dataset'].title()} peaked in {most_covered['Peak_Month']} ({most_covered['Peak_Month_Articles']:,} articles).")
    else:
        print("  - No significant article counts to report.")

    # FDA impact summary
    significant_changes = fda_results[fda_results['Pre_FDA_Articles'] > 0]
    if not significant_changes.empty:
        max_increase_idx = significant_changes['Percent_Change'].idxmax()
        max_increase = significant_changes.loc[max_increase_idx]
        print(
            f"  - Largest FDA Impact: The **{max_increase['FDA_Event']}** event drove the largest **{max_increase['Change_Direction']}** of coverage in the **{max_increase['Dataset'].title()}** dataset, showing a **{max_increase['Percent_Change']:+.1f}%** change.")
    else:
        print("  - No baseline (pre-event) data available to calculate percent change for FDA events.")

    # Top overall source
    if not source_analysis.empty:
        overall_sources = source_analysis.groupby('Source')['Article_Count'].sum().nlargest(1)
        if not overall_sources.empty:
            top_source = overall_sources.index[0]
            top_source_count = overall_sources.iloc[0]
            print(f"  - Overall Dominant Source: **{top_source}** was the most prolific across all queries ({top_source_count:,} articles).")


def main():
    """Main analysis function to orchestrate data loading, analysis, and reporting."""
    print("MASLD Media Cloud Analysis - Comparative Study")
    print("=" * 60)

    # Step 1: Setup folders
    if not setup_folders():
        return

    # Step 2: Load all datasets
    datasets = load_media_cloud_datasets()

    if not datasets:
        print("No datasets found or loaded successfully. Please check your file organization.")
        return

    print(f"\nSuccessfully loaded {len(datasets)} datasets for analysis")

    # Step 3: Basic statistics
    stats_df = analyze_basic_statistics(datasets)

    # Step 4: Timeline analysis
    create_comparative_timeline(datasets)

    # Step 5: FDA impact analysis
    fda_results = analyze_fda_impact(datasets)

    # Step 6: Source analysis
    source_analysis_df = analyze_media_sources(datasets) # Capture the returned DataFrame

    # Step 7: Generate summary report
    generate_summary_report(datasets, stats_df, fda_results, source_analysis_df)

    print(f"\nANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"All outputs saved to: {MEDIA_CLOUD_RESULTS}/")


if __name__ == "__main__":
    # Fallback mechanism for standalone run
    try:
        from config import DATA_DIR, RESULTS_DIR, MEDIA_CLOUD_DATASETS, FDA_EVENT_DATES
    except ImportError:
        # Fallback if running standalone outside the project structure
        DATA_DIR = '../data'
        RESULTS_DIR = '../results'
        MEDIA_CLOUD_DATASETS = {
            'disease': 'disease_focused',
            'resmetirom': 'resmetirom_focused',
            'glp1': 'glp1_focused'
        }
        FDA_EVENT_DATES = {
            'Resmetirom Approval': '2024-03-14',
            'GLP-1 Anticipated Approval': '2025-08-15'
        }
        print("Warning: Could not import configuration. Using default relative paths and hardcoded config.")

    # Re-declare paths using the imported/defaulted values
    MEDIA_CLOUD_RESULTS = os.path.join(RESULTS_DIR, 'media_cloud')
    MEDIA_CLOUD_DATA = os.path.join(DATA_DIR, 'media_cloud')

    main()
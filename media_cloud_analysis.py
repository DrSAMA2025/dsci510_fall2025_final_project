# media_cloud_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from datetime import datetime

print("Starting Media Cloud Analysis for MASLD Awareness Project")

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def setup_folders():
    """Create necessary folders for analysis"""
    folders = [
        'analysis/media_cloud'
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created folder: {folder}")


def load_media_cloud_datasets():
    """Load all Media Cloud datasets from the three focused folders"""
    print("\nLoading Media Cloud datasets...")

    base_path = "data/media_cloud"
    datasets = {}

    # Define the datasets and their folders
    dataset_config = {
        'disease': 'disease_focused',
        'resmetirom': 'resmetirom_focused',
        'glp1': 'glp1_focused'
    }

    for dataset_name, folder_name in dataset_config.items():
        folder_path = os.path.join(base_path, folder_name)

        if os.path.exists(folder_path):
            print(f"\nProcessing {dataset_name} dataset from {folder_name}")

            dataset_data = {}

            # Find and load counts file
            counts_files = glob.glob(os.path.join(folder_path, "*-counts.csv"))
            if counts_files:
                counts_df = pd.read_csv(counts_files[0])
                dataset_data['counts'] = counts_df
                print(f"  Loaded counts data: {os.path.basename(counts_files[0])}")
            else:
                print(f"  No counts file found in {folder_name}")
                continue

            # Find and load top-words file
            words_files = glob.glob(os.path.join(folder_path, "*-top-words.csv"))
            if words_files:
                words_df = pd.read_csv(words_files[0])
                dataset_data['words'] = words_df
                print(f"  Loaded words data: {os.path.basename(words_files[0])}")

            # Find and load top-sources file
            sources_files = glob.glob(os.path.join(folder_path, "*-top-sources.csv"))
            if sources_files:
                sources_df = pd.read_csv(sources_files[0])
                dataset_data['sources'] = sources_df
                print(f"  Loaded sources data: {os.path.basename(sources_files[0])}")

            datasets[dataset_name] = dataset_data

        else:
            print(f"Folder not found: {folder_path}")

    return datasets


def analyze_basic_statistics(datasets):
    """Generate basic statistics for each dataset"""
    print("\n" + "=" * 70)
    print("BASIC DATASET STATISTICS")
    print("=" * 70)

    stats_data = []

    for name, data in datasets.items():
        print(f"\n{name.upper()} DATASET ANALYSIS:")

        if 'counts' in data:
            counts_df = data['counts']

            # Convert date column
            counts_df['date'] = pd.to_datetime(counts_df['date'])

            total_articles = counts_df['count'].sum()
            date_range = f"{counts_df['date'].min().strftime('%Y-%m-%d')} to {counts_df['date'].max().strftime('%Y-%m-%d')}"
            num_days = len(counts_df)

            # Monthly analysis
            monthly = counts_df.set_index('date').resample('M').sum()
            peak_month = monthly['count'].idxmax().strftime('%Y-%m')
            peak_articles = monthly['count'].max()
            avg_daily = counts_df['count'].mean()

            print(f"  Timeline: {date_range}")
            print(f"  Total articles: {total_articles:,}")
            print(f"  Number of days: {num_days}")
            print(f"  Average daily articles: {avg_daily:.2f}")
            print(f"  Peak month: {peak_month} ({peak_articles} articles)")

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
            print(f"  Unique terms: {len(words_df)}")
            if len(words_df) > 0:
                top_5_words = words_df.head(5)[['term', 'term_count']].to_string(index=False)
                print(f"  Top 5 terms:\n{top_5_words}")

        if 'sources' in data:
            sources_df = data['sources']
            print(f"  Unique sources: {len(sources_df)}")
            if len(sources_df) > 0:
                top_5_sources = sources_df.head(5)[['source', 'count']].to_string(index=False)
                print(f"  Top 5 sources:\n{top_5_sources}")

    # Save statistics to CSV
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv('analysis/media_cloud/dataset_statistics.csv', index=False)
    print(f"\nSaved dataset statistics to: analysis/media_cloud/dataset_statistics.csv")

    return stats_df


def create_comparative_timeline(datasets):
    """Create comparative timeline plot of all three datasets"""
    print("\nCreating comparative timeline analysis...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Define colors and styles for each dataset
    colors = {'disease': '#2E86AB', 'resmetirom': '#A23B72', 'glp1': '#F18F01'}
    styles = {'disease': '-', 'resmetirom': '--', 'glp1': '-.'}

    # Plot 1: Individual monthly trends
    for name, data in datasets.items():
        if 'counts' in data:
            counts_df = data['counts'].copy()
            counts_df['date'] = pd.to_datetime(counts_df['date'])

            # Resample to monthly
            monthly = counts_df.set_index('date').resample('M').sum()

            ax1.plot(monthly.index, monthly['count'],
                     label=name, linewidth=2.5, alpha=0.8,
                     color=colors.get(name), linestyle=styles.get(name))

    # Add FDA approval dates
    fda_dates = {
        'Resmetirom Approval': '2024-03-14',
        'GLP-1 for MASLD': '2025-08-15'
    }

    for label, date_str in fda_dates.items():
        date = pd.to_datetime(date_str)
        ax1.axvline(date, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax1.text(date, ax1.get_ylim()[1] * 0.9, label, rotation=90,
                 verticalalignment='top', fontweight='bold', fontsize=9)

    ax1.set_title('Media Coverage Timeline: MASLD and Related Drugs (2023-2025)',
                  fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylabel('Monthly Article Count', fontsize=11)
    ax1.legend(title='Search Query Type')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    # Plot 2: Cumulative coverage comparison
    for name, data in datasets.items():
        if 'counts' in data:
            counts_df = data['counts'].copy()
            counts_df['date'] = pd.to_datetime(counts_df['date'])
            counts_df['cumulative'] = counts_df['count'].cumsum()

            ax2.plot(counts_df['date'], counts_df['cumulative'],
                     label=name, linewidth=2, alpha=0.8, color=colors.get(name))

    ax2.set_title('Cumulative Media Coverage Over Time', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Cumulative Article Count', fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('analysis/media_cloud/comparative_timeline.png', dpi=300, bbox_inches='tight')
    plt.savefig('analysis/media_cloud/comparative_timeline.pdf', bbox_inches='tight')
    plt.show()

    print("Saved comparative timeline to: analysis/media_cloud/comparative_timeline.png")


def analyze_fda_impact(datasets):
    """Analyze media coverage impact around FDA approval dates"""
    print("\n" + "=" * 70)
    print("FDA APPROVAL IMPACT ANALYSIS")
    print("=" * 70)

    fda_dates = {
        'Resmetirom': '2024-03-14',
        'GLP-1_for_MASLD': '2025-08-15'
    }

    results = []

    for name, data in datasets.items():
        if 'counts' in data:
            counts_df = data['counts'].copy()
            counts_df['date'] = pd.to_datetime(counts_df['date'])

            for drug, fda_date in fda_dates.items():
                fda_dt = pd.to_datetime(fda_date)

                # 30 days before and after FDA approval
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

                if pre_count > 0:
                    percent_change = ((post_count - pre_count) / pre_count) * 100
                    change_direction = "increase" if percent_change > 0 else "decrease"
                else:
                    percent_change = float('inf') if post_count > 0 else 0
                    change_direction = "N/A"

                results.append({
                    'Dataset': name,
                    'FDA_Event': drug,
                    'Pre_FDA_Articles': pre_count,
                    'Post_FDA_Articles': post_count,
                    'Percent_Change': percent_change,
                    'Change_Direction': change_direction,
                    'Analysis_Period': '30_days'
                })

                print(f"\n{name.upper()} - {drug} FDA Approval:")
                print(f"  30 days before: {pre_count} articles")
                print(f"  30 days after:  {post_count} articles")
                if pre_count > 0:
                    print(f"  Change: {percent_change:+.1f}% {change_direction}")
                else:
                    print(f"  Change: No baseline articles for comparison")

    # Save FDA impact results
    results_df = pd.DataFrame(results)
    results_df.to_csv('analysis/media_cloud/fda_impact_analysis.csv', index=False)
    print(f"\nSaved FDA impact analysis to: analysis/media_cloud/fda_impact_analysis.csv")

    return results_df


def analyze_media_sources(datasets):
    """Analyze and compare media sources across datasets"""
    print("\n" + "=" * 70)
    print("MEDIA SOURCE ANALYSIS")
    print("=" * 70)

    source_analysis = []

    for name, data in datasets.items():
        if 'sources' in data:
            sources_df = data['sources'].copy()

            # Get top 10 sources for this dataset
            top_sources = sources_df.head(10)

            for _, row in top_sources.iterrows():
                source_analysis.append({
                    'Dataset': name,
                    'Source': row['source'],
                    'Article_Count': row['count'],
                    'Rank': _ + 1
                })

            print(f"\n{name.upper()} - Top 5 Media Sources:")
            print(top_sources.head(5)[['source', 'count']].to_string(index=False))

    # Save source analysis
    source_df = pd.DataFrame(source_analysis)
    source_df.to_csv('analysis/media_cloud/source_analysis.csv', index=False)
    print(f"\nSaved source analysis to: analysis/media_cloud/source_analysis.csv")

    # Create source comparison visualization
    plt.figure(figsize=(12, 8))

    for name in datasets.keys():
        if 'sources' in datasets[name]:
            sources_df = datasets[name]['sources'].head(10)
            plt.barh([f"{source} ({name})" for source in sources_df['source']],
                     sources_df['count'], alpha=0.7, label=name)

    plt.xlabel('Article Count')
    plt.title('Top Media Sources by Search Query Type', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('analysis/media_cloud/top_sources_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Saved source comparison to: analysis/media_cloud/top_sources_comparison.png")

    return source_df


def generate_summary_report(datasets, stats_df, fda_results, source_analysis):
    """Generate a comprehensive summary report"""
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY REPORT")
    print("=" * 70)

    total_articles = stats_df['Total_Articles'].sum()

    print(f"\nOVERVIEW:")
    print(f"Total articles across all queries: {total_articles:,}")
    print(f"Time period covered: Jan 2023 - Oct 2025")
    print(f"Number of search queries analyzed: {len(datasets)}")

    print(f"\nCOVERAGE BY QUERY TYPE:")
    for _, row in stats_df.iterrows():
        percentage = (row['Total_Articles'] / total_articles) * 100
        print(f"  {row['Dataset']}: {row['Total_Articles']:,} articles ({percentage:.1f}%)")

    print(f"\nKEY FINDINGS:")

    # Most covered topic
    most_covered = stats_df.loc[stats_df['Total_Articles'].idxmax()]
    print(f"  Most covered topic: {most_covered['Dataset']} ({most_covered['Total_Articles']:,} articles)")

    # FDA impact summary
    significant_changes = fda_results[fda_results['Pre_FDA_Articles'] > 0]
    if len(significant_changes) > 0:
        max_increase = significant_changes.loc[significant_changes['Percent_Change'].idxmax()]
        print(
            f"  Largest FDA impact: {max_increase['Dataset']} for {max_increase['FDA_Event']} ({max_increase['Percent_Change']:+.1f}%)")

    print(f"\nANALYSIS COMPLETE!")
    print("All results saved to: analysis/media_cloud/")


def main():
    """Main analysis function"""
    print("MASLD Media Cloud Analysis - Comparative Study")
    print("=" * 60)

    # Step 1: Setup folders
    setup_folders()

    # Step 2: Load all datasets
    datasets = load_media_cloud_datasets()

    if not datasets:
        print("No datasets found. Please check your file organization.")
        return

    print(f"\nSuccessfully loaded {len(datasets)} datasets for analysis")

    # Step 3: Basic statistics
    stats_df = analyze_basic_statistics(datasets)

    # Step 4: Timeline analysis
    create_comparative_timeline(datasets)

    # Step 5: FDA impact analysis
    fda_results = analyze_fda_impact(datasets)

    # Step 6: Source analysis
    source_analysis = analyze_media_sources(datasets)

    # Step 7: Generate summary report
    generate_summary_report(datasets, stats_df, fda_results, source_analysis)

    print(f"\nANALYSIS COMPLETED SUCCESSFULLY!")
    print("All outputs saved to: analysis/media_cloud/")


if __name__ == "__main__":
    main()
# pubmed_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
from collections import Counter
import os

# --- ANALYSIS CONSTANTS ---
# Use constants for key dates to centralize them for easy updating
RESMETIROM_APPROVAL_DATE = '2024-03'
SEMAGLUTIDE_APPROVAL_DATE = '2025-08'
# --- END CONSTANTS ---


def setup_analysis_environment():
    """
    Set up plotting style and create output folders.

    Returns:
        str: The path to the created analysis folder.
    """
    # Use a clean, modern plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")

    # Create analysis subfolder for PubMed
    pubmed_analysis_folder = 'analysis/pubmed'
    if not os.path.exists(pubmed_analysis_folder):
        os.makedirs(pubmed_analysis_folder)

    return pubmed_analysis_folder


def load_latest_pubmed_data():
    """
    Load the most recent PubMed dataset from the 'data' directory.

    Raises:
        FileNotFoundError: If no data files are found.

    Returns:
        tuple: (pandas.DataFrame, str: latest_file)
    """
    pubmed_files = [f for f in os.listdir('data') if f.startswith('pubmed_masld_articles') and f.endswith('.csv')]
    if not pubmed_files:
        raise FileNotFoundError("No PubMed data files found. Please run pubmed_data_collection.py first.")

    # Find the latest file based on creation time (or modification time)
    latest_file = max(pubmed_files, key=lambda x: os.path.getctime(os.path.join('data', x)))
    filepath = os.path.join('data', latest_file)

    print(f"Loading PubMed data from: {latest_file}")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} articles")

    return df, latest_file


def clean_and_preprocess_data(df):
    """
    Clean and preprocess PubMed data for analysis by converting dates and
    creating boolean flags for key terms.

    Returns:
        pandas.DataFrame: The cleaned and enriched DataFrame.
    """
    # Convert publication date to datetime, coercing errors
    df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')

    # Extract year, month, and period for time-series analysis
    df['year'] = df['publication_date'].dt.year
    df['month'] = df['publication_date'].dt.month
    df['year_month'] = df['publication_date'].dt.to_period('M')

    # Drop rows with missing publication dates after coercion
    df.dropna(subset=['publication_date'], inplace=True)

    # Helper function for checking mentions in title or abstract
    def check_mentions(df, terms):
        return df['title'].str.contains(terms, case=False, na=False) | df['abstract'].str.contains(terms, case=False,
                                                                                                   na=False)

    # Create flags for key disease terms
    df['mentions_masld'] = check_mentions(df, 'MASLD|MAFLD')
    df['mentions_nafld'] = check_mentions(df, 'NAFLD')
    df['mentions_nash'] = check_mentions(df, 'NASH')
    df['mentions_mash'] = check_mentions(df, 'MASH')

    # Create flags for key drug terms
    df['mentions_resmetirom'] = check_mentions(df, 'resmetirom|rezdiffra')
    df['mentions_glp1'] = check_mentions(df, 'semaglutide|ozempic|wegovy|glp-1|glp1|liraglutide')

    # Combined mentions (both disease and drug)
    df['mentions_masld_resmetirom'] = df['mentions_masld'] & df['mentions_resmetirom']
    df['mentions_masld_glp1'] = df['mentions_masld'] & df['mentions_glp1']
    df['mentions_nafld_resmetirom'] = df['mentions_nafld'] & df['mentions_resmetirom']
    df['mentions_nafld_glp1'] = df['mentions_nafld'] & df['mentions_glp1']

    return df


def analyze_publication_trends(df, analysis_folder):
    """Analyze overall publication trends over time and plot key FDA approval dates."""
    print("Analyzing publication trends...")

    # Monthly publication count
    monthly_trends = df.groupby('year_month').size()

    # Plot monthly trends
    plt.figure(figsize=(14, 8))
    monthly_trends.plot(kind='line', linewidth=2.5, marker='o', color='#2E86AB')

    # Add FDA approval dates using defined constants
    plt.axvline(pd.Period(RESMETIROM_APPROVAL_DATE, 'M'), color='red', linestyle='--', alpha=0.8, linewidth=2,
                label=f'Resmetirom FDA Approval (Mar {RESMETIROM_APPROVAL_DATE.split("-")[0]})')
    plt.axvline(pd.Period(SEMAGLUTIDE_APPROVAL_DATE, 'M'), color='orange', linestyle='--', alpha=0.8, linewidth=2,
                label=f'Semaglutide FDA Approval (Aug {SEMAGLUTIDE_APPROVAL_DATE.split("-")[0]})')

    plt.title('PubMed Publications on MASLD and Related Treatments (2023-2025)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Date')
    plt.ylabel('Number of Publications')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{analysis_folder}/pubmed_publication_trends.png', dpi=300, bbox_inches='tight')
    plt.close()

    return monthly_trends


def analyze_term_mentions(df, analysis_folder):
    """Analyze mentions of key terms (disease names and drugs) over time."""
    print("Analyzing key term mentions...")

    # Prepare monthly data for key terms
    monthly_data = df.groupby('year_month').agg({
        'mentions_masld': 'sum',
        'mentions_nafld': 'sum',
        'mentions_resmetirom': 'sum',
        'mentions_glp1': 'sum',
        'mentions_masld_resmetirom': 'sum',
        'mentions_masld_glp1': 'sum'
    })

    # Plot term mentions
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 1, 1)
    plt.plot(monthly_data.index.astype(str), monthly_data['mentions_masld'], label='MASLD', linewidth=2.5, marker='o',
             color='blue')
    plt.plot(monthly_data.index.astype(str), monthly_data['mentions_nafld'], label='NAFLD', linewidth=2.5, marker='s',
             color='green')
    plt.plot(monthly_data.index.astype(str), monthly_data['mentions_resmetirom'], label='Resmetirom', linewidth=2.5,
             marker='^', color='red')
    plt.plot(monthly_data.index.astype(str), monthly_data['mentions_glp1'], label='GLP-1', linewidth=2.5, marker='d',
             color='purple')

    # Use string constants for axvline when x-axis is string-converted PeriodIndex
    plt.axvline(RESMETIROM_APPROVAL_DATE, color='red', linestyle='--', alpha=0.7, label='Resmetirom FDA Approval')
    plt.axvline(SEMAGLUTIDE_APPROVAL_DATE, color='orange', linestyle='--', alpha=0.7, label='Semaglutide FDA Approval')

    plt.title('Key Term Mentions in PubMed Publications Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Number of Mentions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    # Plot combined mentions (disease + drug)
    plt.subplot(2, 1, 2)
    plt.plot(monthly_data.index.astype(str), monthly_data['mentions_masld_resmetirom'], label='MASLD + Resmetirom',
             linewidth=2.5, marker='o', color='darkred')
    plt.plot(monthly_data.index.astype(str), monthly_data['mentions_masld_glp1'], label='MASLD + GLP-1', linewidth=2.5,
             marker='s', color='darkviolet')

    plt.axvline(RESMETIROM_APPROVAL_DATE, color='red', linestyle='--', alpha=0.7)
    plt.axvline(SEMAGLUTIDE_APPROVAL_DATE, color='orange', linestyle='--', alpha=0.7)

    plt.title('Combined Disease + Drug Mentions in PubMed Publications', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Number of Mentions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(f'{analysis_folder}/pubmed_term_mentions.png', dpi=300, bbox_inches='tight')
    plt.close()

    return monthly_data


def analyze_terminology_adoption(df, analysis_folder):
    """Analyze adoption of new terminology (MASLD vs NAFLD) over time."""
    print("Analyzing terminology adoption...")

    # Calculate terminology adoption over time
    terminology_trends = df.groupby('year_month').agg({
        'mentions_masld': 'sum',
        'mentions_nafld': 'sum'
    })

    # Calculate the ratio of MASLD mentions relative to total MASLD/NAFLD mentions
    terminology_trends['total_mentions'] = terminology_trends['mentions_masld'] + terminology_trends['mentions_nafld']
    # Use fillna(0) to avoid division by zero in case of months with no mentions
    terminology_trends['masld_ratio'] = (terminology_trends['mentions_masld'] / terminology_trends['total_mentions']).fillna(0)

    # Plot terminology adoption
    plt.figure(figsize=(14, 8))

    plt.subplot(1, 2, 1)
    plt.plot(terminology_trends.index.astype(str), terminology_trends['mentions_masld'], label='MASLD', linewidth=2.5,
             marker='o', color='blue')
    plt.plot(terminology_trends.index.astype(str), terminology_trends['mentions_nafld'], label='NAFLD', linewidth=2.5,
             marker='s', color='green')
    plt.axvline(RESMETIROM_APPROVAL_DATE, color='red', linestyle='--', alpha=0.7)
    plt.axvline(SEMAGLUTIDE_APPROVAL_DATE, color='orange', linestyle='--', alpha=0.7)
    plt.title('MASLD vs NAFLD Terminology Usage (Raw Count)', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Number of Mentions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    plt.subplot(1, 2, 2)
    plt.plot(terminology_trends.index.astype(str), terminology_trends['masld_ratio'] * 100, linewidth=2.5, marker='o',
             color='purple')
    plt.axvline(RESMETIROM_APPROVAL_DATE, color='red', linestyle='--', alpha=0.7, label='Resmetirom FDA Approval')
    plt.axvline(SEMAGLUTIDE_APPROVAL_DATE, color='orange', linestyle='--', alpha=0.7, label='Semaglutide FDA Approval')
    plt.title('MASLD as Percentage of Total Terminology', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('MASLD Percentage (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(f'{analysis_folder}/pubmed_terminology_adoption.png', dpi=300, bbox_inches='tight')
    plt.close()

    return terminology_trends


def analyze_journal_distribution(df, analysis_folder):
    """Analyze distribution of publications across journals and save a plot."""
    print("Analyzing journal distribution...")

    # Top journals by publication count
    top_journals = df['journal'].value_counts().head(15)

    plt.figure(figsize=(12, 8))
    top_journals.plot(kind='barh', color='skyblue')
    plt.title('Top 15 Journals Publishing MASLD Research (2023-2025)', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Publications')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{analysis_folder}/pubmed_top_journals.png', dpi=300, bbox_inches='tight')
    plt.close()

    return top_journals


def generate_summary_statistics(df, analysis_folder):
    """Generate comprehensive summary statistics and save them to CSV."""
    print("Generating summary statistics...")

    summary_stats = {
        'total_publications': len(df),
        # Assuming 'has_abstract' is a column with True/False (or 1/0)
        'publications_with_abstracts': df['has_abstract'].sum() if 'has_abstract' in df.columns else 'N/A',
        'publications_mentioning_masld': df['mentions_masld'].sum(),
        'publications_mentioning_nafld': df['mentions_nafld'].sum(),
        'publications_mentioning_nash': df['mentions_nash'].sum(),
        'publications_mentioning_mash': df['mentions_mash'].sum(),
        'publications_mentioning_resmetirom': df['mentions_resmetirom'].sum(),
        'publications_mentioning_glp1': df['mentions_glp1'].sum(),
        'publications_mentioning_masld_resmetirom': df['mentions_masld_resmetirom'].sum(),
        'publications_mentioning_masld_glp1': df['mentions_masld_glp1'].sum(),
        'unique_journals': df['journal'].nunique(),
        # Retrieve actual min/max dates from the DataFrame
        'date_range_start': df['publication_date'].min(),
        'date_range_end': df['publication_date'].max()
    }

    # Create summary table
    summary_df = pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value'])
    summary_df.to_csv(f'{analysis_folder}/pubmed_summary_statistics.csv', index=False)

    # Yearly breakdown
    yearly_stats = df.groupby('year').agg({
        'pubmed_id': 'count',
        'mentions_masld': 'sum',
        'mentions_nafld': 'sum',
        'mentions_resmetirom': 'sum',
        'mentions_glp1': 'sum'
    }).rename(columns={'pubmed_id': 'total_publications'})

    yearly_stats.to_csv(f'{analysis_folder}/pubmed_yearly_statistics.csv')

    return summary_stats, yearly_stats


def create_comprehensive_report(df, summary_stats, analysis_folder, source_file):
    """Create a comprehensive analysis report in text format."""
    report_file = f"{analysis_folder}/pubmed_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    # Convert start/end dates to strings for reporting (already done in summary_stats)
    start_date_str = summary_stats['date_range_start'].strftime('%Y-%m-%d')
    end_date_str = summary_stats['date_range_end'].strftime('%Y-%m-%d')

    with open(report_file, 'w') as f:
        f.write("COMPREHENSIVE PUBMED ANALYSIS REPORT - MASLD PROJECT\n")
        f.write("=" * 60 + "\n\n")

        f.write("STUDY OVERVIEW\n")
        f.write("-" * 20 + "\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Data Source: {source_file}\n")
        # Use dynamic dates from summary_stats
        f.write(f"Study Period: {start_date_str} to {end_date_str}\n")
        f.write(f"Key FDA Dates: Resmetirom (Mar {RESMETIROM_APPROVAL_DATE}), Semaglutide (Aug {SEMAGLUTIDE_APPROVAL_DATE})\n\n")

        f.write("DATA SUMMARY\n")
        f.write("-" * 20 + "\n")
        # Use a list of keys for consistent order, filtering out non-numerical 'N/A'
        report_keys = [
            'total_publications',
            'unique_journals',
            'publications_with_abstracts',
            'publications_mentioning_masld',
            'publications_mentioning_nafld',
            'publications_mentioning_resmetirom',
            'publications_mentioning_glp1',
            'publications_mentioning_masld_resmetirom',
            'publications_mentioning_masld_glp1',
        ]

        for key in report_keys:
            f.write(f"{key.replace('_', ' ').title()}: {summary_stats.get(key, 'N/A')}\n")

        # Calculate total publications for percentage base
        total = summary_stats['total_publications']

        f.write("\nKEY FINDINGS (PERCENTAGES)\n")
        f.write("-" * 20 + "\n")
        if total > 0:
            f.write(
                f"MASLD mentioned in: {summary_stats['publications_mentioning_masld']} articles ({summary_stats['publications_mentioning_masld'] / total * 100:.1f}%)\n")
            f.write(
                f"NAFLD mentioned in: {summary_stats['publications_mentioning_nafld']} articles ({summary_stats['publications_mentioning_nafld'] / total * 100:.1f}%)\n")
            f.write(
                f"Resmetirom mentioned in: {summary_stats['publications_mentioning_resmetirom']} articles ({summary_stats['publications_mentioning_resmetirom'] / total * 100:.1f}%)\n")
            f.write(
                f"GLP-1 agonists mentioned in: {summary_stats['publications_mentioning_glp1']} articles ({summary_stats['publications_mentioning_glp1'] / total * 100:.1f}%)\n")
        else:
            f.write("No publications found for percentage calculation.\n")

        f.write(f"\nFILES GENERATED\n")
        f.write("-" * 20 + "\n")
        f.write("Visualizations (in analysis/pubmed/):\n")
        f.write("  - pubmed_publication_trends.png\n")
        f.write("  - pubmed_term_mentions.png\n")
        f.write("  - pubmed_terminology_adoption.png\n")
        f.write("  - pubmed_top_journals.png\n")
        f.write("\nData Tables (in analysis/pubmed/):\n")
        f.write("  - pubmed_summary_statistics.csv\n")
        f.write("  - pubmed_yearly_statistics.csv\n")
        f.write(f"  - {os.path.basename(report_file)}\n")

    return report_file


def main():
    """Main function to coordinate PubMed analysis"""
    print("Starting PubMed Data Analysis for MASLD Project")
    print("=" * 50)

    try:
        # Setup environment
        analysis_folder = setup_analysis_environment()

        # Load data
        df, source_file = load_latest_pubmed_data()

        # Clean and preprocess
        df = clean_and_preprocess_data(df)

        # Perform analyses
        monthly_trends = analyze_publication_trends(df, analysis_folder)
        term_mentions = analyze_term_mentions(df, analysis_folder)
        terminology_adoption = analyze_terminology_adoption(df, analysis_folder)
        journal_distribution = analyze_journal_distribution(df, analysis_folder)
        summary_stats, yearly_stats = generate_summary_statistics(df, analysis_folder)

        # Create comprehensive report
        report_file = create_comprehensive_report(df, summary_stats, analysis_folder, source_file)

        # Print summary
        print("\nANALYSIS COMPLETE")
        print("=" * 50)
        print(f"Total publications analyzed: {len(df)}")
        print(f"MASLD mentions: {summary_stats['publications_mentioning_masld']}")
        print(f"NAFLD mentions: {summary_stats['publications_mentioning_nafld']}")
        print(f"Resmetirom mentions: {summary_stats['publications_mentioning_resmetirom']}")
        print(f"GLP-1 mentions: {summary_stats['publications_mentioning_glp1']}")
        print(f"All outputs saved to: {analysis_folder}/")
        print(f"Comprehensive report: {report_file}")

    except Exception as e:
        print(f"Error during analysis: {e}")
        # Re-raise the exception for clearer debugging if run from outside main()
        # raise


if __name__ == "__main__":
    main()
# pubmed_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
from collections import Counter
import os


def setup_analysis_environment():
    """Set up plotting style and folders"""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    # Create analysis subfolder for PubMed
    pubmed_analysis_folder = 'analysis/pubmed'
    if not os.path.exists(pubmed_analysis_folder):
        os.makedirs(pubmed_analysis_folder)

    return pubmed_analysis_folder


def load_latest_pubmed_data():
    """Load the most recent PubMed dataset"""
    pubmed_files = [f for f in os.listdir('data') if f.startswith('pubmed_masld_articles') and f.endswith('.csv')]
    if not pubmed_files:
        raise FileNotFoundError("No PubMed data files found. Please run pubmed_data_collection.py first.")

    latest_file = max(pubmed_files, key=lambda x: os.path.getctime(os.path.join('data', x)))
    filepath = os.path.join('data', latest_file)

    print(f"Loading PubMed data from: {latest_file}")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} articles")

    return df, latest_file


def clean_and_preprocess_data(df):
    """Clean and preprocess PubMed data for analysis"""
    # Convert publication date
    df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')

    # Extract year and month for analysis
    df['year'] = df['publication_date'].dt.year
    df['month'] = df['publication_date'].dt.month
    df['year_month'] = df['publication_date'].dt.to_period('M')

    # Create flags for key terms in titles and abstracts
    df['mentions_masld'] = df['title'].str.contains('MASLD|MAFLD', case=False, na=False) | df['abstract'].str.contains(
        'MASLD|MAFLD', case=False, na=False)
    df['mentions_nafld'] = df['title'].str.contains('NAFLD', case=False, na=False) | df['abstract'].str.contains(
        'NAFLD', case=False, na=False)
    df['mentions_nash'] = df['title'].str.contains('NASH', case=False, na=False) | df['abstract'].str.contains('NASH',
                                                                                                               case=False,
                                                                                                               na=False)
    df['mentions_mash'] = df['title'].str.contains('MASH', case=False, na=False) | df['abstract'].str.contains('MASH',
                                                                                                               case=False,
                                                                                                               na=False)

    # Drug mentions
    df['mentions_resmetirom'] = df['title'].str.contains('resmetirom|rezdiffra', case=False, na=False) | df[
        'abstract'].str.contains('resmetirom|rezdiffra', case=False, na=False)
    df['mentions_glp1'] = df['title'].str.contains('semaglutide|ozempic|wegovy|glp-1|glp1', case=False, na=False) | df[
        'abstract'].str.contains('semaglutide|ozempic|wegovy|glp-1|glp1', case=False, na=False)

    # Combined mentions (both disease and drug)
    df['mentions_masld_resmetirom'] = df['mentions_masld'] & df['mentions_resmetirom']
    df['mentions_masld_glp1'] = df['mentions_masld'] & df['mentions_glp1']
    df['mentions_nafld_resmetirom'] = df['mentions_nafld'] & df['mentions_resmetirom']
    df['mentions_nafld_glp1'] = df['mentions_nafld'] & df['mentions_glp1']

    return df


def analyze_publication_trends(df, analysis_folder):
    """Analyze publication trends over time"""
    print("Analyzing publication trends...")

    # Monthly publication count
    monthly_trends = df.groupby('year_month').size()

    # Plot monthly trends
    plt.figure(figsize=(14, 8))
    monthly_trends.plot(kind='line', linewidth=2.5, marker='o', color='#2E86AB')

    # Add FDA approval dates
    plt.axvline(pd.Period('2024-03', 'M'), color='red', linestyle='--', alpha=0.8, linewidth=2,
                label='Resmetirom FDA Approval (Mar 2024)')
    plt.axvline(pd.Period('2025-08', 'M'), color='orange', linestyle='--', alpha=0.8, linewidth=2,
                label='Semaglutide FDA Approval (Aug 2025)')

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
    """Analyze mentions of key terms over time"""
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

    plt.axvline('2024-03', color='red', linestyle='--', alpha=0.7, label='Resmetirom FDA Approval')
    plt.axvline('2025-08', color='orange', linestyle='--', alpha=0.7, label='Semaglutide FDA Approval')

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

    plt.axvline('2024-03', color='red', linestyle='--', alpha=0.7)
    plt.axvline('2025-08', color='orange', linestyle='--', alpha=0.7)

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
    """Analyze adoption of new terminology (MASLD vs NAFLD)"""
    print("Analyzing terminology adoption...")

    # Calculate terminology adoption over time
    terminology_trends = df.groupby('year_month').agg({
        'mentions_masld': 'sum',
        'mentions_nafld': 'sum'
    })

    terminology_trends['masld_ratio'] = terminology_trends['mentions_masld'] / (
                terminology_trends['mentions_masld'] + terminology_trends['mentions_nafld'])
    terminology_trends['total_mentions'] = terminology_trends['mentions_masld'] + terminology_trends['mentions_nafld']

    # Plot terminology adoption
    plt.figure(figsize=(14, 8))

    plt.subplot(1, 2, 1)
    plt.plot(terminology_trends.index.astype(str), terminology_trends['mentions_masld'], label='MASLD', linewidth=2.5,
             marker='o', color='blue')
    plt.plot(terminology_trends.index.astype(str), terminology_trends['mentions_nafld'], label='NAFLD', linewidth=2.5,
             marker='s', color='green')
    plt.axvline('2024-03', color='red', linestyle='--', alpha=0.7)
    plt.axvline('2025-08', color='orange', linestyle='--', alpha=0.7)
    plt.title('MASLD vs NAFLD Terminology Usage', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Number of Mentions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    plt.subplot(1, 2, 2)
    plt.plot(terminology_trends.index.astype(str), terminology_trends['masld_ratio'] * 100, linewidth=2.5, marker='o',
             color='purple')
    plt.axvline('2024-03', color='red', linestyle='--', alpha=0.7, label='Resmetirom FDA Approval')
    plt.axvline('2025-08', color='orange', linestyle='--', alpha=0.7, label='Semaglutide FDA Approval')
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
    """Analyze distribution of publications across journals"""
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
    """Generate comprehensive summary statistics"""
    print("Generating summary statistics...")

    summary_stats = {
        'total_publications': len(df),
        'publications_with_abstracts': df['has_abstract'].sum(),
        'publications_mentioning_masld': df['mentions_masld'].sum(),
        'publications_mentioning_nafld': df['mentions_nafld'].sum(),
        'publications_mentioning_nash': df['mentions_nash'].sum(),
        'publications_mentioning_mash': df['mentions_mash'].sum(),
        'publications_mentioning_resmetirom': df['mentions_resmetirom'].sum(),
        'publications_mentioning_glp1': df['mentions_glp1'].sum(),
        'publications_mentioning_masld_resmetirom': df['mentions_masld_resmetirom'].sum(),
        'publications_mentioning_masld_glp1': df['mentions_masld_glp1'].sum(),
        'unique_journals': df['journal'].nunique(),
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
    """Create a comprehensive analysis report"""
    report_file = f"{analysis_folder}/pubmed_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"

    with open(report_file, 'w') as f:
        f.write("COMPREHENSIVE PUBMED ANALYSIS REPORT - MASLD PROJECT\n")
        f.write("=" * 60 + "\n\n")

        f.write("STUDY OVERVIEW\n")
        f.write("-" * 20 + "\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Data Source: {source_file}\n")
        f.write(f"Study Period: 2023-01-01 to 2025-10-28\n")
        f.write(f"Key FDA Dates: Resmetirom (Mar 14, 2024), Semaglutide (Aug 15, 2025)\n\n")

        f.write("DATA SUMMARY\n")
        f.write("-" * 20 + "\n")
        for key, value in summary_stats.items():
            f.write(f"{key.replace('_', ' ').title()}: {value}\n")

        f.write("\nKEY FINDINGS\n")
        f.write("-" * 20 + "\n")
        f.write(
            f"MASLD was mentioned in {summary_stats['publications_mentioning_masld']} publications ({summary_stats['publications_mentioning_masld'] / summary_stats['total_publications'] * 100:.1f}% of total)\n")
        f.write(
            f"NAFLD was mentioned in {summary_stats['publications_mentioning_nafld']} publications ({summary_stats['publications_mentioning_nafld'] / summary_stats['total_publications'] * 100:.1f}% of total)\n")
        f.write(
            f"Resmetirom was mentioned in {summary_stats['publications_mentioning_resmetirom']} publications ({summary_stats['publications_mentioning_resmetirom'] / summary_stats['total_publications'] * 100:.1f}% of total)\n")
        f.write(
            f"GLP-1 agonists were mentioned in {summary_stats['publications_mentioning_glp1']} publications ({summary_stats['publications_mentioning_glp1'] / summary_stats['total_publications'] * 100:.1f}% of total)\n")
        f.write(
            f"MASLD + Resmetirom combined mentions: {summary_stats['publications_mentioning_masld_resmetirom']} publications\n")
        f.write(
            f"MASLD + GLP-1 combined mentions: {summary_stats['publications_mentioning_masld_glp1']} publications\n")

        f.write(f"\nFILES GENERATED\n")
        f.write("-" * 20 + "\n")
        f.write("Visualizations:\n")
        f.write("  - pubmed_publication_trends.png\n")
        f.write("  - pubmed_term_mentions.png\n")
        f.write("  - pubmed_terminology_adoption.png\n")
        f.write("  - pubmed_top_journals.png\n")
        f.write("\nData Tables:\n")
        f.write("  - pubmed_summary_statistics.csv\n")
        f.write("  - pubmed_yearly_statistics.csv\n")

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
        raise


if __name__ == "__main__":
    main()
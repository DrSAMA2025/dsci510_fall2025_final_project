# reddit_sentiment_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import glob
import os

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def load_data():
    """Load the most recent Reddit dataset"""
    reddit_files = glob.glob('data/reddit_data_*.csv')
    latest_file = max(reddit_files, key=os.path.getctime)
    print(f"Analyzing: {latest_file}")

    df = pd.read_csv(latest_file)
    print(f"Loaded {len(df)} records from {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df, latest_file


def analyze_sentiment(df):
    """Perform sentiment analysis on Reddit data"""
    analyzer = SentimentIntensityAnalyzer()

    # Combine post and comment text for analysis
    df['text_to_analyze'] = df['comment_text'].fillna(df['post_text'])

    # Calculate sentiment scores
    def get_sentiment(text):
        if pd.isna(text):
            return 0.0
        scores = analyzer.polarity_scores(str(text))
        return scores['compound']

    print("Calculating sentiment scores...")
    df['sentiment'] = df['text_to_analyze'].apply(get_sentiment)
    return df


def create_daily_summary(df):
    """Create daily sentiment summary"""
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    df['datetime'] = pd.to_datetime(df['timestamp'])

    daily_summary = df.groupby('date').agg({
        'sentiment': ['mean', 'count', 'std'],
        'post_score': 'mean',
        'comment_score': 'mean'
    }).round(4)

    daily_summary.columns = ['_'.join(col).strip() for col in daily_summary.columns.values]
    daily_summary = daily_summary.reset_index()
    daily_summary['date_dt'] = pd.to_datetime(daily_summary['date'])

    daily_summary = daily_summary.rename(columns={
        'sentiment_mean': 'avg_sentiment',
        'sentiment_count': 'daily_activity',
        'sentiment_std': 'sentiment_std',
        'post_score_mean': 'avg_post_score',
        'comment_score_mean': 'avg_comment_score'
    })

    return daily_summary, df


def detect_key_mentions(df):
    """Detect mentions of key drugs and conditions"""
    key_terms = {
        'resmetirom': ['resmetirom', 'rezdiffra', 'madrigal'],
        'semaglutide': ['semaglutide', 'ozempic', 'wegovy', 'novo nordisk'],
        'masld': ['masld', 'nafld', 'nash', 'mash', 'fatty liver']
    }

    mentions = {}
    mention_details = {}

    for category, terms in key_terms.items():
        pattern = '|'.join(terms)
        mask = df['text_to_analyze'].str.contains(pattern, case=False, na=False)
        mentions[category] = len(df[mask])
        df[f'mentions_{category}'] = mask.astype(int)

        # Get sentiment for mentions
        mention_sentiment = df[mask]['sentiment'].mean()
        mention_details[category] = {
            'count': len(df[mask]),
            'sentiment': mention_sentiment,
            'percentage': len(df[mask]) / len(df) * 100
        }

    return mentions, df, mention_details


def create_plots(daily_summary, df, analysis_folder):
    """Generate analysis plots for presentation"""
    print("Generating plots for presentation...")

    # Plot 1: Daily Sentiment Trend with FDA Dates
    plt.figure(figsize=(14, 7))
    plt.plot(daily_summary['date_dt'], daily_summary['avg_sentiment'],
             linewidth=2.5, alpha=0.8, label='Daily Avg Sentiment', color='#2E86AB')

    # Add FDA approval dates with labels
    plt.axvline(pd.to_datetime('2024-03-14'), color='red', linestyle='--',
                alpha=0.8, linewidth=2, label='Resmetirom FDA Approval (Mar 14, 2024)')
    plt.axvline(pd.to_datetime('2025-08-15'), color='orange', linestyle='--',
                alpha=0.8, linewidth=2, label='Semaglutide FDA Approval (Aug 15, 2025)')

    plt.title('Reddit Sentiment Trends Around MASLD Drug FDA Approvals\n(2023-2025)',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Average Sentiment Score', fontsize=12)
    plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=1)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{analysis_folder}/sentiment_trend_fda_dates.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Sentiment by Subreddit
    plt.figure(figsize=(12, 8))
    subreddit_sentiment = df.groupby('subreddit')['sentiment'].mean().sort_values(ascending=False)
    subreddit_counts = df.groupby('subreddit').size()

    # Only plot subreddits with significant data
    significant_subs = subreddit_counts[subreddit_counts > 30].index
    subreddit_sentiment = subreddit_sentiment[significant_subs]

    colors = ['#4CAF50' if x > 0 else '#F44336' for x in subreddit_sentiment.values]
    bars = plt.bar(range(len(subreddit_sentiment)), subreddit_sentiment.values,
                   color=colors, alpha=0.8, edgecolor='black')

    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, subreddit_counts[significant_subs])):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.3f}\n(n={count})', ha='center', va='bottom', fontsize=9)

    plt.title('Average Sentiment by Subreddit Community', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Subreddit', fontsize=12)
    plt.ylabel('Average Sentiment Score', fontsize=12)
    plt.xticks(range(len(subreddit_sentiment)), subreddit_sentiment.index, rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{analysis_folder}/sentiment_by_subreddit.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 3: Sentiment Distribution
    plt.figure(figsize=(12, 6))
    n, bins, patches = plt.hist(df['sentiment'], bins=50, alpha=0.7,
                                edgecolor='black', color='#2196F3')
    plt.axvline(df['sentiment'].mean(), color='red', linestyle='--', linewidth=2,
                label=f'Overall Mean: {df["sentiment"].mean():.3f}')
    plt.title('Distribution of Reddit Sentiment Scores for MASLD Discussions',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Sentiment Score (-1 = Negative, +1 = Positive)', fontsize=12)
    plt.ylabel('Frequency (Number of Posts/Comments)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{analysis_folder}/sentiment_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Generated 3 presentation plots in {analysis_folder}/")


def create_manuscript_tables(daily_summary, df, mention_details, analysis_folder):
    """Create tables for manuscript/paper"""
    print("Creating tables for manuscript...")

    # Table 1: Overall Statistics
    overall_stats = pd.DataFrame({
        'Metric': [
            'Total Records',
            'Date Range',
            'Overall Sentiment Mean',
            'Overall Sentiment Std',
            'Positive Sentiment (%)',
            'Neutral Sentiment (%)',
            'Negative Sentiment (%)'
        ],
        'Value': [
            f"{len(df):,}",
            f"{daily_summary['date'].min()} to {daily_summary['date'].max()}",
            f"{df['sentiment'].mean():.3f}",
            f"{df['sentiment'].std():.3f}",
            f"{(len(df[df['sentiment'] > 0.05]) / len(df) * 100):.1f}%",
            f"{(len(df[(df['sentiment'] >= -0.05) & (df['sentiment'] <= 0.05)]) / len(df) * 100):.1f}%",
            f"{(len(df[df['sentiment'] < -0.05]) / len(df) * 100):.1f}%"
        ]
    })

    # Table 2: Subreddit Analysis
    subreddit_analysis = df.groupby('subreddit').agg({
        'sentiment': ['count', 'mean', 'std'],
        'post_score': 'mean',
        'comment_score': 'mean'
    }).round(3)
    subreddit_analysis.columns = ['count', 'sentiment_mean', 'sentiment_std', 'avg_post_score', 'avg_comment_score']
    subreddit_analysis = subreddit_analysis.sort_values('count', ascending=False)

    # Table 3: Key Term Mentions
    mention_table = pd.DataFrame([
        {
            'Term Category': category,
            'Mentions': details['count'],
            'Percentage': f"{details['percentage']:.1f}%",
            'Avg Sentiment': f"{details['sentiment']:.3f}"
        }
        for category, details in mention_details.items()
    ])

    # Save tables
    overall_stats.to_csv(f'{analysis_folder}/reddit_table_overall_statistics.csv', index=False)
    subreddit_analysis.to_csv(f'{analysis_folder}/table_subreddit_analysis.csv')
    mention_table.to_csv(f'{analysis_folder}/reddit_table_key_mentions.csv', index=False)

    print(f"Generated 3 manuscript tables in {analysis_folder}/")


def create_comprehensive_report(df, daily_summary, mention_details, source_file, analysis_folder):
    """Create comprehensive report for both presentation and manuscript"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')

    report_file = f"{analysis_folder}/reddit_analysis_comprehensive_report_{timestamp}.txt"

    with open(report_file, 'w') as f:
        f.write("COMPREHENSIVE MASLD REDDIT SENTIMENT ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write("STUDY OVERVIEW\n")
        f.write("-" * 20 + "\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Data Source: {source_file}\n")
        f.write(f"Study Period: 2023-01-01 to 2025-10-28\n")
        f.write(f"Key FDA Dates: Resmetirom (Mar 14, 2024), Semaglutide (Aug 15, 2025)\n\n")

        f.write("DATA SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Records Analyzed: {len(df):,}\n")
        f.write(f"Posts: {len(df[df['type'] == 'post']):,}\n")
        f.write(f"Comments: {len(df[df['type'] == 'comment']):,}\n")
        f.write(f"Date Range in Data: {daily_summary['date'].min()} to {daily_summary['date'].max()}\n")
        f.write(f"Days with Activity: {len(daily_summary):,}\n\n")

        f.write("OVERALL SENTIMENT RESULTS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Mean Sentiment: {df['sentiment'].mean():.3f}\n")
        f.write(f"Standard Deviation: {df['sentiment'].std():.3f}\n")

        positive = len(df[df['sentiment'] > 0.05])
        neutral = len(df[(df['sentiment'] >= -0.05) & (df['sentiment'] <= 0.05)])
        negative = len(df[df['sentiment'] < -0.05])

        f.write(f"\nSentiment Distribution:\n")
        f.write(f"  Positive (>0.05): {positive:,} ({positive / len(df) * 100:.1f}%)\n")
        f.write(f"  Neutral (-0.05 to 0.05): {neutral:,} ({neutral / len(df) * 100:.1f}%)\n")
        f.write(f"  Negative (<-0.05): {negative:,} ({negative / len(df) * 100:.1f}%)\n\n")

        f.write("KEY TERM MENTION ANALYSIS\n")
        f.write("-" * 30 + "\n")
        for category, details in mention_details.items():
            f.write(f"{category.upper()}:\n")
            f.write(f"  Mentions: {details['count']:,} ({details['percentage']:.1f}% of total)\n")
            f.write(f"  Average Sentiment: {details['sentiment']:.3f}\n\n")

        f.write("SUBREPDDIT ANALYSIS\n")
        f.write("-" * 20 + "\n")
        subreddit_summary = df.groupby('subreddit').agg({
            'sentiment': ['count', 'mean']
        }).round(3)
        subreddit_summary.columns = ['count', 'sentiment_mean']
        subreddit_summary = subreddit_summary.sort_values('count', ascending=False)

        for subreddit, row in subreddit_summary.iterrows():
            f.write(f"r/{subreddit}: {row['count']:,} records, sentiment: {row['sentiment_mean']:.3f}\n")

        f.write(f"\nFILES GENERATED\n")
        f.write("-" * 20 + "\n")
        f.write("Visualizations (for presentation):\n")
        f.write("  - sentiment_trend_fda_dates.png\n")
        f.write("  - sentiment_by_subreddit.png\n")
        f.write("  - sentiment_distribution.png\n")
        f.write("\nTables (for manuscript):\n")
        f.write("  - reddit_table_overall_statistics.csv\n")
        f.write("  - table_subreddit_analysis.csv\n")
        f.write("  - reddit_table_key_mentions.csv\n")
        f.write("\nData Files:\n")
        f.write("  - reddit_sentiment_daily_*.csv\n")
        f.write("  - reddit_data_with_sentiment_*.csv\n")

    return report_file


def main():
    print("=== MASLD REDDIT SENTIMENT ANALYSIS ===")
    print("Generating outputs for BOTH presentation and manuscript...")

    # Create analysis folder
    analysis_folder = 'analysis'
    if not os.path.exists(analysis_folder):
        os.makedirs(analysis_folder)

    # Load and analyze data
    df, source_file = load_data()
    df = analyze_sentiment(df)
    daily_summary, df = create_daily_summary(df)
    mentions, df, mention_details = detect_key_mentions(df)

    # Generate outputs for presentation
    create_plots(daily_summary, df, analysis_folder)

    # Generate outputs for manuscript
    create_manuscript_tables(daily_summary, df, mention_details, analysis_folder)

    # Save data files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    daily_summary.to_csv(f'{analysis_folder}/reddit_sentiment_daily_{timestamp}.csv', index=False)
    df.to_csv(f'{analysis_folder}/reddit_data_with_sentiment_{timestamp}.csv', index=False)

    # Create comprehensive report
    report_file = create_comprehensive_report(df, daily_summary, mention_details, source_file, analysis_folder)

    # Final summary
    print(f"\nANALYSIS COMPLETE - READY FOR PUBLICATION & PRESENTATION")
    print(f"For Presentation: 3 professional plots generated")
    print(f"For Manuscript: 3 detailed tables generated")
    print(f"Overall Sentiment: {df['sentiment'].mean():.3f}")
    print(f"All outputs saved to: /{analysis_folder}/")
    print(f"Comprehensive report: {report_file}")

    return df, daily_summary


if __name__ == "__main__":
    df, daily_summary = main()
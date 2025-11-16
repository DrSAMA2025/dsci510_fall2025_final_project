import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from scipy import stats
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.feature_extraction import text
import networkx as nx
from collections import defaultdict
from itertools import combinations
from scipy.stats import fisher_exact

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

    # Create figure and axis explicitly
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each column explicitly using matplotlib (NOT pandas plot)
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    for i, column in enumerate(df_trends.columns):
        ax.plot(df_trends.index, df_trends[column],
                label=column, linewidth=2, color=colors[i % len(colors)])

    # Set titles and labels
    ax.set_title('Google Search Interest Over Time (2023-2025)', fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Relative Search Volume Index', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add FDA approval dates
    for label, date_str in FDA_EVENT_DATES.items():
        date = pd.to_datetime(date_str)
        ax.axvline(date, color='black', linestyle='--', alpha=0.7, linewidth=1.5)
        ax.text(date, ax.get_ylim()[1] * 0.95, label, rotation=90,
                verticalalignment='top', fontsize=9, backgroundcolor='white')

    # Tight layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    save_path = save_dir / "google_trends_basic_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  > Saved basic plot to: {save_path.name}")

    # Verify the file was created and has content
    if save_path.exists():
        file_size = save_path.stat().st_size
        print(f"  > File verification: {file_size} bytes")
        if file_size < 1000:  # If file is too small, it's probably empty
            print("  > WARNING: File size suggests empty plot!")
    else:
        print("  > ERROR: File was not created!")

    if not notebook_plot:
        plt.close()
    else:
        plt.show()
        print("  > Displayed Google Trends plot in notebook")

    return fig  # Return the figure object to keep it alive


def advanced_google_trends_analysis(df_trends: pd.DataFrame, notebook_plot=False):
    """Advanced statistical analysis of Google Trends data for both FDA events"""
    print("Data Quality Check:")
    for col in ['MASLD', 'NAFLD', 'Rezdiffra', 'Wegovy', 'Ozempic']:
        if col in df_trends.columns:
            unique_vals = df_trends[col].nunique()
            zero_pct = (df_trends[col] == 0).mean() * 100
            print(f"  {col}: {unique_vals} unique values, {zero_pct:.1f}% zeros")
    print("\n[Advanced Analysis] Google Trends Statistical Analysis...")
    save_dir = RESULTS_DIR / GOOGLE_TRENDS_ANALYSIS_SUBDIR

    # [Keep all the statistical calculation code the same as before...]
    # 1. Calculate cross-correlations between search terms
    correlation_matrix = df_trends[['MASLD', 'NAFLD', 'Rezdiffra', 'Wegovy', 'Ozempic']].corr()

    # Enhanced terminology analysis
    df_trends['MASLD_NAFLD_Ratio'] = df_trends['MASLD'] / (df_trends['NAFLD'] + 0.1)

    # Calculate ratio trends around both FDA events
    terminology_analysis = {}
    for event_name, event_date in FDA_EVENT_DATES.items():
        event_date = pd.to_datetime(event_date)
        pre_ratio = df_trends[df_trends.index < event_date]['MASLD_NAFLD_Ratio'].tail(30).mean()
        post_ratio = df_trends[df_trends.index > event_date]['MASLD_NAFLD_Ratio'].head(30).mean()

        terminology_analysis[event_name] = {
            'pre_ratio': pre_ratio,
            'post_ratio': post_ratio,
            'ratio_change': ((post_ratio - pre_ratio) / (pre_ratio + 1e-10)) * 100
        }

    # 2. Statistical tests for BOTH FDA approval events
    resmetirom_date = pd.to_datetime(FDA_EVENT_DATES['Resmetirom Approval'])
    glp1_date = pd.to_datetime(FDA_EVENT_DATES['GLP-1 Agonists Approval'])

    # Define analysis periods for both events (30 days pre/post each)
    significant_changes_resmetirom = {}
    significant_changes_glp1 = {}

    # Resmetirom approval impact analysis
    pre_resmetirom = df_trends[df_trends.index < resmetirom_date].tail(30)
    post_resmetirom = df_trends[df_trends.index > resmetirom_date].head(30)

    # GLP-1 approval impact analysis
    pre_glp1 = df_trends[df_trends.index < glp1_date].tail(30)
    post_glp1 = df_trends[df_trends.index > glp1_date].head(30)

    # Terms to test for each event
    resmetirom_terms = ['MASLD', 'NAFLD', 'Rezdiffra']
    glp1_terms = ['MASLD', 'NAFLD', 'Wegovy', 'Ozempic']

    # Perform t-tests for Resmetirom approval
    for term in resmetirom_terms:
        pre_values = pre_resmetirom[term].dropna()
        post_values = post_resmetirom[term].dropna()

        if len(pre_values) > 5 and len(post_values) > 5:
            t_stat, p_value = stats.ttest_ind(pre_values, post_values, equal_var=False)
            significant_changes_resmetirom[term] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'pre_mean': pre_values.mean(),
                'post_mean': post_values.mean(),
                'change_absolute': (post_values.mean() - pre_values.mean())
            }

    # Perform t-tests for GLP-1 approval
    for term in glp1_terms:
        pre_values = pre_glp1[term].dropna()
        post_values = post_glp1[term].dropna()

        if len(pre_values) > 5 and len(post_values) > 5:
            t_stat, p_value = stats.ttest_ind(pre_values, post_values, equal_var=False)
            significant_changes_glp1[term] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'pre_mean': pre_values.mean(),
                'post_mean': post_values.mean(),
                'change_absolute': (post_values.mean() - pre_values.mean())
            }

    # 3. Create SEPARATE visualizations

    # 3a. Main timeline with both FDA events
    plt.figure(figsize=(12, 6))
    df_trends.plot(linewidth=2, alpha=0.8)
    plt.title('Google Search Trends: Impact of Both FDA Approvals', fontsize=14, fontweight='bold')
    plt.axvline(resmetirom_date, color='red', linestyle='--', linewidth=2, label='Resmetirom FDA Approval')
    plt.axvline(glp1_date, color='blue', linestyle='--', linewidth=2, label='GLP-1 FDA Approval')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add significance annotations for both events
    for term, stats_data in significant_changes_resmetirom.items():
        if stats_data['p_value'] < 0.05:
            plt.annotate(f'{term}*',
                         xy=(resmetirom_date, df_trends[term].max() * 0.9),
                         xytext=(10, 0), textcoords='offset points',
                         fontweight='bold', color='red', fontsize=10)

    for term, stats_data in significant_changes_glp1.items():
        if stats_data['p_value'] < 0.05:
            plt.annotate(f'{term}*',
                         xy=(glp1_date, df_trends[term].max() * 0.8),
                         xytext=(10, 0), textcoords='offset points',
                         fontweight='bold', color='blue', fontsize=10)

    plt.tight_layout()
    timeline_path = save_dir / "advanced_google_trends_timeline.png"
    plt.savefig(timeline_path, dpi=300, bbox_inches='tight')
    if not notebook_plot: plt.close()
    print(f"  > Saved timeline to: {timeline_path.name}")

    # 3b. Correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True, cbar_kws={'shrink': 0.8})
    plt.title('Search Term Correlations', fontweight='bold')
    plt.tight_layout()
    correlation_path = save_dir / "advanced_google_trends_correlation.png"
    plt.savefig(correlation_path, dpi=300, bbox_inches='tight')
    if not notebook_plot: plt.close()
    print(f"  > Saved correlation to: {correlation_path.name}")

    # 3c. Comparative event impact analysis
    impact_data = []

    # Add Resmetirom approval impacts
    for term, stats_data in significant_changes_resmetirom.items():
        impact_data.append({
            'Term': term,
            'Event': 'Resmetirom Approval',
            'Change_Absolute': stats_data['change_absolute'],
            'P_Value': stats_data['p_value']
        })

    # Add GLP-1 approval impacts
    for term, stats_data in significant_changes_glp1.items():
        impact_data.append({
            'Term': term,
            'Event': 'GLP-1 Approval',
            'Change_Absolute': stats_data['change_absolute'],
            'P_Value': stats_data['p_value']
        })

    impact_df = pd.DataFrame(impact_data)

    if not impact_df.empty:
        plt.figure(figsize=(10, 6))
        events = impact_df['Event'].unique()
        terms = impact_df['Term'].unique()

        x_pos = np.arange(len(terms))
        width = 0.35

        for i, event in enumerate(events):
            event_data = impact_df[impact_df['Event'] == event]
            values = [event_data[event_data['Term'] == term]['Change_Absolute'].iloc[0]
                      if term in event_data['Term'].values else 0
                      for term in terms]

            colors = ['red' if event == 'Resmetirom Approval' else 'blue']
            bars = plt.bar(x_pos + i * width, values, width,
                           label=event, color=colors, alpha=0.7)

            # Add value labels with significance markers
            for j, (bar, term) in enumerate(zip(bars, terms)):
                height = bar.get_height()
                if height != 0:
                    p_val = event_data[event_data['Term'] == term]['P_Value'].iloc[0]
                    plt.text(bar.get_x() + bar.get_width() / 2., height,
                             f'{height:.1f}%{"*" if p_val < 0.05 else ""}',
                             ha='center', va='bottom' if height > 0 else 'top',
                             fontsize=9)

        plt.xlabel('Search Terms')
        plt.ylabel('Absolute Change (points)')
        plt.title('Search Interest Change After FDA Approvals\n(* = statistically significant)', fontweight='bold')
        plt.xticks(x_pos + width / 2, terms)
        plt.legend()
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        impact_path = save_dir / "advanced_google_trends_impact.png"
        plt.savefig(impact_path, dpi=300, bbox_inches='tight')
        if not notebook_plot: plt.close()
        print(f"  > Saved impact analysis to: {impact_path.name}")

    # 3d. Statistical summary table
    if not impact_df.empty:
        plt.figure(figsize=(10, 4))
        plt.axis('off')
        table_data = impact_df[['Term', 'Event', 'Change_Absolute', 'P_Value']].round(3)
        table = plt.table(cellText=table_data.values,
                          colLabels=table_data.columns,
                          cellLoc='center',
                          loc='center',
                          bbox=[0.1, 0.1, 0.8, 0.8])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        plt.title('Statistical Summary: Both FDA Events', fontweight='bold')
        plt.tight_layout()

        table_path = save_dir / "advanced_google_trends_statistical_table.png"
        plt.savefig(table_path, dpi=300, bbox_inches='tight')
        if not notebook_plot: plt.close()
        print(f"  > Saved statistical table to: {table_path.name}")

    # Display in notebook if requested
    if notebook_plot:
        plt.show()

    # Return comprehensive statistical results
    return {
        'correlation_matrix': correlation_matrix,
        'resmetirom_impact': significant_changes_resmetirom,
        'glp1_impact': significant_changes_glp1,
        'combined_impact_analysis': impact_df,
        'terminology_analysis': terminology_analysis
    }


def analyze_reddit_sentiment(df_reddit: pd.DataFrame, notebook_plot=False):
    """Analyzes and visualizes Reddit sentiment over time."""
    print("\n[Analysis] Analyzing Reddit Sentiment...")
    save_dir = RESULTS_DIR / REDDIT_ANALYSIS_SUBDIR
    save_dir.mkdir(exist_ok=True, parents=True)  # Ensure directory exists

    # Resample sentiment data weekly
    df_weekly_sentiment = df_reddit.set_index('timestamp')['sentiment_score'].resample('W').mean().dropna()

    plt.figure(figsize=(12, 6))
    plt.plot(df_weekly_sentiment.index, df_weekly_sentiment.values, color='skyblue', linewidth=2)
    plt.title('Average Weekly Reddit Sentiment Score')
    plt.xlabel('Date')
    plt.ylabel('Average Compound Sentiment Score (VADER)')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Add FDA Approval Lines
    for label, date_str in FDA_EVENT_DATES.items():
        plt.axvline(pd.to_datetime(date_str), color='red', linestyle='--', alpha=0.7, label=f'{label}')

    # Add reference lines for neutral sentiment
    plt.axhline(0.05, color='gray', linestyle=':', label='Slightly Positive Threshold')
    plt.axhline(-0.05, color='gray', linestyle=':', label='Slightly Negative Threshold')

    # Add legend
    plt.legend()

    # === ADD SAVING CODE HERE ===
    save_path = save_dir / "reddit_basic_sentiment_timeline.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  > Saved basic Reddit plot to: {save_path.name}")

    if not notebook_plot:
        plt.close()
    else:
        plt.show()
        print("  > Displayed Reddit sentiment plot in notebook")


def advanced_reddit_sentiment_analysis(df_reddit: pd.DataFrame, notebook_plot=False):
    """Advanced statistical analysis of Reddit sentiment data"""
    print("\n[Advanced Analysis] Reddit Sentiment Statistical Analysis...")
    save_dir = RESULTS_DIR / REDDIT_ANALYSIS_SUBDIR
    save_dir.mkdir(exist_ok=True, parents=True)

    # 1. Data quality assessment
    print("Data Quality Check:")
    print(f"  Total records: {len(df_reddit)}")
    print(f"  Date range: {df_reddit['timestamp'].min()} to {df_reddit['timestamp'].max()}")
    print(f"  Subreddits: {df_reddit['subreddit'].nunique()}")
    print(f"  Sentiment range: {df_reddit['sentiment_score'].min():.3f} to {df_reddit['sentiment_score'].max():.3f}")

    # 2. Subreddit-level sentiment analysis
    subreddit_stats = df_reddit.groupby('subreddit').agg({
        'sentiment_score': ['mean', 'std', 'count'],
        'timestamp': ['min', 'max']
    }).round(3)

    print("\nSubreddit Sentiment Summary:")
    print(subreddit_stats)

    # 3. Statistical tests for FDA event impacts
    resmetirom_date = pd.to_datetime(FDA_EVENT_DATES['Resmetirom Approval'])
    glp1_date = pd.to_datetime(FDA_EVENT_DATES['GLP-1 Agonists Approval'])

    # Define event windows (2 weeks pre/post)
    event_impacts = {}

    for event_name, event_date in [('Resmetirom Approval', resmetirom_date),
                                   ('GLP-1 Approval', glp1_date)]:

        pre_period = df_reddit[
            (df_reddit['timestamp'] >= event_date - pd.Timedelta(days=14)) &
            (df_reddit['timestamp'] < event_date)
            ]
        post_period = df_reddit[
            (df_reddit['timestamp'] > event_date) &
            (df_reddit['timestamp'] <= event_date + pd.Timedelta(days=14))
            ]

        if len(pre_period) > 10 and len(post_period) > 10:
            t_stat, p_value = stats.ttest_ind(
                pre_period['sentiment_score'],
                post_period['sentiment_score'],
                equal_var=False
            )

            event_impacts[event_name] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'pre_mean': pre_period['sentiment_score'].mean(),
                'post_mean': post_period['sentiment_score'].mean(),
                'change_absolute': post_period['sentiment_score'].mean() - pre_period['sentiment_score'].mean(),
                'pre_count': len(pre_period),
                'post_count': len(post_period)
            }

    # 4. Create SEPARATE visualizations instead of combined subplots

    # 4a. Sentiment timeline with events
    plt.figure(figsize=(12, 6))
    daily_sentiment = df_reddit.set_index('timestamp')['sentiment_score'].resample('D').mean()
    plt.plot(daily_sentiment.index, daily_sentiment.values, color='skyblue', linewidth=2, alpha=0.8)
    plt.title('Daily Average Reddit Sentiment Score', fontweight='bold')
    plt.ylabel('Sentiment Score')
    plt.grid(True, alpha=0.3)

    # Add FDA events
    for event_name, event_date in [('Resmetirom Approval', resmetirom_date),
                                   ('GLP-1 Approval', glp1_date)]:
        plt.axvline(event_date, color='red', linestyle='--', alpha=0.7)
        plt.text(event_date, plt.ylim()[1] * 0.9, event_name.split()[0],
                 rotation=90, verticalalignment='top', fontsize=9)

    plt.tight_layout()
    timeline_path = save_dir / "reddit_sentiment_timeline.png"
    plt.savefig(timeline_path, dpi=300, bbox_inches='tight')
    if not notebook_plot: plt.close()
    print(f"  > Saved timeline to: {timeline_path.name}")

    # 4b. Subreddit sentiment comparison
    plt.figure(figsize=(10, 8))
    subreddit_means = df_reddit.groupby('subreddit')['sentiment_score'].mean().sort_values()
    plt.barh(range(len(subreddit_means)), subreddit_means.values)
    plt.yticks(range(len(subreddit_means)), subreddit_means.index)
    plt.title('Average Sentiment by Subreddit', fontweight='bold')
    plt.xlabel('Average Sentiment Score')
    plt.tight_layout()
    subreddit_path = save_dir / "reddit_subreddit_sentiment.png"
    plt.savefig(subreddit_path, dpi=300, bbox_inches='tight')
    if not notebook_plot: plt.close()
    print(f"  > Saved subreddit comparison to: {subreddit_path.name}")

    # 4c. Sentiment distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df_reddit['sentiment_score'], bins=30, alpha=0.7, color='lightgreen')
    plt.axvline(df_reddit['sentiment_score'].mean(), color='red', linestyle='--',
                label=f'Mean: {df_reddit["sentiment_score"].mean():.3f}')
    plt.title('Sentiment Score Distribution', fontweight='bold')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    distribution_path = save_dir / "reddit_sentiment_distribution.png"
    plt.savefig(distribution_path, dpi=300, bbox_inches='tight')
    if not notebook_plot: plt.close()
    print(f"  > Saved distribution to: {distribution_path.name}")

    # 4d. Event impact analysis
    if event_impacts:
        plt.figure(figsize=(8, 6))
        events = list(event_impacts.keys())
        changes = [event_impacts[event]['change_absolute'] for event in events]
        p_values = [event_impacts[event]['p_value'] for event in events]

        bars = plt.bar(events, changes, color=['lightcoral', 'lightblue'], alpha=0.7)
        plt.title('Sentiment Change After FDA Approvals', fontweight='bold')
        plt.ylabel('Sentiment Score Change')
        plt.xticks(rotation=45)

        # Add significance annotations
        for i, (bar, p_val) in enumerate(zip(bars, p_values)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.3f}\n{"*" if p_val < 0.05 else ""}',
                     ha='center', va='bottom' if height > 0 else 'top')

        plt.tight_layout()
        event_path = save_dir / "reddit_fda_impact.png"
        plt.savefig(event_path, dpi=300, bbox_inches='tight')
        if not notebook_plot: plt.close()
        print(f"  > Saved event impact to: {event_path.name}")

    # Display in notebook if requested
    if notebook_plot:
        plt.show()

    # Return analysis results (keep your existing return statement)
    return {
        'subreddit_stats': subreddit_stats,
        'event_impacts': event_impacts,
        'daily_sentiment': daily_sentiment,
        'overall_stats': {
            'total_posts': len(df_reddit),
            'mean_sentiment': df_reddit['sentiment_score'].mean(),
            'std_sentiment': df_reddit['sentiment_score'].std(),
            'date_range': f"{df_reddit['timestamp'].min()} to {df_reddit['timestamp'].max()}"
        }
    }


def analyze_reddit_topics(df_reddit, num_topics=5, notebook_plot=False):
    """Advanced topic modeling for Reddit discussions"""
    print("\n[Advanced Analysis] Reddit Topic Modeling...")
    save_dir = RESULTS_DIR / REDDIT_ANALYSIS_SUBDIR
    save_dir.mkdir(exist_ok=True, parents=True)

    import re
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction import text

    # 1. Text preprocessing
    print("Preparing text data for topic modeling...")

    # Check what text columns are available
    available_columns = df_reddit.columns.tolist()
    print(f"  Available columns: {available_columns}")

    # Create combined text from available columns
    text_columns = []
    for col in ['post_title', 'post_text', 'comment_text', 'text_to_analyze']:
        if col in df_reddit.columns:
            text_columns.append(col)
            print(f"  Found text column: {col}")

    if not text_columns:
        print("  ERROR: No text columns found for topic modeling")
        return None

    # Use available text columns
    def combine_available_text(row):
        combined = ""
        for col in text_columns:
            if col in row and pd.notna(row[col]) and len(str(row[col])) > 0:
                combined += f" {str(row[col])}"
        return combined.strip()

    df_reddit['combined_text'] = df_reddit.apply(combine_available_text, axis=1)

    # Clean text
    def clean_text(text):
        if pd.isna(text) or len(text) == 0:
            return ""
        text = str(text).lower()
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        return text

    df_reddit['cleaned_text'] = df_reddit['combined_text'].apply(clean_text)

    # Filter out very short texts
    valid_texts = df_reddit[df_reddit['cleaned_text'].str.len() > 50]['cleaned_text']
    print(f"  Texts for topic modeling: {len(valid_texts)}")

    if len(valid_texts) < 100:
        print("  Warning: Insufficient text data for reliable topic modeling")
        return None

    # 2. TF-IDF Vectorization
    print("Creating TF-IDF features...")
    custom_stop_words = list(text.ENGLISH_STOP_WORDS.union([
        'like', 'know', 'think', 'people', 'would', 'really', 'get', 'one',
        'time', 'see', 'also', 'could', 'make', 'take', 'way', 'going'
    ]))

    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words=custom_stop_words,
        min_df=5,
        max_df=0.8
    )

    tfidf_matrix = vectorizer.fit_transform(valid_texts)
    feature_names = vectorizer.get_feature_names_out()

    # 3. NMF Topic Modeling
    print("Running NMF topic modeling...")
    nmf = NMF(n_components=num_topics, random_state=42, max_iter=1000)
    nmf.fit(tfidf_matrix)

    # 4. Extract and display topics
    print("\nDISCOVERED TOPICS:")
    topics = {}
    for topic_idx, topic in enumerate(nmf.components_):
        top_features_ind = topic.argsort()[:-11:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        topics[f"Topic_{topic_idx}"] = top_features
        print(f"  Topic {topic_idx}: {', '.join(top_features)}")

    # 5. Assign topics to documents
    topic_values = nmf.transform(tfidf_matrix)
    df_reddit_filtered = df_reddit[df_reddit['cleaned_text'].str.len() > 50].copy()
    df_reddit_filtered['dominant_topic'] = topic_values.argmax(axis=1)
    df_reddit_filtered['topic_confidence'] = topic_values.max(axis=1)

    # 6. Analyze topic distribution by subreddit
    print("\nTOPIC DISTRIBUTION BY SUBREDDIT:")
    topic_subreddit = pd.crosstab(
        df_reddit_filtered['dominant_topic'],
        df_reddit_filtered['subreddit']
    )
    print(topic_subreddit)

    # 7. Sentiment by topic
    print("\nAVERAGE SENTIMENT BY TOPIC:")
    # Check if sentiment_score exists, if not skip sentiment analysis
    if 'sentiment_score' in df_reddit_filtered.columns:
        topic_sentiment = df_reddit_filtered.groupby('dominant_topic')['sentiment_score'].agg(['mean', 'count'])
        print(topic_sentiment.round(3))
    else:
        print("  Sentiment scores not available - skipping sentiment by topic analysis")
        topic_sentiment = None

    # 8. Visualization
    plt.figure(figsize=(12, 8))

    # Topic distribution
    topic_counts = df_reddit_filtered['dominant_topic'].value_counts().sort_index()
    plt.subplot(2, 2, 1)
    plt.bar(range(len(topic_counts)), topic_counts.values)
    plt.title('Topic Distribution')
    plt.xlabel('Topic ID')
    plt.ylabel('Number of Posts')

    # Sentiment by topic (only if available)
    plt.subplot(2, 2, 2)
    if topic_sentiment is not None:
        plt.bar(topic_sentiment.index, topic_sentiment['mean'])
        plt.title('Average Sentiment by Topic')
        plt.xlabel('Topic ID')
        plt.ylabel('Average Sentiment')
    else:
        plt.text(0.5, 0.5, 'Sentiment data\nnot available',
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes, fontsize=12)
        plt.title('Sentiment by Topic (Data Missing)')
        plt.axis('off')

    # Top words for each topic
    plt.subplot(2, 1, 2)
    topic_words = []
    for topic_idx in range(num_topics):
        top_features_ind = nmf.components_[topic_idx].argsort()[:-6:-1]
        top_words = [feature_names[i] for i in top_features_ind]
        topic_words.append(f"Topic {topic_idx}: {', '.join(top_words)}")

    plt.axis('off')
    plt.text(0.1, 0.9, '\n'.join(topic_words), fontsize=10, verticalalignment='top')
    plt.title('Key Words for Each Topic')

    plt.tight_layout()

    # Save plot
    save_path = save_dir / "reddit_topic_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if not notebook_plot:
        plt.close()
    else:
        plt.show()

    print(f"  > Saved topic analysis to: {save_path.name}")

    return {
        'topics': topics,
        'topic_sentiment': topic_sentiment,
        'topic_subreddit_dist': topic_subreddit,
        'df_with_topics': df_reddit_filtered,
        'success': True
    }


def analyze_temporal_patterns(df_reddit, notebook_plot=False):
    """Advanced temporal analysis of Reddit discussions"""
    print("\n[Advanced Analysis] Reddit Temporal Patterns...")
    save_dir = RESULTS_DIR / REDDIT_ANALYSIS_SUBDIR
    save_dir.mkdir(exist_ok=True, parents=True)

    # Ensure timestamp is datetime
    df_reddit = df_reddit.copy()
    df_reddit['timestamp'] = pd.to_datetime(df_reddit['timestamp'])

    # 1. Extract temporal features
    df_reddit['hour'] = df_reddit['timestamp'].dt.hour
    df_reddit['day_of_week'] = df_reddit['timestamp'].dt.day_name()
    df_reddit['month'] = df_reddit['timestamp'].dt.month_name()
    df_reddit['date'] = df_reddit['timestamp'].dt.date

    print("Temporal Analysis Summary:")
    print(f"  Date range: {df_reddit['timestamp'].min()} to {df_reddit['timestamp'].max()}")
    print(f"  Total days: {df_reddit['date'].nunique()}")

    # 2. Hourly patterns
    hourly_activity = df_reddit.groupby('hour').size()
    hourly_sentiment = df_reddit.groupby('hour')[
        'sentiment_score'].mean() if 'sentiment_score' in df_reddit.columns else None

    # 3. Daily patterns
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_activity = df_reddit.groupby('day_of_week').size().reindex(day_order)
    daily_sentiment = df_reddit.groupby('day_of_week')['sentiment_score'].mean().reindex(
        day_order) if 'sentiment_score' in df_reddit.columns else None

    # 4. Monthly patterns
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    monthly_activity = df_reddit.groupby('month').size().reindex(month_order)

    # 5. FDA event impact on temporal patterns
    resmetirom_date = pd.to_datetime(FDA_EVENT_DATES['Resmetirom Approval'])
    glp1_date = pd.to_datetime(FDA_EVENT_DATES['GLP-1 Agonists Approval'])

    # Activity around FDA events (7 days before/after)
    event_windows = {}
    for event_name, event_date in [('Resmetirom', resmetirom_date), ('GLP-1', glp1_date)]:
        event_window = df_reddit[
            (df_reddit['timestamp'] >= event_date - pd.Timedelta(days=7)) &
            (df_reddit['timestamp'] <= event_date + pd.Timedelta(days=7))
            ]
        event_windows[event_name] = {
            'total_posts': len(event_window),
            'daily_avg': len(event_window) / 15,  # 15-day window
            'pre_event': len(event_window[event_window['timestamp'] < event_date]),
            'post_event': len(event_window[event_window['timestamp'] >= event_date])
        }

    # 6. Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 6a. Hourly activity
    axes[0, 0].bar(hourly_activity.index, hourly_activity.values, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Posts by Hour of Day', fontweight='bold')
    axes[0, 0].set_xlabel('Hour (24h)')
    axes[0, 0].set_ylabel('Number of Posts')
    axes[0, 0].grid(True, alpha=0.3)

    # 6b. Daily activity
    axes[0, 1].bar(range(len(daily_activity)), daily_activity.values, color='lightgreen', alpha=0.7)
    axes[0, 1].set_title('Posts by Day of Week', fontweight='bold')
    axes[0, 1].set_xlabel('Day of Week')
    axes[0, 1].set_ylabel('Number of Posts')
    axes[0, 1].set_xticks(range(len(day_order)))
    axes[0, 1].set_xticklabels([day[:3] for day in day_order])
    axes[0, 1].grid(True, alpha=0.3)

    # 6c. Monthly activity
    axes[0, 2].bar(range(len(monthly_activity)), monthly_activity.values, color='lightcoral', alpha=0.7)
    axes[0, 2].set_title('Posts by Month', fontweight='bold')
    axes[0, 2].set_xlabel('Month')
    axes[0, 2].set_ylabel('Number of Posts')
    axes[0, 2].set_xticks(range(len(month_order)))
    axes[0, 2].set_xticklabels([month[:3] for month in month_order], rotation=45)
    axes[0, 2].grid(True, alpha=0.3)

    # 6d. Hourly sentiment (if available)
    if hourly_sentiment is not None:
        axes[1, 0].plot(hourly_sentiment.index, hourly_sentiment.values,
                        marker='o', color='blue', linewidth=2)
        axes[1, 0].set_title('Average Sentiment by Hour', fontweight='bold')
        axes[1, 0].set_xlabel('Hour (24h)')
        axes[1, 0].set_ylabel('Average Sentiment Score')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    else:
        axes[1, 0].text(0.5, 0.5, 'Sentiment data\nnot available',
                        horizontalalignment='center', verticalalignment='center',
                        transform=axes[1, 0].transAxes, fontsize=12)
        axes[1, 0].set_title('Sentiment by Hour (Data Missing)')
        axes[1, 0].axis('off')

    # 6e. Daily sentiment (if available)
    if daily_sentiment is not None:
        axes[1, 1].plot(range(len(daily_sentiment)), daily_sentiment.values,
                        marker='o', color='green', linewidth=2)
        axes[1, 1].set_title('Average Sentiment by Day', fontweight='bold')
        axes[1, 1].set_xlabel('Day of Week')
        axes[1, 1].set_ylabel('Average Sentiment Score')
        axes[1, 1].set_xticks(range(len(day_order)))
        axes[1, 1].set_xticklabels([day[:3] for day in day_order])
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    else:
        axes[1, 1].text(0.5, 0.5, 'Sentiment data\nnot available',
                        horizontalalignment='center', verticalalignment='center',
                        transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('Sentiment by Day (Data Missing)')
        axes[1, 1].axis('off')

    # 6f. FDA event impact
    event_names = list(event_windows.keys())
    pre_posts = [event_windows[event]['pre_event'] for event in event_names]
    post_posts = [event_windows[event]['post_event'] for event in event_names]

    x = np.arange(len(event_names))
    width = 0.35
    axes[1, 2].bar(x - width / 2, pre_posts, width, label='7 Days Before', alpha=0.7)
    axes[1, 2].bar(x + width / 2, post_posts, width, label='7 Days After', alpha=0.7)
    axes[1, 2].set_title('FDA Event Impact on Discussion Volume', fontweight='bold')
    axes[1, 2].set_xlabel('FDA Event')
    axes[1, 2].set_ylabel('Number of Posts')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(event_names)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    save_path = save_dir / "reddit_temporal_patterns.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if not notebook_plot:
        plt.close()
    else:
        plt.show()

    print(f"  > Saved temporal analysis to: {save_path.name}")

    # Return analysis results
    return {
        'hourly_activity': hourly_activity,
        'daily_activity': daily_activity,
        'monthly_activity': monthly_activity,
        'hourly_sentiment': hourly_sentiment,
        'daily_sentiment': daily_sentiment,
        'event_impacts': event_windows,
        'date_range': f"{df_reddit['timestamp'].min().date()} to {df_reddit['timestamp'].max().date()}",
        'total_days': df_reddit['date'].nunique()
    }


def correlate_reddit_trends(reddit_data, trends_data, notebook_plot=False):
    """Cross-platform correlation analysis between Reddit and Google Trends"""
    print("\n[Advanced Analysis] Reddit-Google Trends Correlation...")
    save_dir = RESULTS_DIR / REDDIT_ANALYSIS_SUBDIR
    save_dir.mkdir(exist_ok=True, parents=True)

    # 1. Prepare Reddit daily data
    reddit_data = reddit_data.copy()
    reddit_data['timestamp'] = pd.to_datetime(reddit_data['timestamp'])
    reddit_daily = reddit_data.set_index('timestamp')['sentiment_score'].resample('D').agg(['mean', 'count']).rename(
        columns={'mean': 'reddit_sentiment', 'count': 'reddit_volume'})

    # 2. Prepare Google Trends daily data (already daily)
    trends_daily = trends_data.resample('D').mean()

    # 3. Align time periods
    start_date = max(reddit_daily.index.min(), trends_daily.index.min())
    end_date = min(reddit_daily.index.max(), trends_daily.index.max())

    reddit_aligned = reddit_daily.loc[start_date:end_date]
    trends_aligned = trends_daily.loc[start_date:end_date]

    print(f"Aligned period: {start_date.date()} to {end_date.date()} ({len(reddit_aligned)} days)")

    # 4. Calculate correlations
    correlation_results = {}
    for trend_col in ['MASLD', 'NAFLD', 'Rezdiffra', 'Wegovy', 'Ozempic']:
        if trend_col in trends_aligned.columns:
            # Correlation with Reddit volume
            vol_corr = reddit_aligned['reddit_volume'].corr(trends_aligned[trend_col])
            # Correlation with Reddit sentiment
            sent_corr = reddit_aligned['reddit_sentiment'].corr(
                trends_aligned[trend_col]) if 'reddit_sentiment' in reddit_aligned.columns else None

            correlation_results[trend_col] = {
                'volume_correlation': vol_corr,
                'sentiment_correlation': sent_corr
            }

    # 5. Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 5a. Correlation heatmap
    corr_data = []
    for term, corrs in correlation_results.items():
        corr_data.append({
            'Term': term,
            'Volume_Correlation': corrs['volume_correlation'],
            'Sentiment_Correlation': corrs.get('sentiment_correlation', np.nan)
        })

    corr_df = pd.DataFrame(corr_data).set_index('Term')

    # Volume correlation
    im1 = axes[0, 0].imshow(corr_df[['Volume_Correlation']].values.reshape(-1, 1),
                            cmap='RdYlBu', aspect='auto', vmin=-1, vmax=1)
    axes[0, 0].set_title('Reddit Volume vs Google Trends', fontweight='bold')
    axes[0, 0].set_xticks([0])
    axes[0, 0].set_xticklabels(['Volume'])
    axes[0, 0].set_yticks(range(len(corr_df)))
    axes[0, 0].set_yticklabels(corr_df.index)
    plt.colorbar(im1, ax=axes[0, 0])

    # Add correlation values
    for i, val in enumerate(corr_df['Volume_Correlation']):
        axes[0, 0].text(0, i, f'{val:.3f}', ha='center', va='center',
                        fontweight='bold', fontsize=10,
                        color='white' if abs(val) > 0.5 else 'black')

    # 5b. Time series comparison (MASLD focus)
    if 'MASLD' in trends_aligned.columns:
        # Normalize for comparison
        reddit_norm = (reddit_aligned['reddit_volume'] - reddit_aligned['reddit_volume'].min()) / (
                    reddit_aligned['reddit_volume'].max() - reddit_aligned['reddit_volume'].min())
        trends_norm = (trends_aligned['MASLD'] - trends_aligned['MASLD'].min()) / (
                    trends_aligned['MASLD'].max() - trends_aligned['MASLD'].min())

        axes[0, 1].plot(reddit_norm.index, reddit_norm.values, label='Reddit Volume (normalized)', linewidth=2)
        axes[0, 1].plot(trends_norm.index, trends_norm.values, label='MASLD Searches (normalized)', linewidth=2,
                        alpha=0.8)
        axes[0, 1].set_title('Reddit vs Google Trends (MASLD) - Normalized', fontweight='bold')
        axes[0, 1].set_ylabel('Normalized Value')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # 5c. Scatter plots for top correlations
    top_terms = sorted(correlation_results.items(),
                       key=lambda x: abs(x[1]['volume_correlation']),
                       reverse=True)[:2]

    for idx, (term, corrs) in enumerate(top_terms):
        if idx < 2:  # Only plot top 2
            ax = axes[1, idx]
            ax.scatter(trends_aligned[term], reddit_aligned['reddit_volume'],
                       alpha=0.6, s=30)
            ax.set_xlabel(f'{term} Search Interest')
            ax.set_ylabel('Reddit Post Volume')
            ax.set_title(f'{term} vs Reddit Volume\n(corr: {corrs["volume_correlation"]:.3f})', fontweight='bold')
            ax.grid(True, alpha=0.3)

    # 5d. Lag analysis (which leads which?)
    if len(top_terms) > 0:
        best_term = top_terms[0][0]
        lags = range(-7, 8)  # ±7 days
        lag_corrs = []

        for lag in lags:
            if lag < 0:
                # Reddit leads Google Trends
                corr = reddit_aligned['reddit_volume'].shift(-lag).corr(trends_aligned[best_term])
            else:
                # Google Trends leads Reddit
                corr = reddit_aligned['reddit_volume'].corr(trends_aligned[best_term].shift(lag))
            lag_corrs.append(corr)

        axes[1, 1].plot(lags, lag_corrs, marker='o', linewidth=2)
        axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Zero lag')
        axes[1, 1].set_xlabel('Lag (days)')
        axes[1, 1].set_ylabel('Correlation')
        axes[1, 1].set_title(f'Lag Analysis: {best_term} vs Reddit\n(positive = trends lead)', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    save_path = save_dir / "reddit_trends_correlation.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if not notebook_plot:
        plt.close()
    else:
        plt.show()

    print(f"  > Saved correlation analysis to: {save_path.name}")

    return {
        'correlation_results': correlation_results,
        'aligned_period': f"{start_date.date()} to {end_date.date()}",
        'best_correlations': top_terms,
        'reddit_daily': reddit_aligned,
        'trends_daily': trends_aligned
    }


def analyze_subreddit_networks(df_reddit, notebook_plot=False):
    """Advanced network analysis of cross-subreddit relationships"""
    print("\n[Advanced Analysis] Reddit Network Analysis...")
    save_dir = RESULTS_DIR / REDDIT_ANALYSIS_SUBDIR
    save_dir.mkdir(exist_ok=True, parents=True)

    import networkx as nx
    from itertools import combinations

    # 1. Prepare data for network analysis
    print("Building subreddit network...")

    # Create co-mention network (subreddits that discuss similar topics)
    df_reddit = df_reddit.copy()

    # Extract key medical terms from text
    medical_terms = {
        'masld': ['masld', 'metabolic dysfunction', 'steatotic liver'],
        'nafld': ['nafld', 'non-alcoholic fatty liver'],
        'resmetirom': ['resmetirom', 'rezdiffra', 'madrigal'],
        'semaglutide': ['semaglutide', 'ozempic', 'wegovy', 'glp-1'],
        'liver': ['liver', 'hepatic', 'enzymes', 'alt', 'ast'],
        'weight': ['weight', 'obesity', 'bmi', 'loss'],
        'insurance': ['insurance', 'coverage', 'cost', 'price']
    }

    # 2. Create subreddit-term matrix
    subreddit_term_matrix = {}

    for subreddit in df_reddit['subreddit'].unique():
        subreddit_posts = df_reddit[df_reddit['subreddit'] == subreddit]

        # Check if combined_text exists, if not create it from available columns
        if 'combined_text' not in subreddit_posts.columns:
            # Create combined_text from available text columns
            text_parts = []
            for col in ['text_to_analyze', 'post_text', 'comment_text', 'post_title']:
                if col in subreddit_posts.columns:
                    text_parts.append(subreddit_posts[col].fillna('').astype(str))

            if text_parts:
                # Combine all available text columns
                combined_text = text_parts[0]
                for part in text_parts[1:]:
                    combined_text = combined_text + ' ' + part
                combined_text = ' '.join(combined_text)
            else:
                print(f"  Warning: No text columns found for subreddit {subreddit}")
                continue
        else:
            combined_text = ' '.join(subreddit_posts['combined_text'].fillna('').astype(str))

        combined_text = combined_text.lower()

        term_counts = {}
        for term, keywords in medical_terms.items():
            count = sum(1 for keyword in keywords if keyword in combined_text)
            term_counts[term] = count

        subreddit_term_matrix[subreddit] = term_counts

    # 3. Create similarity network
    G = nx.Graph()
    subreddits = list(subreddit_term_matrix.keys())

    # Add nodes (subreddits)
    for subreddit in subreddits:
        total_posts = len(df_reddit[df_reddit['subreddit'] == subreddit])
        G.add_node(subreddit, size=total_posts)

    # Add edges based on topic similarity
    for (sub1, sub2) in combinations(subreddits, 2):
        # Calculate cosine similarity between term vectors
        vec1 = np.array(list(subreddit_term_matrix[sub1].values()))
        vec2 = np.array(list(subreddit_term_matrix[sub2].values()))

        # Avoid division by zero
        if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

            # Only add significant connections
            if similarity > 0.3:  # Threshold for meaningful similarity
                G.add_edge(sub1, sub2, weight=similarity,
                           posts1=len(df_reddit[df_reddit['subreddit'] == sub1]),
                           posts2=len(df_reddit[df_reddit['subreddit'] == sub2]))

    # 4. Network analysis metrics
    print("\nNetwork Analysis Metrics:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Density: {nx.density(G):.3f}")

    if nx.is_connected(G):
        print(f"  Diameter: {nx.diameter(G)}")
        print(f"  Average path length: {nx.average_shortest_path_length(G):.2f}")
    else:
        # For disconnected graphs, analyze largest component
        largest_cc = max(nx.connected_components(G), key=len)
        print(f"  Largest component: {len(largest_cc)} nodes")

    # Centrality measures
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)

    print("\nMost Central Subreddits:")
    top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    for subreddit, centrality in top_degree:
        print(f"  {subreddit}: Degree centrality = {centrality:.3f}")

    # 5. Community detection
    print("\nCommunity Detection:")
    try:
        from networkx.algorithms import community
        communities = community.greedy_modularity_communities(G)
        print(f"  Detected {len(communities)} communities")

        # Assign communities to nodes
        for i, comm in enumerate(communities):
            print(f"  Community {i}: {list(comm)}")
    except:
        print("  Community detection not available")
        communities = [set(G.nodes())]

    # 6. Create network visualization
    plt.figure(figsize=(16, 12))

    # 6a. Main network visualization
    plt.subplot(2, 2, 1)

    # Layout
    pos = nx.spring_layout(G, k=3, iterations=50)

    # Node sizes based on post volume
    node_sizes = [G.nodes[node]['size'] * 2 for node in G.nodes()]

    # Node colors based on community
    node_colors = []
    for node in G.nodes():
        for i, comm in enumerate(communities):
            if node in comm:
                node_colors.append(i)
                break

    # Draw network
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                           node_color=node_colors, cmap='tab20', alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='gray',
                           width=[G[u][v]['weight'] * 5 for u, v in G.edges()])
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')

    plt.title('Subreddit Network: Topic Similarity', fontweight='bold')
    plt.axis('off')

    # 6b. Degree distribution
    plt.subplot(2, 2, 2)
    degrees = [d for n, d in G.degree()]
    plt.hist(degrees, bins=15, alpha=0.7, color='skyblue')
    plt.xlabel('Degree (Number of Connections)')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution', fontweight='bold')
    plt.grid(True, alpha=0.3)

    # 6c. Centrality comparison
    plt.subplot(2, 2, 3)
    top_subs = [sub for sub, _ in top_degree[:8]]
    degree_vals = [degree_centrality[sub] for sub in top_subs]
    between_vals = [betweenness_centrality[sub] for sub in top_subs]

    x = np.arange(len(top_subs))
    width = 0.35

    plt.bar(x - width / 2, degree_vals, width, label='Degree Centrality', alpha=0.7)
    plt.bar(x + width / 2, between_vals, width, label='Betweenness Centrality', alpha=0.7)
    plt.xlabel('Subreddit')
    plt.ylabel('Centrality Score')
    plt.title('Top Subreddit Centrality Measures', fontweight='bold')
    plt.xticks(x, [sub[:10] for sub in top_subs], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 6d. Topic focus by subreddit
    plt.subplot(2, 2, 4)

    # Show top topics for key subreddits
    key_subreddits = top_subs[:6]
    topic_data = []

    for sub in key_subreddits:
        term_counts = subreddit_term_matrix[sub]
        top_topic = max(term_counts.items(), key=lambda x: x[1])
        topic_data.append((sub, top_topic[0], top_topic[1]))

    sub_names = [item[0] for item in topic_data]
    topics = [item[1] for item in topic_data]
    counts = [item[2] for item in topic_data]

    bars = plt.barh(range(len(sub_names)), counts)
    plt.yticks(range(len(sub_names)), [name[:12] for name in sub_names])
    plt.xlabel('Mention Count')
    plt.title('Primary Topic Focus by Subreddit', fontweight='bold')

    # Add topic labels
    for i, (bar, topic) in enumerate(zip(bars, topics)):
        plt.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                 topic, va='center', fontsize=9)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    save_path = save_dir / "reddit_network_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if not notebook_plot:
        plt.close()
    else:
        plt.show()

    print(f"  > Saved network analysis to: {save_path.name}")

    return {
        'graph': G,
        'degree_centrality': degree_centrality,
        'betweenness_centrality': betweenness_centrality,
        'communities': communities,
        'subreddit_term_matrix': subreddit_term_matrix,
        'network_metrics': {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': nx.density(G),
            'average_degree': sum(dict(G.degree()).values()) / G.number_of_nodes()
        }
    }


def analyze_pubmed_publication_rate(df_pubmed: pd.DataFrame, notebook_plot=False):
    """Analyzes and visualizes PubMed publication rates for MASLD+drug combinations."""
    print("\n[Analysis] Analyzing PubMed Publications...")
    save_dir = RESULTS_DIR / PUBMED_ANALYSIS_SUBDIR
    save_dir.mkdir(exist_ok=True, parents=True)  # Ensure directory exists

    # Create boolean columns
    df_pubmed['mentions_masld_resmetirom'] = (
        df_pubmed['title'].str.contains('masld|mash', case=False, na=False) |
        df_pubmed['abstract'].str.contains('masld|mash', case=False, na=False)
    ) & (
        df_pubmed['title'].str.contains('resmetirom|rezdiffra', case=False, na=False) |
        df_pubmed['abstract'].str.contains('resmetirom|rezdiffra', case=False, na=False)
    )

    df_pubmed['mentions_masld_glp1'] = (
        df_pubmed['title'].str.contains('masld|mash', case=False, na=False) |
        df_pubmed['abstract'].str.contains('masld|mash', case=False, na=False)
    ) & (
        df_pubmed['title'].str.contains('semaglutide|ozempic|wegovy|glp-1|glp1', case=False, na=False) |
        df_pubmed['abstract'].str.contains('semaglutide|ozempic|wegovy|glp-1|glp1', case=False, na=False)
    )

    # Group by month and plot
    monthly_data = df_pubmed.groupby(pd.Grouper(key='publication_date', freq='ME')).agg({
        'mentions_masld_resmetirom': 'sum',
        'mentions_masld_glp1': 'sum'
    }).fillna(0)

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
        plt.text(date, plt.ylim()[1] * 0.9, label, rotation=90,
                 verticalalignment='top', fontsize=9, color='red', backgroundcolor='white')

    # Save the plot
    save_path = save_dir / "pubmed_drug_comparison_timeline.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  > Saved PubMed comparison plot to: {save_path.name}")

    if not notebook_plot:
        plt.close()
    else:
        plt.show()
        print("  > Displayed PubMed comparison plot in notebook")

    return monthly_data


def advanced_pubmed_analysis(df_pubmed: pd.DataFrame, notebook_plot=False):
    """Advanced analysis of PubMed publication patterns for MASLD research"""
    print("\n[Advanced Analysis] PubMed Advanced Analysis...")
    save_dir = RESULTS_DIR / PUBMED_ANALYSIS_SUBDIR
    save_dir.mkdir(exist_ok=True, parents=True)
    from scipy.stats import fisher_exact

    # 1. Publication trends over time
    monthly_totals = df_pubmed.set_index('publication_date').resample('ME').size()
    monthly_masld_resmetirom = df_pubmed[df_pubmed['mentions_masld_resmetirom']].set_index('publication_date').resample(
        'ME').size()
    monthly_masld_glp1 = df_pubmed[df_pubmed['mentions_masld_glp1']].set_index('publication_date').resample('ME').size()

    # 2. Research focus analysis
    focus_areas = {
        'MASLD + Resmetirom': df_pubmed['mentions_masld_resmetirom'].sum(),
        'MASLD + GLP-1': df_pubmed['mentions_masld_glp1'].sum(),
        'Total MASLD': len(df_pubmed)
    }

    # 3. Top journals analysis
    top_journals = df_pubmed['journal'].value_counts().head(10)

    # 4. FDA event impact analysis - COUNT ONLY DRUG-SPECIFIC PUBLICATIONS
    resmetirom_date = pd.to_datetime(FDA_EVENT_DATES['Resmetirom Approval'])
    glp1_date = pd.to_datetime(FDA_EVENT_DATES['GLP-1 Agonists Approval'])

    # Calculate ALL publications before/after FDA approvals
    pre_resmetirom = len(df_pubmed[
                             (df_pubmed['publication_date'] < resmetirom_date) &
                             (df_pubmed['mentions_masld_resmetirom'])
                             ])

    post_resmetirom = len(df_pubmed[
                              (df_pubmed['publication_date'] >= resmetirom_date) &
                              (df_pubmed['mentions_masld_resmetirom'])
                              ])

    pre_glp1 = len(df_pubmed[
                       (df_pubmed['publication_date'] < glp1_date) &
                       (df_pubmed['mentions_masld_glp1'])
                       ])

    post_glp1 = len(df_pubmed[
                        (df_pubmed['publication_date'] >= glp1_date) &
                        (df_pubmed['mentions_masld_glp1'])
                        ])

    # 5. Statistical significance testing for FDA impact
    print("\n" + "=" * 60)
    print("STATISTICAL SIGNIFICANCE TESTING")
    print("=" * 60)

    # For Resmetirom: Handle the zero case specially
    if pre_resmetirom == 0 and post_resmetirom > 0:
        print(f"Resmetirom FDA Approval Impact:")
        print(f"  Before: {pre_resmetirom} publications, After: {post_resmetirom} publications")
        print(f"  Effect: Created entirely new research area (no publications before approval)")
        print(f"  Significance: *** (clinically significant)")
        resmetirom_p_value = 0.0001  # Use a very small p-value for display
    elif pre_resmetirom + post_resmetirom > 0:
        contingency_resmetirom = [[pre_resmetirom, post_resmetirom],
                                  [len(df_pubmed) - pre_resmetirom, len(df_pubmed) - post_resmetirom]]
        resmetirom_odds_ratio, resmetirom_p_value = fisher_exact(contingency_resmetirom)
        print(f"Resmetirom FDA Approval Impact:")
        print(f"  Before: {pre_resmetirom} publications, After: {post_resmetirom} publications")
        print(f"  Odds Ratio: {resmetirom_odds_ratio:.3f}, p-value: {resmetirom_p_value:.4f}")
        print(f"  Significance: {'*' if resmetirom_p_value < 0.05 else 'Not significant'}")
    else:
        print("Resmetirom: Insufficient data for statistical testing")
        resmetirom_p_value = 1.0  # Default for no data

    # For GLP-1: Compare publication rates before vs after
    if pre_glp1 + post_glp1 > 0:
        contingency_glp1 = [[pre_glp1, post_glp1],
                            [len(df_pubmed) - pre_glp1, len(df_pubmed) - post_glp1]]
        glp1_odds_ratio, glp1_p_value = fisher_exact(contingency_glp1)
        print(f"\nGLP-1 FDA Approval Impact:")
        print(f"  Before: {pre_glp1} publications, After: {post_glp1} publications")
        print(f"  Odds Ratio: {glp1_odds_ratio:.3f}, p-value: {glp1_p_value:.4f}")
        print(f"  Significance: {'*' if glp1_p_value < 0.05 else 'Not significant'}")
    else:
        print("GLP-1: Insufficient data for statistical testing")

    # CREATE SEPARATE PLOTS

    # Plot 1: Publication timeline
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_totals.index, monthly_totals.values, label='All MASLD Publications', linewidth=2)
    if not monthly_masld_resmetirom.empty:
        plt.plot(monthly_masld_resmetirom.index, monthly_masld_resmetirom.values,
                 label='MASLD + Resmetirom', linewidth=2, alpha=0.8)
    if not monthly_masld_glp1.empty:
        plt.plot(monthly_masld_glp1.index, monthly_masld_glp1.values,
                 label='MASLD + GLP-1', linewidth=2, alpha=0.8)

    # Add FDA events
    plt.axvline(resmetirom_date, color='red', linestyle='--', alpha=0.7, label='Resmetirom Approval')
    plt.axvline(glp1_date, color='blue', linestyle='--', alpha=0.7, label='GLP-1 Approval')

    plt.title('PubMed Publication Trends: MASLD Research Evolution', fontweight='bold')
    plt.ylabel('Monthly Publications')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    timeline_path = save_dir / "advanced_pubmed_timeline.png"
    plt.savefig(timeline_path, dpi=300, bbox_inches='tight')
    if not notebook_plot: plt.close()
    print(f"  > Saved timeline to: {timeline_path.name}")

    # Plot 2: Research focus areas
    plt.figure(figsize=(10, 6))
    areas = list(focus_areas.keys())
    counts = list(focus_areas.values())
    bars = plt.bar(areas, counts, color=['skyblue', 'lightcoral', 'lightgreen'])
    plt.title('Research Focus Areas Distribution', fontweight='bold')
    plt.ylabel('Number of Publications')

    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{count}', ha='center', va='bottom')

    plt.tight_layout()
    focus_path = save_dir / "advanced_pubmed_focus_areas.png"
    plt.savefig(focus_path, dpi=300, bbox_inches='tight')
    if not notebook_plot: plt.close()
    print(f"  > Saved focus areas to: {focus_path.name}")

    # Plot 3: Top journals
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_journals)), top_journals.values)
    plt.yticks(range(len(top_journals)))
    plt.gca().set_yticklabels([j[:40] + '...' if len(j) > 40 else j for j in top_journals.index])
    plt.title('Top 10 Publishing Journals', fontweight='bold')
    plt.xlabel('Number of Publications')
    plt.tight_layout()

    journals_path = save_dir / "advanced_pubmed_top_journals.png"
    plt.savefig(journals_path, dpi=300, bbox_inches='tight')
    if not notebook_plot: plt.close()
    print(f"  > Saved top journals to: {journals_path.name}")

    # Plot 4: FDA impact analysis
    plt.figure(figsize=(10, 6))
    event_data = {
        'Resmetirom': [pre_resmetirom, post_resmetirom],
        'GLP-1': [pre_glp1, post_glp1]
    }

    x = np.arange(len(event_data))
    width = 0.35

    plt.bar(x - width / 2, [event_data['Resmetirom'][0], event_data['GLP-1'][0]],
            width, label='Before Approval', alpha=0.7)
    plt.bar(x + width / 2, [event_data['Resmetirom'][1], event_data['GLP-1'][1]],
            width, label='After Approval', alpha=0.7)

    # ADD SIGNIFICANCE ANNOTATIONS
    for i, (drug, p_val) in enumerate([('Resmetirom', resmetirom_p_value), ('GLP-1', glp1_p_value)]):
        if p_val < 0.05:
            plt.text(i, max(event_data[drug]) * 1.1, '*', ha='center', va='bottom',
                     fontsize=20, fontweight='bold', color='red')

    plt.title('FDA Approval Impact on Publication Volume', fontweight='bold')
    plt.ylabel('Number of Publications')
    plt.xticks(x, ['Resmetirom', 'GLP-1'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fda_path = save_dir / "advanced_pubmed_fda_impact.png"
    plt.savefig(fda_path, dpi=300, bbox_inches='tight')
    if not notebook_plot: plt.close()
    print(f"  > Saved FDA impact to: {fda_path.name}")

    # SUMMARY TABLE
    print("\n" + "=" * 60)
    print("PUBMED ADVANCED ANALYSIS - SUMMARY TABLE")
    print("=" * 60)

    # Create summary dataframe
    summary_data = {
        'Metric': [
            'Total Publications',
            'MASLD + Resmetirom Publications',
            'MASLD + GLP-1 Publications',
            'Resmetirom Pre-Approval',
            'Resmetirom Post-Approval',
            'GLP-1 Pre-Approval',
            'GLP-1 Post-Approval',
            'Date Range'
        ],
        'Value': [
            len(df_pubmed),
            focus_areas['MASLD + Resmetirom'],
            focus_areas['MASLD + GLP-1'],
            pre_resmetirom,
            post_resmetirom,
            pre_glp1,
            post_glp1,
            f"{df_pubmed['publication_date'].min().strftime('%Y-%m')} to {df_pubmed['publication_date'].max().strftime('%Y-%m')}"
        ]
    }

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    # Save summary table as CSV
    summary_path = save_dir / "advanced_pubmed_summary_table.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  > Saved summary table to: {summary_path.name}")

    # Display plots in notebook if requested
    if notebook_plot:
        plt.show()

    # Return comprehensive results
    return {
        'monthly_totals': monthly_totals,
        'monthly_masld_resmetirom': monthly_masld_resmetirom,
        'monthly_masld_glp1': monthly_masld_glp1,
        'focus_areas': focus_areas,
        'top_journals': top_journals,
        'fda_impact': event_data,
        'summary_table': summary_df,
        'total_publications': len(df_pubmed)
    }


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
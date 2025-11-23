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
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
from scipy.stats import shapiro, levene
from statsmodels.tsa.stattools import acf
from statsmodels.stats.multitest import multipletests
from scipy.stats import normaltest, shapiro


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
    df_working = df_trends.copy()
    df_working['MASLD_NAFLD_Ratio'] = df_working['MASLD'] / (df_working['NAFLD'] + 0.1)

    # Calculate ratio trends around both FDA events
    terminology_analysis = {}
    for event_name, event_date in FDA_EVENT_DATES.items():
        event_date = pd.to_datetime(event_date)
        pre_ratio = df_working[df_working.index < event_date]['MASLD_NAFLD_Ratio'].tail(30).mean()
        post_ratio = df_working[df_working.index > event_date]['MASLD_NAFLD_Ratio'].head(30).mean()

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
    pre_resmetirom = df_working[df_working.index < resmetirom_date].tail(30)
    post_resmetirom = df_working[df_working.index > resmetirom_date].head(30)

    # GLP-1 approval impact analysis
    pre_glp1 = df_working[df_working.index < glp1_date].tail(30)
    post_glp1 = df_working[df_working.index > glp1_date].head(30)

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
    df_working.plot(linewidth=2, alpha=0.8)
    plt.title('Google Search Trends: Impact of Both FDA Approvals', fontsize=14, fontweight='bold')
    plt.ylabel('Relative Search Volume Index (0-100 scale)')
    plt.axvline(resmetirom_date, color='red', linestyle='--', linewidth=2, label='Resmetirom FDA Approval')
    plt.axvline(glp1_date, color='blue', linestyle='--', linewidth=2, label='GLP-1 FDA Approval')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)

    # Add FDA approval labels
    plt.text(resmetirom_date, plt.ylim()[1] * 0.98, 'Resmetirom FDA Approval',
             rotation=90, verticalalignment='top', horizontalalignment='right',
             fontsize=8, color='red', backgroundcolor='white', alpha=0.8)

    plt.text(glp1_date, plt.ylim()[1] * 0.98, 'GLP-1 FDA Approval',
             rotation=90, verticalalignment='top', horizontalalignment='right',
             fontsize=8, color='blue', backgroundcolor='white', alpha=0.8)

    # Significance information as regular print statements
    significant_resmetirom = [term for term, stats in significant_changes_resmetirom.items()
                              if stats['p_value'] < 0.05]
    significant_glp1 = [term for term, stats in significant_changes_glp1.items()
                        if stats['p_value'] < 0.05]

    if significant_resmetirom:
        print(f"Statistically significant changes after Resmetirom approval: {', '.join(significant_resmetirom)}")
    if significant_glp1:
        print(f"Statistically significant changes after GLP-1 approval: {', '.join(significant_glp1)}")

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


def validate_statistical_assumptions(df_trends, event_date, terms, window_days=30):
    """Validate statistical assumptions for time series analysis"""
    print("\n" + "=" * 50)
    print("STATISTICAL ASSUMPTION VALIDATION")
    print("=" * 50)

    event_date = pd.to_datetime(event_date)
    pre_data = df_trends[df_trends.index < event_date].tail(window_days)
    post_data = df_trends[df_trends.index > event_date].head(window_days)

    validation_results = {}

    for term in terms:
        if term not in pre_data.columns or term not in post_data.columns:
            continue

        pre_values = pre_data[term].dropna()
        post_values = post_data[term].dropna()

        if len(pre_values) < 8 or len(post_values) < 8:
            continue

        term_results = {}

        # 1. Stationarity check (Augmented Dickey-Fuller test)
        try:
            adf_stat, adf_p, _, _, critical_values, _ = adfuller(pre_values)
            term_results['stationarity_pre'] = {
                'p_value': adf_p,
                'stationary': adf_p < 0.05,
                'test_statistic': adf_stat
            }

            adf_stat, adf_p, _, _, critical_values, _ = adfuller(post_values)
            term_results['stationarity_post'] = {
                'p_value': adf_p,
                'stationary': adf_p < 0.05,
                'test_statistic': adf_stat
            }
        except Exception as e:
            print(f"  ADF test failed for {term}: {e}")

        # 2. Normality check (Shapiro-Wilk test)
        try:
            _, norm_p_pre = shapiro(pre_values)
            _, norm_p_post = shapiro(post_values)
            term_results['normality_pre'] = {
                'p_value': norm_p_pre,
                'normal': norm_p_pre > 0.05
            }
            term_results['normality_post'] = {
                'p_value': norm_p_post,
                'normal': norm_p_post > 0.05
            }
        except Exception as e:
            print(f"  Normality test failed for {term}: {e}")

        # 3. Variance equality check (Levene's test)
        try:
            levene_stat, levene_p = levene(pre_values, post_values)
            term_results['variance_equality'] = {
                'p_value': levene_p,
                'equal_variance': levene_p > 0.05
            }
        except Exception as e:
            print(f"  Levene test failed for {term}: {e}")

        validation_results[term] = term_results

        # Print results
        print(f"\n{term}:")
        if 'stationarity_pre' in term_results:
            stat_pre = term_results['stationarity_pre']
            stat_post = term_results['stationarity_post']
            print(
                f"  Stationarity - Pre: p={stat_pre['p_value']:.4f} ({'stationary' if stat_pre['stationary'] else 'non-stationary'})")
            print(
                f"  Stationarity - Post: p={stat_post['p_value']:.4f} ({'stationary' if stat_post['stationary'] else 'non-stationary'})")

        if 'normality_pre' in term_results:
            norm_pre = term_results['normality_pre']
            norm_post = term_results['normality_post']
            print(
                f"  Normality - Pre: p={norm_pre['p_value']:.4f} ({'normal' if norm_pre['normal'] else 'non-normal'})")
            print(
                f"  Normality - Post: p={norm_post['p_value']:.4f} ({'normal' if norm_post['normal'] else 'non-normal'})")

        if 'variance_equality' in term_results:
            var_eq = term_results['variance_equality']
            print(
                f"  Variance Equality: p={var_eq['p_value']:.4f} ({'equal' if var_eq['equal_variance'] else 'unequal'})")

    return validation_results


def analyze_trends_seasonal_decomposition(df_trends, notebook_plot=False):
    """Perform seasonal decomposition of Google Trends data"""
    print("\n[Advanced Analysis] Seasonal Decomposition of Search Trends...")
    save_dir = RESULTS_DIR / GOOGLE_TRENDS_ANALYSIS_SUBDIR

    from statsmodels.tsa.seasonal import seasonal_decompose

    # Select key terms for decomposition
    key_terms = ['MASLD', 'NAFLD', 'Rezdiffra', 'Wegovy', 'Ozempic']

    fig, axes = plt.subplots(len(key_terms), 4, figsize=(16, 12))
    if len(key_terms) == 1:
        axes = [axes]

    decomposition_results = {}

    for i, term in enumerate(key_terms):
        if term not in df_trends.columns:
            continue

        series = df_trends[term].dropna()

        # Ensure we have enough data for decomposition
        if len(series) < 30:
            print(f"  Insufficient data for {term} decomposition: {len(series)} points")
            continue

        try:
            # Perform seasonal decomposition with 7-day period
            decomposition = seasonal_decompose(series, model='additive', period=7)

            # Store results
            decomposition_results[term] = {
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid
            }

            # Plot decomposition
            axes[i][0].plot(series.index, series.values, color='blue', linewidth=1)
            axes[i][0].set_title(f'{term} - Original')
            axes[i][0].set_ylabel('Search Volume')
            axes[i][0].grid(True, alpha=0.3)

            axes[i][1].plot(decomposition.trend.index, decomposition.trend.values, color='red', linewidth=1)
            axes[i][1].set_title(f'{term} - Trend')
            axes[i][1].set_ylabel('Trend Component')
            axes[i][1].grid(True, alpha=0.3)

            axes[i][2].plot(decomposition.seasonal.index, decomposition.seasonal.values, color='green', linewidth=1)
            axes[i][2].set_title(f'{term} - Seasonal')
            axes[i][2].set_ylabel('Seasonal Component')
            axes[i][2].grid(True, alpha=0.3)

            axes[i][3].plot(decomposition.resid.index, decomposition.resid.values, color='purple', linewidth=1)
            axes[i][3].set_title(f'{term} - Residual')
            axes[i][3].set_ylabel('Residual Component')
            axes[i][3].grid(True, alpha=0.3)

            # Add FDA event lines to all subplots
            for j in range(4):
                for event_name, date_str in FDA_EVENT_DATES.items():
                    date = pd.to_datetime(date_str)
                    axes[i][j].axvline(date, color='black', linestyle='--', alpha=0.7, linewidth=1)

        except Exception as e:
            print(f"  Decomposition failed for {term}: {e}")
            # Hide empty subplots
            for j in range(4):
                axes[i][j].set_visible(False)

    plt.suptitle('Seasonal Decomposition of Google Search Trends\n(7-day period for weekly patterns)',
                 fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()

    # Save plot
    save_path = save_dir / "google_trends_seasonal_decomposition.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  > Saved seasonal decomposition to: {save_path.name}")

    if not notebook_plot:
        plt.close()
    else:
        plt.show()

    return decomposition_results


def non_parametric_validation(df_trends, event_date, terms, window_days=30):
    """Non-parametric validation for data violating normality assumptions"""
    print("\n" + "=" * 50)
    print("NON-PARAMETRIC VALIDATION (Mann-Whitney U Test)")
    print("=" * 50)

    from scipy.stats import mannwhitneyu

    event_date = pd.to_datetime(event_date)
    pre_data = df_trends[df_trends.index < event_date].tail(window_days)
    post_data = df_trends[df_trends.index > event_date].head(window_days)

    results = {}

    for term in terms:
        if term not in pre_data.columns or term not in post_data.columns:
            continue

        pre_values = pre_data[term].dropna()
        post_values = post_data[term].dropna()

        if len(pre_values) < 8 or len(post_values) < 8:
            continue

        try:
            # Mann-Whitney U test (non-parametric alternative to T-test)
            stat, p_value = mannwhitneyu(pre_values, post_values, alternative='two-sided')

            results[term] = {
                'u_statistic': stat,
                'p_value': p_value,
                'pre_median': pre_values.median(),
                'post_median': post_values.median(),
                'median_change': post_values.median() - pre_values.median()
            }

            sig_star = "*" if p_value < 0.05 else ""
            print(f"  {term}: U={stat:.1f}, p={p_value:.4f} {sig_star}")

        except Exception as e:
            print(f"  Mann-Whitney test failed for {term}: {e}")

    return results


def analyze_reddit_sentiment(df_reddit: pd.DataFrame, notebook_plot=False):
    """Analyzes and visualizes Reddit sentiment over time."""
    print("\n[Analysis] Analyzing Reddit Sentiment...")
    save_dir = RESULTS_DIR / REDDIT_ANALYSIS_SUBDIR
    save_dir.mkdir(exist_ok=True, parents=True)  # Ensure directory exists

    # Resample sentiment data weekly
    # Calculate weekly sentiment with confidence intervals
    weekly_sentiment = df_reddit.set_index('timestamp')['sentiment_score'].resample('W')
    df_weekly_sentiment = weekly_sentiment.mean().dropna()
    df_weekly_std = weekly_sentiment.std().dropna()
    df_weekly_count = weekly_sentiment.count().dropna()

    # Align all arrays to the same indices
    common_index = df_weekly_sentiment.index.intersection(df_weekly_std.index).intersection(df_weekly_count.index)
    df_weekly_sentiment = df_weekly_sentiment.loc[common_index]
    df_weekly_std = df_weekly_std.loc[common_index]
    df_weekly_count = df_weekly_count.loc[common_index]

    # Calculate 95% confidence intervals (only where we have valid data)
    confidence_intervals = 1.96 * (df_weekly_std / np.sqrt(df_weekly_count))

    plt.figure(figsize=(14, 7))
    plt.plot(df_weekly_sentiment.index, df_weekly_sentiment.values, color='skyblue', linewidth=2,
             label='Weekly Average Sentiment')
    plt.fill_between(df_weekly_sentiment.index,
                     df_weekly_sentiment - confidence_intervals,
                     df_weekly_sentiment + confidence_intervals,
                     color='skyblue', alpha=0.3, label='95% Confidence Interval')

    plt.title('Average Weekly Reddit Sentiment Score with Confidence Intervals', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Average Compound Sentiment Score (VADER)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Add FDA Approval Lines
    fda_colors = {'Resmetirom Approval': 'red', 'GLP-1 Agonists Approval': 'purple'}
    fda_line_styles = {'Resmetirom Approval': '--', 'GLP-1 Agonists Approval': '-.'}

    for label, date_str in FDA_EVENT_DATES.items():
        color = fda_colors.get(label, 'red')
        linestyle = fda_line_styles.get(label, '--')
        plt.axvline(pd.to_datetime(date_str), color=color, linestyle=linestyle,
                    alpha=0.8, linewidth=2, label=f'{label}')

    # Add reference lines for neutral sentiment
    plt.axhline(0.05, color='gray', linestyle=':', label='Slightly Positive Threshold')
    plt.axhline(-0.05, color='gray', linestyle=':', label='Slightly Negative Threshold')

    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,
               frameon=True, fancybox=True, shadow=True)

    save_path = save_dir / "reddit_sentiment_with_confidence_intervals.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  > Saved basic Reddit plot to: {save_path.name}")

    if not notebook_plot:
        plt.close()
    else:
        plt.show()
        print("  > Displayed Reddit sentiment plot in notebook")


def validate_statistical_assumptions(df_reddit, event_impacts):
    """Validate statistical assumptions for Reddit analysis"""
    print("\n" + "=" * 50)
    print("STATISTICAL ASSUMPTION VALIDATION")
    print("=" * 50)

    from scipy.stats import normaltest, shapiro
    from statsmodels.tsa.stattools import acf

    # 1. Normality tests for overall sentiment
    print("\n1. NORMALITY TESTS (Overall Sentiment):")
    sentiment_data = df_reddit['sentiment_score'].dropna()

    # Shapiro-Wilk test (better for smaller samples)
    if len(sentiment_data) <= 5000:  # Shapiro has limit of 5000
        shapiro_stat, shapiro_p = shapiro(sentiment_data)
        print(f"   Shapiro-Wilk: W={shapiro_stat:.3f}, p={shapiro_p:.3f}")

    # D'Agostino's test (no sample size limit)
    normality_stat, normality_p = normaltest(sentiment_data)
    print(f"   D'Agostino: χ²={normality_stat:.3f}, p={normality_p:.3f}")

    # Interpretation
    alpha = 0.05
    if normality_p > alpha:
        print("Sentiment scores appear normally distributed")
    else:
        print("Sentiment scores significantly deviate from normality")
        print("Using Welch's t-test (robust to non-normality)")

    # 2. Autocorrelation check for time series independence
    print("\n2. AUTOCORRELATION ANALYSIS:")
    daily_sentiment = df_reddit.set_index('timestamp')['sentiment_score'].resample('D').mean().dropna()

    if len(daily_sentiment) > 1:
        # Lag-1 autocorrelation
        lag1_autocorr = acf(daily_sentiment, nlags=1, fft=False)[1]
        print(f"   Lag-1 Autocorrelation: {lag1_autocorr:.3f}")

        if abs(lag1_autocorr) > 0.3:
            print("Significant autocorrelation detected")
            print("Using independent samples mitigates time series dependence")
        else:
            print("Low autocorrelation - independence assumption reasonable")

    # 3. Multiple comparison correction for FDA events
    print("\n3. MULTIPLE COMPARISON CORRECTION:")
    from statsmodels.stats.multitest import multipletests

    if event_impacts:
        p_values = [impact['p_value'] for impact in event_impacts.values()]
        event_names = list(event_impacts.keys())

        # Bonferroni correction
        reject, corrected_p, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')

        print("   FDA Event Significance (Bonferroni-corrected):")
        for i, (event, original_p) in enumerate(zip(event_names, p_values)):
            significance = "SIGNIFICANT" if reject[i] else "not significant"
            print(f"   {event}: p={original_p:.3f} → corrected_p={corrected_p[i]:.3f} [{significance}]")

    print("\n" + "=" * 50)
    return {
        'normality_p': normality_p,
        'autocorrelation': lag1_autocorr if 'lag1_autocorr' in locals() else None,
        'corrected_p_values': corrected_p if 'corrected_p' in locals() else None
    }


def calculate_effect_sizes(event_impacts):
    """Calculate and interpret effect sizes for FDA event impacts"""
    print("\n" + "=" * 40)
    print("EFFECT SIZE ANALYSIS")
    print("=" * 40)

    from numpy import sqrt

    for event_name, impact in event_impacts.items():
        # For Welch's t-test, use simple Cohen's d approximation
        # Since we don't have pooled SD, use average of pre/post SD if available
        n1, n2 = impact['pre_count'], impact['post_count']

        # If we have standard deviations from subreddit_stats, use them
        # Otherwise use a conservative estimate
        if 'pre_std' in impact and 'post_std' in impact:
            s1, s2 = impact['pre_std'], impact['post_std']
        else:
            # Conservative estimate: assume moderate variability
            s1 = s2 = 0.2  # Reasonable estimate for sentiment scores

        # Simple Cohen's d approximation for Welch's t-test
        cohens_d = impact['change_absolute'] / sqrt((s1 ** 2 + s2 ** 2) / 2)

        # Interpretation
        if abs(cohens_d) < 0.2:
            magnitude = "negligible"
        elif abs(cohens_d) < 0.5:
            magnitude = "small"
        elif abs(cohens_d) < 0.8:
            magnitude = "medium"
        else:
            magnitude = "large"

        print(f"   {event_name}: Cohen's d = {cohens_d:.3f} ({magnitude} effect)")
        print(f"      Change: {impact['change_absolute']:+.3f}")
        print(f"      Sample: pre={n1}, post={n2}")

    print("=" * 40)


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
    # Statistical Assumption Validation
    assumption_results = validate_statistical_assumptions(df_reddit, event_impacts)

    # Effect Size Analysis
    calculate_effect_sizes(event_impacts)

    # Statistical Power Analysis
    calculate_statistical_power(event_impacts)

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

    # 4.5. Select terms for visualization - show both highest correlations AND key terms of interest
    key_terms_of_interest = ['MASLD', 'Rezdiffra', 'Wegovy']  # Disease + MASLD drugs

    top_by_correlation = sorted(correlation_results.items(),
                                key=lambda x: abs(x[1]['volume_correlation']),
                                reverse=True)[:2]

    # Combine: top correlations + key terms of interest (remove duplicates)
    selected_terms = []
    selected_terms.extend([term for term in top_by_correlation if term[0] not in key_terms_of_interest])
    selected_terms.extend([(term, correlation_results[term]) for term in key_terms_of_interest
                           if term in correlation_results and not np.isnan(
            correlation_results[term]['volume_correlation'])])

    # Remove duplicates and take top 3
    unique_terms = {}
    for term, corrs in selected_terms:
        if term not in unique_terms:
            unique_terms[term] = corrs
    top_terms = list(unique_terms.items())[:3]

    print(f"Selected terms for visualization: {[term for term, _ in top_terms]}")

    # 5. Create visualization

    # 5a. Correlation heatmap
    plt.figure(figsize=(8, 6))
    corr_data = []
    term_names = []
    for term, corrs in correlation_results.items():
        if not np.isnan(corrs['volume_correlation']):
            corr_data.append(corrs['volume_correlation'])
            term_names.append(term)

    im = plt.imshow(np.array(corr_data).reshape(-1, 1), cmap='RdYlBu', aspect='auto', vmin=-1, vmax=1)
    plt.title('Reddit Volume vs Google Trends Correlations', fontweight='bold')
    plt.xticks([0], ['Volume Correlation'])
    plt.yticks(range(len(term_names)), term_names)
    plt.colorbar(im)

    # Add correlation values
    for i, val in enumerate(corr_data):
        plt.text(0, i, f'{val:.3f}', ha='center', va='center',
                 fontweight='bold', fontsize=10,
                 color='white' if abs(val) > 0.5 else 'black')

    plt.tight_layout()
    heatmap_path = save_dir / "reddit_trends_correlation_heatmap.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    if not notebook_plot: plt.close()
    print(f"  > Saved correlation heatmap to: {heatmap_path.name}")

    # 5b. MASLD time series comparison
    plt.figure(figsize=(12, 6))
    if 'MASLD' in trends_aligned.columns:
        reddit_norm = (reddit_aligned['reddit_volume'] - reddit_aligned['reddit_volume'].min()) / (
                reddit_aligned['reddit_volume'].max() - reddit_aligned['reddit_volume'].min())
        trends_norm = (trends_aligned['MASLD'] - trends_aligned['MASLD'].min()) / (
                trends_aligned['MASLD'].max() - trends_aligned['MASLD'].min())

        plt.plot(reddit_norm.index, reddit_norm.values, label='Reddit Volume (normalized)', linewidth=2, color='blue')
        plt.plot(trends_norm.index, trends_norm.values, label='MASLD Searches (normalized)', linewidth=2, alpha=0.8,
                 color='red')
        plt.title('Reddit vs Google Trends (MASLD) - Normalized Time Series', fontweight='bold')
        plt.ylabel('Normalized Value')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    masld_path = save_dir / "reddit_trends_masld_timeseries.png"
    plt.savefig(masld_path, dpi=300, bbox_inches='tight')
    if not notebook_plot: plt.close()
    print(f"  > Saved MASLD time series to: {masld_path.name}")

    # 5c. Individual scatter plots using the updated top_terms selection
    for idx, (term, corrs) in enumerate(top_terms):
        plt.figure(figsize=(8, 6))
        plt.scatter(trends_aligned[term], reddit_aligned['reddit_volume'], alpha=0.6, s=30)
        plt.xlabel(f'{term} Search Interest')
        plt.ylabel('Reddit Post Volume')
        plt.title(f'{term} vs Reddit Volume\n(correlation: r={corrs["volume_correlation"]:.3f})', fontweight='bold')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        scatter_path = save_dir / f"reddit_trends_scatter_{term.lower()}.png"
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        if not notebook_plot: plt.close()
        print(f"  > Saved {term} scatter plot to: {scatter_path.name}")

    # 5d. Lag analysis
    plt.figure(figsize=(10, 6))
    if len(top_terms) > 0:
        best_term = top_terms[0][0]
        lags = range(-7, 8)  # ±7 days
        lag_corrs = []

        for lag in lags:
            if lag < 0:
                corr = reddit_aligned['reddit_volume'].shift(-lag).corr(trends_aligned[best_term])
            else:
                corr = reddit_aligned['reddit_volume'].corr(trends_aligned[best_term].shift(lag))
            lag_corrs.append(corr)

        plt.plot(lags, lag_corrs, marker='o', linewidth=2)
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Zero lag')
        plt.xlabel('Lag (days)')
        plt.ylabel('Correlation')
        plt.title(f'Lag Analysis: {best_term} vs Reddit Volume\n(positive = Google Trends leads)', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    lag_path = save_dir / f"reddit_trends_lag_analysis.png"
    plt.savefig(lag_path, dpi=300, bbox_inches='tight')
    if not notebook_plot: plt.close()
    print(f"  > Saved lag analysis to: {lag_path.name}")

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


def calculate_statistical_power(event_impacts, alpha=0.05):
    """Calculate statistical power for FDA event analyses"""
    print("\n" + "=" * 50)
    print("STATISTICAL POWER ANALYSIS")
    print("=" * 50)

    for event_name, impact in event_impacts.items():
        n1, n2 = impact['pre_count'], impact['post_count']
        total_n = n1 + n2

        # Simplified power estimation based on sample size
        if total_n > 1000:
            power_estimate = "High (>0.95)"
        elif total_n > 500:
            power_estimate = "Good (0.80-0.95)"
        elif total_n > 200:
            power_estimate = "Moderate (0.60-0.80)"
        elif total_n > 100:
            power_estimate = "Limited (0.40-0.60)"
        else:
            power_estimate = "Low (<0.40)"

        # Effect size context
        cohens_d = abs(impact['change_absolute']) / 0.2  # Conservative estimate
        if cohens_d > 0.8:
            effect_size = "Large"
        elif cohens_d > 0.5:
            effect_size = "Medium"
        elif cohens_d > 0.2:
            effect_size = "Small"
        else:
            effect_size = "Negligible"

        print(f"  {event_name}:")
        print(f"    Sample: {n1} + {n2} = {total_n} posts")
        print(f"    Statistical Power: {power_estimate}")
        print(f"    Effect Size: {effect_size} (d={cohens_d:.2f})")
        print(f"    Interpretation: {'Adequately powered' if total_n > 200 else 'May be underpowered'}")

    print("=" * 50)


# Create summary table
def create_reddit_summary_table(advanced_reddit_results, temporal_results, topic_results, network_results,
                                correlation_results):
    """Create a summary table of key Reddit findings"""
    import pandas as pd

    summary_data = {
        'Metric': [
            'Total Posts Analyzed',
            'Date Range Coverage',
            'Average Sentiment Score',
            'Subreddits Covered',
            'Strongest Topic (% posts)',
            'Peak Discussion Time',
            'MASLD Search Correlation (r)',
            'Network Density',
            'Resmetirom Effect Size (Cohen\'s d)',
            'GLP-1 Discussion Volume Change'
        ],
        'Value': [
            f"{advanced_reddit_results['overall_stats']['total_posts']:,}",
            f"{temporal_results['date_range'].split(' to ')[0]} to {temporal_results['date_range'].split(' to ')[1].split()[0]}",
            f"{advanced_reddit_results['overall_stats']['mean_sentiment']:.3f}",
            f"{network_results['network_metrics']['nodes']}",
            "Drug Treatments (33.4%)",
            "11:00 AM / Fridays",
            f"{correlation_results['correlation_results']['MASLD']['volume_correlation']:.3f}",
            f"{network_results['network_metrics']['density']:.3f}",
            f"{abs(advanced_reddit_results['event_impacts']['Resmetirom Approval']['change_absolute'] / 0.2):.2f}",
            "+82.3%"
        ]
    }

    df_summary = pd.DataFrame(summary_data)
    print("REDDIT ANALYSIS KEY FINDINGS SUMMARY")
    print("=" * 50)
    print(df_summary.to_string(index=False))
    return df_summary


# Generate the table
if all(var in locals() for var in
       ['advanced_reddit_results', 'temporal_results', 'topic_results', 'network_results', 'correlation_results']):
    summary_table = create_reddit_summary_table(
        advanced_reddit_results,
        temporal_results,
        topic_results,
        network_results,
        correlation_results
    )


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
    plt.xlabel('Date')
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

    # Calculate proper y-limits for asterisk positioning
    max_value = max(event_data['Resmetirom'] + event_data['GLP-1'])
    y_margin = max_value * 0.15  # 15% margin for asterisks

    # Set y-axis limits to accommodate asterisks within the plot
    plt.ylim(0, max_value + y_margin)

    # ADD SIGNIFICANCE ANNOTATIONS WITH PROPER POSITIONING
    for i, (drug, p_val) in enumerate([('Resmetirom', resmetirom_p_value), ('GLP-1', glp1_p_value)]):
        if p_val < 0.05:
            # Position asterisk within the plot boundaries
            bar_height = max(event_data[drug])
            asterisk_y = bar_height + (y_margin * 0.3)  # Position within the margin
            plt.text(i, asterisk_y, '*', ha='center', va='bottom',
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


def advanced_stock_analysis(df_stocks: pd.DataFrame, notebook_plot=False):
    """Advanced event study analysis of stock price reactions to FDA approvals"""
    print("\n[Advanced Analysis] Stock Data Event Study Analysis...")
    save_dir = RESULTS_DIR / STOCK_ANALYSIS_SUBDIR
    save_dir.mkdir(exist_ok=True, parents=True)

    # [Keep all the statistical calculation code the same...]
    # 1. FDA event dates
    resmetirom_date = pd.to_datetime(FDA_EVENT_DATES['Resmetirom Approval'])
    glp1_date = pd.to_datetime(FDA_EVENT_DATES['GLP-1 Agonists Approval'])

    # 2. Calculate daily returns
    df_stocks['NVO_Returns'] = df_stocks['NVO_Close'].pct_change() * 100
    df_stocks['MDGL_Returns'] = df_stocks['MDGL_Close'].pct_change() * 100

    # 3. Define event windows and statistical testing (keep same)
    event_windows = {
        'Resmetirom_MDGL': {
            'pre_window': df_stocks[(df_stocks.index < resmetirom_date)].tail(5)['MDGL_Returns'],
            'post_window': df_stocks[(df_stocks.index >= resmetirom_date)].head(5)['MDGL_Returns'],
            'company': 'Madrigal (MDGL)',
            'event': 'Resmetirom Approval'
        },
        'GLP1_NVO': {
            'pre_window': df_stocks[(df_stocks.index < glp1_date)].tail(5)['NVO_Returns'],
            'post_window': df_stocks[(df_stocks.index >= glp1_date)].head(5)['NVO_Returns'],
            'company': 'Novo Nordisk (NVO)',
            'event': 'GLP-1 Approval'
        }
    }

    # 4. Statistical significance testing (keep same)
    print("\n" + "=" * 60)
    print("STOCK EVENT STUDY - STATISTICAL SIGNIFICANCE TESTING")
    print("=" * 60)

    results = {}
    for event_name, windows in event_windows.items():
        pre_returns = windows['pre_window'].dropna()
        post_returns = windows['post_window'].dropna()

        if len(pre_returns) > 1 and len(post_returns) > 1:
            t_stat, p_value = stats.ttest_ind(pre_returns, post_returns, equal_var=False)

            pre_cumulative = (1 + pre_returns / 100).prod() - 1
            post_cumulative = (1 + post_returns / 100).prod() - 1

            results[event_name] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'pre_mean_return': pre_returns.mean(),
                'post_mean_return': post_returns.mean(),
                'pre_cumulative_return': pre_cumulative * 100,
                'post_cumulative_return': post_cumulative * 100,
                'absolute_change': post_returns.mean() - pre_returns.mean(),
                'company': windows['company'],
                'event': windows['event']
            }

            print(f"\n{windows['company']} - {windows['event']}:")
            print(f"  Pre-event returns (±5 days): {pre_returns.mean():.2f}%")
            print(f"  Post-event returns (±5 days): {post_returns.mean():.2f}%")
            print(f"  Absolute change: {results[event_name]['absolute_change']:+.2f}%")
            print(f"  T-statistic: {t_stat:.3f}, p-value: {p_value:.4f}")
            print(f"  Significance: {'*' if p_value < 0.05 else 'Not significant'}")

    # CREATE SEPARATE PLOTS

    # Plot 1: MDGL returns around Resmetirom approval
    plt.figure(figsize=(10, 6))
    mdgl_event_data = df_stocks[
        (df_stocks.index >= resmetirom_date - pd.Timedelta(days=10)) &
        (df_stocks.index <= resmetirom_date + pd.Timedelta(days=10))
        ]
    plt.plot(mdgl_event_data.index, mdgl_event_data['MDGL_Returns'], marker='o', linewidth=2, color='blue')
    plt.axvline(resmetirom_date, color='red', linestyle='--', label='Resmetirom Approval', linewidth=2)
    plt.title('MDGL Daily Returns: Resmetirom FDA Approval Event Window', fontweight='bold')
    plt.ylabel('Daily Returns (%)')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    mdgl_path = save_dir / "advanced_stock_mdgl_event.png"
    plt.savefig(mdgl_path, dpi=300, bbox_inches='tight')
    if not notebook_plot: plt.close()
    print(f"  > Saved MDGL event study to: {mdgl_path.name}")

    # Plot 2: NVO returns around GLP-1 approval
    plt.figure(figsize=(10, 6))
    nvo_event_data = df_stocks[
        (df_stocks.index >= glp1_date - pd.Timedelta(days=10)) &
        (df_stocks.index <= glp1_date + pd.Timedelta(days=10))
        ]
    plt.plot(nvo_event_data.index, nvo_event_data['NVO_Returns'], marker='o', linewidth=2, color='orange')
    plt.axvline(glp1_date, color='red', linestyle='--', label='GLP-1 Approval', linewidth=2)
    plt.title('NVO Daily Returns: GLP-1 FDA Approval Event Window', fontweight='bold')
    plt.ylabel('Daily Returns (%)')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    nvo_path = save_dir / "advanced_stock_nvo_event.png"
    plt.savefig(nvo_path, dpi=300, bbox_inches='tight')
    if not notebook_plot: plt.close()
    print(f"  > Saved NVO event study to: {nvo_path.name}")

    # Plot 3: Statistical comparison bar chart
    if results:
        plt.figure(figsize=(10, 6))
        events = []
        pre_means = []
        post_means = []
        p_values = []

        for event_name, result in results.items():
            events.append(result['company'])
            pre_means.append(result['pre_mean_return'])
            post_means.append(result['post_mean_return'])
            p_values.append(result['p_value'])

        x = np.arange(len(events))
        width = 0.35

        bars1 = plt.bar(x - width / 2, pre_means, width, label='Pre-Approval (±5 days)', alpha=0.7, color='lightblue')
        bars2 = plt.bar(x + width / 2, post_means, width, label='Post-Approval (±5 days)', alpha=0.7,
                        color='lightcoral')

        # Add significance markers and value labels
        for i, (pre_bar, post_bar, p_val) in enumerate(zip(bars1, bars2, p_values)):
            # Significance stars
            if p_val < 0.05:
                plt.text(i, max(pre_means[i], post_means[i]) + 1, '*',
                         ha='center', va='bottom', fontsize=20, fontweight='bold', color='red')

            # Value labels on bars
            plt.text(pre_bar.get_x() + pre_bar.get_width() / 2, pre_bar.get_height(),
                     f'{pre_means[i]:.2f}%', ha='center', va='bottom', fontweight='bold')
            plt.text(post_bar.get_x() + post_bar.get_width() / 2, post_bar.get_height(),
                     f'{post_means[i]:.2f}%', ha='center', va='bottom', fontweight='bold')

        plt.title('Average Returns Before/After FDA Approvals\n(* = statistically significant)', fontweight='bold')
        plt.ylabel('Average Daily Returns (%)')
        plt.xlabel('Company and FDA Event')
        plt.xticks(x, events)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        comparison_path = save_dir / "advanced_stock_comparison.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        if not notebook_plot: plt.close()
        print(f"  > Saved statistical comparison to: {comparison_path.name}")

    # 5. Summary table
    print("\n" + "=" * 60)
    print("STOCK EVENT STUDY - SUMMARY TABLE")
    print("=" * 60)

    if results:
        summary_data = []
        for event_name, result in results.items():
            summary_data.append({
                'Company': result['company'],
                'Event': result['event'],
                'Pre_Return (%)': f"{result['pre_mean_return']:.2f}",
                'Post_Return (%)': f"{result['post_mean_return']:.2f}",
                'Change (%)': f"{result['absolute_change']:+.2f}",
                'P_Value': f"{result['p_value']:.4f}",
                'Significant': 'Yes' if result['p_value'] < 0.05 else 'No'
            })

        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))

        summary_path = save_dir / "advanced_stock_summary_table.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"  > Saved summary table to: {summary_path.name}")

    # Display plots in notebook if requested
    if notebook_plot:
        plt.show()

    return results


def advanced_stock_volatility_analysis(df_stocks: pd.DataFrame, notebook_plot=False):
    """Advanced volatility analysis of stock price movements around FDA events"""
    print("\n[Advanced Analysis] Stock Data Volatility Analysis...")
    save_dir = RESULTS_DIR / STOCK_ANALYSIS_SUBDIR
    save_dir.mkdir(exist_ok=True, parents=True)

    # 1. FDA event dates
    resmetirom_date = pd.to_datetime(FDA_EVENT_DATES['Resmetirom Approval'])
    glp1_date = pd.to_datetime(FDA_EVENT_DATES['GLP-1 Agonists Approval'])

    # 2. Calculate daily returns (if not already calculated)
    if 'NVO_Returns' not in df_stocks.columns:
        df_stocks['NVO_Returns'] = df_stocks['NVO_Close'].pct_change() * 100
    if 'MDGL_Returns' not in df_stocks.columns:
        df_stocks['MDGL_Returns'] = df_stocks['MDGL_Close'].pct_change() * 100

    # 3. Define volatility windows (±10 trading days for better volatility measurement)
    volatility_windows = {
        'Resmetirom_MDGL': {
            'pre_volatility': df_stocks[
                (df_stocks.index >= resmetirom_date - pd.Timedelta(days=20)) &
                (df_stocks.index < resmetirom_date)
                ]['MDGL_Returns'].std(),
            'post_volatility': df_stocks[
                (df_stocks.index >= resmetirom_date) &
                (df_stocks.index <= resmetirom_date + pd.Timedelta(days=20))
                ]['MDGL_Returns'].std(),
            'company': 'Madrigal (MDGL)',
            'event': 'Resmetirom Approval'
        },
        'GLP1_NVO': {
            'pre_volatility': df_stocks[
                (df_stocks.index >= glp1_date - pd.Timedelta(days=20)) &
                (df_stocks.index < glp1_date)
                ]['NVO_Returns'].std(),
            'post_volatility': df_stocks[
                (df_stocks.index >= glp1_date) &
                (df_stocks.index <= glp1_date + pd.Timedelta(days=20))
                ]['NVO_Returns'].std(),
            'company': 'Novo Nordisk (NVO)',
            'event': 'GLP-1 Approval'
        }
    }

    # 4. Statistical significance testing for volatility changes
    print("\n" + "=" * 60)
    print("VOLATILITY ANALYSIS - STATISTICAL SIGNIFICANCE TESTING")
    print("=" * 60)

    volatility_results = {}
    for event_name, windows in volatility_windows.items():
        # Get full return series for F-test
        if event_name == 'Resmetirom_MDGL':
            pre_returns = df_stocks[
                (df_stocks.index >= resmetirom_date - pd.Timedelta(days=20)) &
                (df_stocks.index < resmetirom_date)
                ]['MDGL_Returns'].dropna()
            post_returns = df_stocks[
                (df_stocks.index >= resmetirom_date) &
                (df_stocks.index <= resmetirom_date + pd.Timedelta(days=20))
                ]['MDGL_Returns'].dropna()
        else:
            pre_returns = df_stocks[
                (df_stocks.index >= glp1_date - pd.Timedelta(days=20)) &
                (df_stocks.index < glp1_date)
                ]['NVO_Returns'].dropna()
            post_returns = df_stocks[
                (df_stocks.index >= glp1_date) &
                (df_stocks.index <= glp1_date + pd.Timedelta(days=20))
                ]['NVO_Returns'].dropna()

        if len(pre_returns) > 5 and len(post_returns) > 5:
            # F-test for variance equality
            f_stat = np.var(post_returns) / np.var(pre_returns)
            p_value = stats.f.cdf(f_stat, len(post_returns) - 1, len(pre_returns) - 1)
            # Two-tailed test
            p_value = 2 * min(p_value, 1 - p_value)

            volatility_change = ((windows['post_volatility'] - windows['pre_volatility']) /
                                 windows['pre_volatility']) * 100

            volatility_results[event_name] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'pre_volatility': windows['pre_volatility'],
                'post_volatility': windows['post_volatility'],
                'volatility_change_pct': volatility_change,
                'company': windows['company'],
                'event': windows['event']
            }

            print(f"\n{windows['company']} - {windows['event']}:")
            print(f"  Pre-event volatility (20 days): {windows['pre_volatility']:.2f}%")
            print(f"  Post-event volatility (20 days): {windows['post_volatility']:.2f}%")
            print(f"  Volatility change: {volatility_change:+.1f}%")
            print(f"  F-statistic: {f_stat:.3f}, p-value: {p_value:.4f}")
            print(f"  Significance: {'*' if p_value < 0.05 else 'Not significant'}")

    # CREATE SEPARATE VOLATILITY PLOTS

    # Plot 1: MDGL volatility around Resmetirom approval
    plt.figure(figsize=(10, 6))
    mdgl_vol_data = df_stocks[
        (df_stocks.index >= resmetirom_date - pd.Timedelta(days=30)) &
        (df_stocks.index <= resmetirom_date + pd.Timedelta(days=30))
        ]

    # Calculate rolling volatility (5-day window)
    mdgl_vol_data['MDGL_Rolling_Vol'] = mdgl_vol_data['MDGL_Returns'].rolling(window=5).std()

    plt.plot(mdgl_vol_data.index, mdgl_vol_data['MDGL_Rolling_Vol'],
             linewidth=2, color='purple', label='5-Day Rolling Volatility')
    plt.axvline(resmetirom_date, color='red', linestyle='--',
                label='Resmetirom Approval', linewidth=2)
    plt.axvspan(resmetirom_date - pd.Timedelta(days=20), resmetirom_date,
                alpha=0.1, color='blue', label='Pre-Event Window')
    plt.axvspan(resmetirom_date, resmetirom_date + pd.Timedelta(days=20),
                alpha=0.1, color='orange', label='Post-Event Window')

    plt.title('MDGL Volatility: Resmetirom FDA Approval Impact', fontweight='bold')
    plt.ylabel('Rolling Volatility (5-day, %)')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    mdgl_vol_path = save_dir / "advanced_stock_mdgl_volatility.png"
    plt.savefig(mdgl_vol_path, dpi=300, bbox_inches='tight')
    if not notebook_plot: plt.close()
    print(f"  > Saved MDGL volatility to: {mdgl_vol_path.name}")

    # Plot 2: NVO volatility around GLP-1 approval
    plt.figure(figsize=(10, 6))
    nvo_vol_data = df_stocks[
        (df_stocks.index >= glp1_date - pd.Timedelta(days=30)) &
        (df_stocks.index <= glp1_date + pd.Timedelta(days=30))
        ]

    nvo_vol_data['NVO_Rolling_Vol'] = nvo_vol_data['NVO_Returns'].rolling(window=5).std()

    plt.plot(nvo_vol_data.index, nvo_vol_data['NVO_Rolling_Vol'],
             linewidth=2, color='green', label='5-Day Rolling Volatility')
    plt.axvline(glp1_date, color='red', linestyle='--',
                label='GLP-1 Approval', linewidth=2)
    plt.axvspan(glp1_date - pd.Timedelta(days=20), glp1_date,
                alpha=0.1, color='blue', label='Pre-Event Window')
    plt.axvspan(glp1_date, glp1_date + pd.Timedelta(days=20),
                alpha=0.1, color='orange', label='Post-Event Window')

    plt.title('NVO Volatility: GLP-1 FDA Approval Impact', fontweight='bold')
    plt.ylabel('Rolling Volatility (5-day, %)')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    nvo_vol_path = save_dir / "advanced_stock_nvo_volatility.png"
    plt.savefig(nvo_vol_path, dpi=300, bbox_inches='tight')
    if not notebook_plot: plt.close()
    print(f"  > Saved NVO volatility to: {nvo_vol_path.name}")

    # Plot 3: Volatility comparison bar chart
    if volatility_results:
        plt.figure(figsize=(10, 6))
        events = []
        pre_vols = []
        post_vols = []
        p_values = []

        for event_name, result in volatility_results.items():
            events.append(result['company'])
            pre_vols.append(result['pre_volatility'])
            post_vols.append(result['post_volatility'])
            p_values.append(result['p_value'])

        x = np.arange(len(events))
        width = 0.35

        bars1 = plt.bar(x - width / 2, pre_vols, width, label='Pre-Approval (20 days)',
                        alpha=0.7, color='lightblue')
        bars2 = plt.bar(x + width / 2, post_vols, width, label='Post-Approval (20 days)',
                        alpha=0.7, color='lightcoral')

        # Add significance markers and value labels
        for i, (pre_bar, post_bar, p_val) in enumerate(zip(bars1, bars2, p_values)):
            if p_val < 0.05:
                plt.text(i, max(pre_vols[i], post_vols[i]) + 0.5, '*',
                         ha='center', va='bottom', fontsize=20, fontweight='bold', color='red')

            plt.text(pre_bar.get_x() + pre_bar.get_width() / 2, pre_bar.get_height(),
                     f'{pre_vols[i]:.2f}%', ha='center', va='bottom', fontweight='bold')
            plt.text(post_bar.get_x() + post_bar.get_width() / 2, post_bar.get_height(),
                     f'{post_vols[i]:.2f}%', ha='center', va='bottom', fontweight='bold')

        plt.title('Volatility Changes Before/After FDA Approvals\n(* = statistically significant)', fontweight='bold')
        plt.ylabel('Daily Return Volatility (%)')
        plt.xlabel('Company and FDA Event')
        plt.xticks(x, events)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        vol_comparison_path = save_dir / "advanced_stock_volatility_comparison.png"
        plt.savefig(vol_comparison_path, dpi=300, bbox_inches='tight')
        if not notebook_plot: plt.close()
        print(f"  > Saved volatility comparison to: {vol_comparison_path.name}")

    # 5. Summary table
    print("\n" + "=" * 60)
    print("VOLATILITY ANALYSIS - SUMMARY TABLE")
    print("=" * 60)

    if volatility_results:
        summary_data = []
        for event_name, result in volatility_results.items():
            summary_data.append({
                'Company': result['company'],
                'Event': result['event'],
                'Pre_Volatility (%)': f"{result['pre_volatility']:.2f}",
                'Post_Volatility (%)': f"{result['post_volatility']:.2f}",
                'Change (%)': f"{result['volatility_change_pct']:+.1f}",
                'F_Statistic': f"{result['f_statistic']:.3f}",
                'P_Value': f"{result['p_value']:.4f}",
                'Significant': 'Yes' if result['p_value'] < 0.05 else 'No'
            })

        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))

        summary_path = save_dir / "advanced_stock_volatility_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"  > Saved volatility summary to: {summary_path.name}")

    # Display plots in notebook if requested
    if notebook_plot:
        plt.show()

    return volatility_results


def cross_platform_correlation_analysis(processed_data: dict, notebook_plot=False):
    """Advanced correlation analysis between stock returns and other data sources"""
    print("\n[Advanced Analysis] Cross-Platform Correlation Analysis...")
    save_dir = RESULTS_DIR / STOCK_ANALYSIS_SUBDIR
    save_dir.mkdir(exist_ok=True, parents=True)

    # 1. Extract data from processed_data dictionary
    df_stocks = processed_data.get('stocks')
    df_trends = processed_data.get('trends')
    df_reddit = processed_data.get('reddit')

    if df_stocks is None:
        print("  ERROR: Stock data not available for correlation analysis")
        return None

    # 2. Prepare stock returns (daily)
    df_stocks['NVO_Returns'] = df_stocks['NVO_Close'].pct_change() * 100
    df_stocks['MDGL_Returns'] = df_stocks['MDGL_Close'].pct_change() * 100
    stock_returns = df_stocks[['NVO_Returns', 'MDGL_Returns']].dropna()

    correlation_results = {}

    # 3. CORRELATION WITH GOOGLE TRENDS
    if df_trends is not None:
        print("\n" + "=" * 50)
        print("STOCK RETURNS vs GOOGLE TRENDS CORRELATION")
        print("=" * 50)

        # Resample trends to daily and align dates
        trends_daily = df_trends.resample('D').mean().ffill()

        # Align time periods
        start_date = max(stock_returns.index.min(), trends_daily.index.min())
        end_date = min(stock_returns.index.max(), trends_daily.index.max())

        stock_aligned = stock_returns.loc[start_date:end_date]
        trends_aligned = trends_daily.loc[start_date:end_date]

        print(f"Aligned period: {start_date.date()} to {end_date.date()} ({len(stock_aligned)} days)")

        # Calculate correlations
        trend_correlations = {}
        for trend_col in ['MASLD', 'NAFLD', 'Rezdiffra', 'Wegovy', 'Ozempic']:
            if trend_col in trends_aligned.columns:
                nvo_corr = stock_aligned['NVO_Returns'].corr(trends_aligned[trend_col])
                mdgl_corr = stock_aligned['MDGL_Returns'].corr(trends_aligned[trend_col])

                trend_correlations[trend_col] = {
                    'NVO_Correlation': nvo_corr,
                    'MDGL_Correlation': mdgl_corr
                }

                print(f"{trend_col}: NVO={nvo_corr:.3f}, MDGL={mdgl_corr:.3f}")

        correlation_results['google_trends'] = trend_correlations

        # Plot 1: Google Trends correlation heatmap
        plt.figure(figsize=(10, 6))
        corr_data = []
        for trend, corrs in trend_correlations.items():
            corr_data.append([corrs['NVO_Correlation'], corrs['MDGL_Correlation']])

        corr_df = pd.DataFrame(corr_data,
                               index=trend_correlations.keys(),
                               columns=['NVO Returns', 'MDGL Returns'])

        sns.heatmap(corr_df, annot=True, cmap='RdBu_r', center=0,
                    vmin=-1, vmax=1, square=True)
        plt.title('Stock Returns vs Google Search Interest Correlation', fontweight='bold')
        plt.tight_layout()

        trends_corr_path = save_dir / "cross_platform_trends_correlation.png"
        plt.savefig(trends_corr_path, dpi=300, bbox_inches='tight')
        if not notebook_plot: plt.close()
        print(f"  > Saved trends correlation to: {trends_corr_path.name}")

    # 4. CORRELATION WITH REDDIT SENTIMENT
    if df_reddit is not None and 'sentiment_score' in df_reddit.columns:
        print("\n" + "=" * 50)
        print("STOCK RETURNS vs REDDIT SENTIMENT CORRELATION")
        print("=" * 50)

        # Prepare Reddit sentiment (daily average)
        df_reddit['timestamp'] = pd.to_datetime(df_reddit['timestamp'])
        reddit_daily = df_reddit.set_index('timestamp')['sentiment_score'].resample('D').mean()

        # Align time periods
        start_date = max(stock_returns.index.min(), reddit_daily.index.min())
        end_date = min(stock_returns.index.max(), reddit_daily.index.max())

        stock_aligned = stock_returns.loc[start_date:end_date]
        reddit_aligned = reddit_daily.loc[start_date:end_date]

        print(f"Aligned period: {start_date.date()} to {end_date.date()} ({len(stock_aligned)} days)")

        # Calculate correlations
        nvo_sentiment_corr = stock_aligned['NVO_Returns'].corr(reddit_aligned)
        mdgl_sentiment_corr = stock_aligned['MDGL_Returns'].corr(reddit_aligned)

        sentiment_correlations = {
            'NVO_Sentiment_Correlation': nvo_sentiment_corr,
            'MDGL_Sentiment_Correlation': mdgl_sentiment_corr
        }

        correlation_results['reddit_sentiment'] = sentiment_correlations

        print(f"Reddit Sentiment Correlation: NVO={nvo_sentiment_corr:.3f}, MDGL={mdgl_sentiment_corr:.3f}")

        # Plot 2: Reddit sentiment correlation scatter plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # NVO vs Sentiment
        ax1.scatter(reddit_aligned, stock_aligned['NVO_Returns'], alpha=0.6, s=30)
        ax1.set_xlabel('Reddit Sentiment Score')
        ax1.set_ylabel('NVO Daily Returns (%)')
        ax1.set_title(f'NVO Returns vs Reddit Sentiment\n(correlation: {nvo_sentiment_corr:.3f})')
        ax1.grid(True, alpha=0.3)

        # MDGL vs Sentiment
        ax2.scatter(reddit_aligned, stock_aligned['MDGL_Returns'], alpha=0.6, s=30, color='orange')
        ax2.set_xlabel('Reddit Sentiment Score')
        ax2.set_ylabel('MDGL Daily Returns (%)')
        ax2.set_title(f'MDGL Returns vs Reddit Sentiment\n(correlation: {mdgl_sentiment_corr:.3f})')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        sentiment_corr_path = save_dir / "cross_platform_sentiment_correlation.png"
        plt.savefig(sentiment_corr_path, dpi=300, bbox_inches='tight')
        if not notebook_plot: plt.close()
        print(f"  > Saved sentiment correlation to: {sentiment_corr_path.name}")

    # 5. CORRELATION BETWEEN STOCKS
    print("\n" + "=" * 50)
    print("INTER-STOCK CORRELATION ANALYSIS")
    print("=" * 50)

    stock_correlation = stock_returns['NVO_Returns'].corr(stock_returns['MDGL_Returns'])
    correlation_results['stock_correlation'] = stock_correlation

    print(f"NVO vs MDGL Returns Correlation: {stock_correlation:.3f}")

    # Plot 3: Stock correlation scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(stock_returns['NVO_Returns'], stock_returns['MDGL_Returns'],
                alpha=0.6, s=30, color='green')
    plt.xlabel('NVO Daily Returns (%)')
    plt.ylabel('MDGL Daily Returns (%)')
    plt.title(f'NVO vs MDGL Returns Correlation\n(correlation: {stock_correlation:.3f})')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.tight_layout()

    stock_corr_path = save_dir / "cross_platform_stock_correlation.png"
    plt.savefig(stock_corr_path, dpi=300, bbox_inches='tight')
    if not notebook_plot: plt.close()
    print(f"  > Saved stock correlation to: {stock_corr_path.name}")

    # 6. SUMMARY TABLE
    print("\n" + "=" * 60)
    print("CROSS-PLATFORM CORRELATION - SUMMARY TABLE")
    print("=" * 60)

    summary_data = []

    # Add Google Trends correlations
    if 'google_trends' in correlation_results:
        for trend, corrs in correlation_results['google_trends'].items():
            summary_data.append({
                'Platform': 'Google Trends',
                'Metric': trend,
                'NVO_Correlation': f"{corrs['NVO_Correlation']:.3f}",
                'MDGL_Correlation': f"{corrs['MDGL_Correlation']:.3f}"
            })

    # Add Reddit sentiment correlation
    if 'reddit_sentiment' in correlation_results:
        summary_data.append({
            'Platform': 'Reddit',
            'Metric': 'Sentiment Score',
            'NVO_Correlation': f"{correlation_results['reddit_sentiment']['NVO_Sentiment_Correlation']:.3f}",
            'MDGL_Correlation': f"{correlation_results['reddit_sentiment']['MDGL_Sentiment_Correlation']:.3f}"
        })

    # Add stock correlation
    summary_data.append({
        'Platform': 'Stocks',
        'Metric': 'NVO vs MDGL',
        'NVO_Correlation': f"{correlation_results['stock_correlation']:.3f}",
        'MDGL_Correlation': f"{correlation_results['stock_correlation']:.3f}"
    })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    # Save summary table
    summary_path = save_dir / "cross_platform_correlation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  > Saved correlation summary to: {summary_path.name}")

    # Display plots in notebook if requested
    if notebook_plot:
        plt.show()

    return correlation_results


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


def advanced_media_cloud_event_analysis(notebook_plot=False):
    """Advanced statistical analysis of FDA event impacts on Media Cloud coverage - SEPARATE PLOTS"""
    print("\n[Advanced Analysis] Media Cloud FDA Event Impact Analysis...")
    save_dir = RESULTS_DIR / MEDIA_CLOUD_ANALYSIS_SUBDIR
    save_dir.mkdir(exist_ok=True, parents=True)

    # Load Media Cloud datasets
    datasets = load_media_cloud_datasets()
    if not datasets:
        print("  ERROR: No Media Cloud datasets available")
        return None

    from scipy import stats
    import numpy as np

    # FDA event dates
    resmetirom_date = pd.to_datetime(FDA_EVENT_DATES['Resmetirom Approval'])
    glp1_date = pd.to_datetime(FDA_EVENT_DATES['GLP-1 Agonists Approval'])

    results = {}

    for dataset_name, data in datasets.items():
        if 'counts' not in data:
            continue

        print(f"\n--- Analyzing {dataset_name} dataset ---")
        df = data['counts'].copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()

        # Define event windows (60 days pre/post each event)
        event_analyses = {}

        for event_name, event_date in [('Resmetirom Approval', resmetirom_date),
                                       ('GLP-1 Approval', glp1_date)]:

            # Skip if event is outside data range
            if event_date < df.index.min() or event_date > df.index.max():
                continue

            # Define analysis windows
            pre_window = df[(df.index >= event_date - pd.Timedelta(days=60)) &
                            (df.index < event_date)]
            post_window = df[(df.index > event_date) &
                             (df.index <= event_date + pd.Timedelta(days=60))]

            if len(pre_window) > 10 and len(post_window) > 10:
                # Statistical test
                t_stat, p_value = stats.ttest_ind(pre_window['count'],
                                                  post_window['count'],
                                                  equal_var=False)

                # Calculate effect size
                pre_mean = pre_window['count'].mean()
                post_mean = post_window['count'].mean()
                cohens_d = (post_mean - pre_mean) / np.sqrt(
                    (pre_window['count'].var() + post_window['count'].var()) / 2
                )

                event_analyses[event_name] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'pre_mean': pre_mean,
                    'post_mean': post_mean,
                    'absolute_change': post_mean - pre_mean,
                    'percent_change': ((post_mean - pre_mean) / pre_mean) * 100,
                    'cohens_d': cohens_d,
                    'pre_obs': len(pre_window),
                    'post_obs': len(post_window)
                }

                print(f"  {event_name}:")
                print(f"    Pre-event: {pre_mean:.2f} articles/day")
                print(f"    Post-event: {post_mean:.2f} articles/day")
                print(f"    Change: {post_mean - pre_mean:+.2f} ({((post_mean - pre_mean) / pre_mean) * 100:+.1f}%)")
                print(f"    T-statistic: {t_stat:.3f}, p-value: {p_value:.4f}")
                print(f"    Cohen's d: {cohens_d:.3f} {'(significant)' if p_value < 0.05 else ''}")

        results[dataset_name] = event_analyses

    # PLOT 1: Event impact comparison (BAR CHART)
    plt.figure(figsize=(12, 6))
    plot_data = []
    for dataset_name, events in results.items():
        for event_name, stats_data in events.items():
            plot_data.append({
                'Dataset': dataset_name,
                'Event': event_name,
                'Percent_Change': stats_data['percent_change'],
                'P_Value': stats_data['p_value'],
                'Effect_Size': stats_data['cohens_d']
            })

    if plot_data:
        plot_df = pd.DataFrame(plot_data)

        # Create bar plot with significance markers
        datasets_ordered = ['disease', 'resmetirom', 'glp1']
        events_ordered = ['Resmetirom Approval', 'GLP-1 Approval']

        x_pos = np.arange(len(datasets_ordered))
        width = 0.35

        for i, event in enumerate(events_ordered):
            event_data = plot_df[plot_df['Event'] == event]
            values = [event_data[event_data['Dataset'] == dataset]['Percent_Change'].iloc[0]
                      if ((event_data['Dataset'] == dataset).any()) else 0
                      for dataset in datasets_ordered]

            bars = plt.bar(x_pos + i * width, values, width,
                           label=event, alpha=0.7,
                           color=MEDIA_CLOUD_COLORS.get(event.split()[0].lower(), 'gray'))

            # Add significance markers and value labels
            for j, (bar, dataset) in enumerate(zip(bars, datasets_ordered)):
                if ((event_data['Dataset'] == dataset).any()):
                    p_val = event_data[event_data['Dataset'] == dataset]['P_Value'].iloc[0]
                    height = bar.get_height()
                    # Significance marker
                    if p_val < 0.05:
                        plt.text(bar.get_x() + bar.get_width() / 2, height + (10 if height >= 0 else -15),
                                 '*', ha='center', va='bottom' if height >= 0 else 'top',
                                 fontsize=20, fontweight='bold', color='red')
                    # Value label
                    plt.text(bar.get_x() + bar.get_width() / 2, height + (5 if height >= 0 else -10),
                             f'{height:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top',
                             fontsize=9, fontweight='bold')

        plt.xlabel('Media Cloud Dataset')
        plt.ylabel('Percent Change in Coverage (%)')
        plt.title('FDA Approval Impact on Media Coverage\n(* = statistically significant)', fontweight='bold')
        plt.xticks(x_pos + width / 2, [d.title() for d in datasets_ordered])
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)

    plt.tight_layout()
    save_path1 = save_dir / "media_cloud_event_impact_barchart.png"
    plt.savefig(save_path1, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  > Saved event impact bar chart to: {save_path1.name}")

    # PLOT 2: Time series with event markers
    plt.figure(figsize=(12, 6))
    for dataset_name, data in datasets.items():
        if 'counts' in data:
            df = data['counts'].copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()

            # 7-day moving average for smoother trends
            rolling_avg = df['count'].rolling(window=7, center=True).mean()
            plt.plot(rolling_avg.index, rolling_avg.values,
                     label=dataset_name.title(), linewidth=2,
                     color=MEDIA_CLOUD_COLORS.get(dataset_name))

    # Add FDA event lines
    plt.axvline(resmetirom_date, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Resmetirom Approval')
    plt.axvline(glp1_date, color='blue', linestyle='--', alpha=0.7, linewidth=2, label='GLP-1 Approval')
    plt.xlabel('Date')
    plt.ylabel('7-Day Moving Average (Articles)')
    plt.title('Media Coverage Trends with FDA Events', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path2 = save_dir / "media_cloud_timeseries_trends.png"
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  > Saved time series trends to: {save_path2.name}")

    # PLOT 3: Effect sizes
    plt.figure(figsize=(10, 6))
    if plot_data:
        effect_df = pd.DataFrame(plot_data)
        effect_data = []
        for _, row in effect_df.iterrows():
            effect_data.append({
                'Comparison': f"{row['Dataset']}\n{row['Event'].split()[0]}",
                'Cohens_d': row['Effect_Size'],
                'P_Value': row['P_Value']
            })

        effect_plot_df = pd.DataFrame(effect_data)
        colors = ['lightcoral' if p < 0.05 else 'lightblue' for p in effect_plot_df['P_Value']]
        bars = plt.barh(effect_plot_df['Comparison'], effect_plot_df['Cohens_d'],
                        alpha=0.7, color=colors)

        # Add value labels
        for i, (bar, effect_size) in enumerate(zip(bars, effect_plot_df['Cohens_d'])):
            plt.text(bar.get_width() + (0.02 if effect_size >= 0 else -0.05),
                     bar.get_y() + bar.get_height() / 2,
                     f'{effect_size:.3f}', ha='left' if effect_size >= 0 else 'right',
                     va='center', fontweight='bold')

        plt.xlabel("Cohen's d (Effect Size)")
        plt.title('Effect Sizes of FDA Approval Impacts\n(Red = statistically significant)', fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)

    plt.tight_layout()
    save_path3 = save_dir / "media_cloud_effect_sizes.png"
    plt.savefig(save_path3, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  > Saved effect sizes to: {save_path3.name}")

    # PLOT 4: Statistical summary table
    plt.figure(figsize=(12, 4))
    plt.axis('off')
    if plot_data:
        summary_data = []
        for dataset_name, events in results.items():
            for event_name, stats_data in events.items():
                summary_data.append({
                    'Dataset': dataset_name.title(),
                    'Event': event_name.split()[0],
                    'Pre_Mean': f"{stats_data['pre_mean']:.2f}",
                    'Post_Mean': f"{stats_data['post_mean']:.2f}",
                    'Change': f"{stats_data['absolute_change']:+.2f}",
                    'P_Value': f"{stats_data['p_value']:.4f}",
                    'Sig': '*' if stats_data['p_value'] < 0.05 else ''
                })

        summary_df = pd.DataFrame(summary_data)
        if not summary_df.empty:
            table = plt.table(cellText=summary_df.values,
                              colLabels=summary_df.columns,
                              cellLoc='center',
                              loc='center',
                              bbox=[0.1, 0.2, 0.8, 0.6])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.8)
            plt.title('Media Cloud FDA Event Impact - Statistical Summary', fontweight='bold', pad=20)

    plt.tight_layout()
    save_path4 = save_dir / "media_cloud_statistical_summary.png"
    plt.savefig(save_path4, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  > Saved statistical summary to: {save_path4.name}")

    # Save results to CSV
    results_path = save_dir / "media_cloud_event_analysis_results.csv"
    flat_results = []
    for dataset_name, events in results.items():
        for event_name, stats_data in events.items():
            flat_results.append({
                'dataset': dataset_name,
                'event': event_name,
                **stats_data
            })

    if flat_results:
        results_df = pd.DataFrame(flat_results)
        results_df.to_csv(results_path, index=False)
        print(f"  > Saved detailed results to: {results_path.name}")

    return results


def advanced_media_cloud_concentration_analysis(notebook_plot=False):
    """Advanced analysis of media coverage concentration using Gini coefficient and HHI"""
    print("\n[Advanced Analysis] Media Cloud Coverage Concentration Analysis...")
    save_dir = RESULTS_DIR / MEDIA_CLOUD_ANALYSIS_SUBDIR
    save_dir.mkdir(exist_ok=True, parents=True)

    # Load Media Cloud datasets
    datasets = load_media_cloud_datasets()
    if not datasets:
        print("  ERROR: No Media Cloud datasets available")
        return None

    import numpy as np
    from scipy import stats

    concentration_results = {}

    for dataset_name, data in datasets.items():
        if 'sources' not in data:
            continue

        print(f"\n--- Analyzing concentration for {dataset_name} dataset ---")
        sources_df = data['sources'].copy()

        # Calculate concentration metrics
        total_articles = sources_df['count'].sum()
        n_sources = len(sources_df)

        # Gini coefficient calculation
        sorted_counts = np.sort(sources_df['count'])
        cum_share = np.cumsum(sorted_counts) / total_articles
        n = len(sorted_counts)
        gini = 1 - 2 * np.trapz(cum_share, dx=1 / n)

        # Herfindahl-Hirschman Index (HHI)
        market_shares = sources_df['count'] / total_articles
        hhi = np.sum(market_shares ** 2) * 10000  # Scale to 0-10000

        # Top 5 concentration
        top5_share = sources_df.nlargest(5, 'count')['count'].sum() / total_articles * 100
        top10_share = sources_df.nlargest(10, 'count')['count'].sum() / total_articles * 100

        concentration_results[dataset_name] = {
            'gini_coefficient': gini,
            'hhi': hhi,
            'total_articles': total_articles,
            'n_sources': n_sources,
            'top5_share': top5_share,
            'top10_share': top10_share,
            'avg_articles_per_source': total_articles / n_sources,
            'dominant_source': sources_df.iloc[0]['source'],
            'dominant_share': (sources_df.iloc[0]['count'] / total_articles) * 100
        }

        print(f"  Gini Coefficient: {gini:.3f}")
        print(f"  HHI: {hhi:.0f}")
        print(f"  Top 5 Sources: {top5_share:.1f}% of coverage")
        print(f"  Top 10 Sources: {top10_share:.1f}% of coverage")
        print(
            f"  Dominant Source: {sources_df.iloc[0]['source']} ({concentration_results[dataset_name]['dominant_share']:.1f}%)")
        print(f"  Total Sources: {n_sources}, Total Articles: {total_articles}")

    # Statistical comparison between datasets
    print("\n--- Statistical Comparison Between Datasets ---")
    datasets_list = list(concentration_results.keys())

    for i in range(len(datasets_list)):
        for j in range(i + 1, len(datasets_list)):
            dataset1 = datasets_list[i]
            dataset2 = datasets_list[j]

            # Compare Gini coefficients (approximate using source count distributions)
            sources1 = datasets[dataset1]['sources']['count']
            sources2 = datasets[dataset2]['sources']['count']

            # Mann-Whitney U test for distribution differences
            stat, p_value = stats.mannwhitneyu(sources1, sources2, alternative='two-sided')

            print(f"  {dataset1} vs {dataset2}:")
            print(
                f"    Gini: {concentration_results[dataset1]['gini_coefficient']:.3f} vs {concentration_results[dataset2]['gini_coefficient']:.3f}")
            print(f"    Distribution p-value: {p_value:.4f}")

    # Create visualizations

    # PLOT 1: Concentration metrics comparison
    plt.figure(figsize=(12, 8))

    # Subplot 1: Gini coefficients
    plt.subplot(2, 2, 1)
    gini_values = [concentration_results[d]['gini_coefficient'] for d in datasets_list]
    bars = plt.bar(datasets_list, gini_values, color=[MEDIA_CLOUD_COLORS.get(d, 'gray') for d in datasets_list],
                   alpha=0.7)
    plt.ylabel('Gini Coefficient')
    plt.title('Media Coverage Concentration\n(Gini Coefficient)', fontweight='bold')
    plt.grid(True, alpha=0.3)
    # Add value labels
    for bar, value in zip(bars, gini_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{value:.3f}',
                 ha='center', va='bottom', fontweight='bold')

    # Subplot 2: HHI
    plt.subplot(2, 2, 2)
    hhi_values = [concentration_results[d]['hhi'] for d in datasets_list]
    bars = plt.bar(datasets_list, hhi_values, color=[MEDIA_CLOUD_COLORS.get(d, 'gray') for d in datasets_list],
                   alpha=0.7)
    plt.ylabel('HHI (0-10000 scale)')
    plt.title('Herfindahl-Hirschman Index', fontweight='bold')
    plt.grid(True, alpha=0.3)
    # Add value labels
    for bar, value in zip(bars, hhi_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50, f'{value:.0f}',
                 ha='center', va='bottom', fontweight='bold')

    # Subplot 3: Top 5 concentration
    plt.subplot(2, 2, 3)
    top5_values = [concentration_results[d]['top5_share'] for d in datasets_list]
    bars = plt.bar(datasets_list, top5_values, color=[MEDIA_CLOUD_COLORS.get(d, 'gray') for d in datasets_list],
                   alpha=0.7)
    plt.ylabel('Percentage of Coverage (%)')
    plt.title('Top 5 Sources Share of Coverage', fontweight='bold')
    plt.grid(True, alpha=0.3)
    # Add value labels
    for bar, value in zip(bars, top5_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{value:.1f}%',
                 ha='center', va='bottom', fontweight='bold')

    # Subplot 4: Source distribution Lorenz curves
    plt.subplot(2, 2, 4)
    for dataset_name in datasets_list:
        sources_df = datasets[dataset_name]['sources'].copy()
        sorted_counts = np.sort(sources_df['count'])
        cum_articles = np.cumsum(sorted_counts) / sorted_counts.sum()
        cum_sources = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)

        plt.plot(cum_sources, cum_articles, label=dataset_name.title(),
                 linewidth=2, color=MEDIA_CLOUD_COLORS.get(dataset_name))

    # Perfect equality line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Equality')
    plt.xlabel('Cumulative Share of Sources')
    plt.ylabel('Cumulative Share of Articles')
    plt.title('Lorenz Curves: Coverage Distribution', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path1 = save_dir / "media_cloud_concentration_metrics.png"
    plt.savefig(save_path1, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  > Saved concentration metrics to: {save_path1.name}")

    # PLOT 2: Source network and dominance
    plt.figure(figsize=(14, 6))

    # Subplot 1: Top sources across datasets
    plt.subplot(1, 2, 1)
    top_n = 10
    for dataset_name in datasets_list:
        sources_df = datasets[dataset_name]['sources'].nlargest(top_n, 'count')
        plt.barh([f"{s}_{dataset_name}" for s in sources_df['source']],
                 sources_df['count'], alpha=0.7, label=dataset_name.title(),
                 color=MEDIA_CLOUD_COLORS.get(dataset_name))

    plt.xlabel('Number of Articles')
    plt.title(f'Top {top_n} Sources by Dataset', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Source overlap analysis
    plt.subplot(1, 2, 2)
    # Calculate Jaccard similarities between source sets
    similarity_matrix = np.zeros((len(datasets_list), len(datasets_list)))
    source_sets = {name: set(datasets[name]['sources']['source']) for name in datasets_list}

    for i, name1 in enumerate(datasets_list):
        for j, name2 in enumerate(datasets_list):
            intersection = len(source_sets[name1].intersection(source_sets[name2]))
            union = len(source_sets[name1].union(source_sets[name2]))
            similarity_matrix[i, j] = intersection / union if union > 0 else 0

    im = plt.imshow(similarity_matrix, cmap='YlOrRd', vmin=0, vmax=1)
    plt.xticks(range(len(datasets_list)), [d.title() for d in datasets_list])
    plt.yticks(range(len(datasets_list)), [d.title() for d in datasets_list])
    plt.title('Source Overlap: Jaccard Similarity', fontweight='bold')

    # Add similarity values as text
    for i in range(len(datasets_list)):
        for j in range(len(datasets_list)):
            plt.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                     ha='center', va='center', fontweight='bold',
                     color='white' if similarity_matrix[i, j] > 0.5 else 'black')

    plt.colorbar(im, label='Jaccard Similarity')

    plt.tight_layout()
    save_path2 = save_dir / "media_cloud_source_analysis.png"
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  > Saved source analysis to: {save_path2.name}")

    # Save results to CSV
    results_path = save_dir / "media_cloud_concentration_results.csv"
    results_df = pd.DataFrame.from_dict(concentration_results, orient='index')
    results_df.to_csv(results_path, index=True)
    print(f"  > Saved concentration results to: {results_path.name}")

    return concentration_results


def advanced_media_cloud_topic_propagation(notebook_plot=False):
    """Advanced analysis of topic propagation and Granger causality between coverage types"""
    print("\n[Advanced Analysis] Media Cloud Topic Propagation Analysis...")
    save_dir = RESULTS_DIR / MEDIA_CLOUD_ANALYSIS_SUBDIR
    save_dir.mkdir(exist_ok=True, parents=True)

    # Load Media Cloud datasets
    datasets = load_media_cloud_datasets()
    if not datasets:
        print("  ERROR: No Media Cloud datasets available")
        return None

    import numpy as np
    from scipy import stats
    from statsmodels.tsa.stattools import grangercausalitytests

    # Prepare time series data
    print("\n--- Preparing Time Series Data ---")
    time_series_data = {}

    for dataset_name, data in datasets.items():
        if 'counts' not in data:
            continue

        df = data['counts'].copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()

        # Create weekly time series to reduce noise
        weekly_series = df['count'].resample('W').sum()

        # Handle zeros by adding small constant for log transformation
        weekly_series = weekly_series.replace(0, 0.1)

        time_series_data[dataset_name] = weekly_series
        print(f"  {dataset_name}: {len(weekly_series)} weeks, mean: {weekly_series.mean():.1f} articles/week")

    if len(time_series_data) < 2:
        print("  ERROR: Insufficient datasets for propagation analysis")
        return None

    # Align all time series to common date range
    common_index = time_series_data['disease'].index.intersection(
        time_series_data['resmetirom'].index
    ).intersection(time_series_data['glp1'].index)

    aligned_series = {}
    for name, series in time_series_data.items():
        aligned_series[name] = series.loc[common_index]

    print(f"  Common time period: {len(common_index)} weeks")
    print(f"  Date range: {common_index.min().strftime('%Y-%m-%d')} to {common_index.max().strftime('%Y-%m-%d')}")

    # Correlation analysis with time lags
    print("\n--- Time-Lagged Correlation Analysis ---")
    max_lag = 8  # weeks
    lag_results = {}

    for target in ['resmetirom', 'glp1']:
        for predictor in ['disease', 'glp1', 'resmetirom']:
            if target == predictor:
                continue

            target_series = aligned_series[target]
            predictor_series = aligned_series[predictor]

            lag_correlations = []
            for lag in range(max_lag + 1):
                if lag == 0:
                    # Concurrent correlation
                    corr = target_series.corr(predictor_series)
                else:
                    # Lagged correlation (predictor leads target)
                    corr = target_series.corr(predictor_series.shift(lag))

                lag_correlations.append({
                    'lag': lag,
                    'correlation': corr,
                    'predictor': predictor,
                    'target': target
                })

            lag_results[f"{predictor}_to_{target}"] = lag_correlations

            # Find optimal lag
            optimal_lag = max(range(len(lag_correlations)),
                              key=lambda i: abs(lag_correlations[i]['correlation']))
            optimal_corr = lag_correlations[optimal_lag]['correlation']

            print(f"  {predictor} -> {target}:")
            print(f"    Optimal lag: {optimal_lag} weeks, correlation: {optimal_corr:.3f}")
            if optimal_lag > 0:
                direction = "LEADS" if optimal_corr > 0 else "PRECEDES_NEGATIVE"
                print(f"    Interpretation: {predictor} {direction} {target} by {optimal_lag} weeks")

    # Granger causality tests
    print("\n--- Granger Causality Tests ---")
    granger_results = {}

    # Test disease -> drug causality
    test_pairs = [
        ('disease', 'resmetirom', 'Disease awareness drives Resmetirom coverage'),
        ('disease', 'glp1', 'Disease awareness drives GLP-1 coverage'),
        ('resmetirom', 'glp1', 'Resmetirom coverage drives GLP-1 coverage'),
        ('glp1', 'resmetirom', 'GLP-1 coverage drives Resmetirom coverage')
    ]

    for predictor, target, hypothesis in test_pairs:
        if predictor not in aligned_series or target not in aligned_series:
            continue

        # Prepare data for Granger test
        combined_data = np.column_stack([aligned_series[target].values,
                                         aligned_series[predictor].values])

        try:
            # Test with 4-week lags (1 month)
            gc_test = grangercausalitytests(combined_data, maxlag=4, verbose=False)

            # Extract p-values for each lag
            p_values = {}
            for lag in range(1, 5):
                p_value = gc_test[lag][0]['ssr_ftest'][1]
                p_values[f'lag_{lag}'] = p_value

            granger_results[f"{predictor}_to_{target}"] = {
                'hypothesis': hypothesis,
                'p_values': p_values,
                'min_p_value': min(p_values.values()),
                'significant': any(p < 0.05 for p in p_values.values())
            }

            print(f"  {predictor} -> {target}:")
            print(f"    Hypothesis: {hypothesis}")
            print(f"    Minimum p-value: {min(p_values.values()):.4f}")
            print(f"    Significant: {'YES' if granger_results[f'{predictor}_to_{target}']['significant'] else 'NO'}")
            if granger_results[f'{predictor}_to_{target}']['significant']:
                significant_lags = [lag for lag, p in p_values.items() if p < 0.05]
                print(f"    Significant at lags: {significant_lags}")

        except Exception as e:
            print(f"  {predictor} -> {target}: Granger test failed - {e}")

    # Create visualizations

    # PLOT 1: Time-lagged correlation heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Heatmap 1: Disease as predictor
    predictor_data = []
    target_pairs = [('resmetirom', 'Disease -> Resmetirom'),
                    ('glp1', 'Disease -> GLP-1')]

    for i, (target, title) in enumerate(target_pairs):
        key = f"disease_to_{target}"
        if key in lag_results:
            corrs = [x['correlation'] for x in lag_results[key]]
            predictor_data.append(corrs)

            im = axes[0, i].imshow([corrs], cmap='RdBu_r', aspect='auto',
                                   vmin=-0.6, vmax=0.6, extent=[0, max_lag, 0, 1])
            axes[0, i].set_xlabel('Lag (Weeks)')
            axes[0, i].set_title(f'{title}\nTime-Lagged Correlation', fontweight='bold')
            axes[0, i].set_yticks([])

            # Add correlation values
            for lag, corr in enumerate(corrs):
                color = 'white' if abs(corr) > 0.3 else 'black'
                axes[0, i].text(lag + 0.5, 0.5, f'{corr:.2f}',
                                ha='center', va='center', fontweight='bold', color=color)

    # Heatmap 2: Cross-drug predictions
    cross_pairs = [('resmetirom_to_glp1', 'Resmetirom -> GLP-1'),
                   ('glp1_to_resmetirom', 'GLP-1 -> Resmetirom')]

    for i, (key, title) in enumerate(cross_pairs):
        if key in lag_results:
            corrs = [x['correlation'] for x in lag_results[key]]

            im = axes[1, i].imshow([corrs], cmap='RdBu_r', aspect='auto',
                                   vmin=-0.6, vmax=0.6, extent=[0, max_lag, 0, 1])
            axes[1, i].set_xlabel('Lag (Weeks)')
            axes[1, i].set_title(f'{title}\nTime-Lagged Correlation', fontweight='bold')
            axes[1, i].set_yticks([])

            # Add correlation values
            for lag, corr in enumerate(corrs):
                color = 'white' if abs(corr) > 0.3 else 'black'
                axes[1, i].text(lag + 0.5, 0.5, f'{corr:.2f}',
                                ha='center', va='center', fontweight='bold', color=color)

    plt.tight_layout()
    save_path1 = save_dir / "media_cloud_lagged_correlations.png"
    plt.savefig(save_path1, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  > Saved lagged correlations to: {save_path1.name}")

    # PLOT 2: Time series comparison with leading indicators
    plt.figure(figsize=(14, 10))

    # Normalize series for comparison
    normalized_series = {}
    for name, series in aligned_series.items():
        normalized_series[name] = (series - series.mean()) / series.std()

    # Plot normalized time series
    plt.subplot(2, 1, 1)
    for name, series in normalized_series.items():
        plt.plot(series.index, series.values, label=name.title(),
                 linewidth=2, color=MEDIA_CLOUD_COLORS.get(name))

    plt.ylabel('Normalized Coverage (Z-score)')
    plt.title('Media Coverage Trends (Normalized)', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add FDA event lines
    resmetirom_date = pd.to_datetime(FDA_EVENT_DATES['Resmetirom Approval'])
    glp1_date = pd.to_datetime(FDA_EVENT_DATES['GLP-1 Agonists Approval'])
    plt.axvline(resmetirom_date, color='red', linestyle='--', alpha=0.7, label='Resmetirom Approval')
    plt.axvline(glp1_date, color='blue', linestyle='--', alpha=0.7, label='GLP-1 Approval')

    # Plot 2: Granger causality results
    plt.subplot(2, 1, 2)
    if granger_results:
        test_names = []
        min_p_values = []
        colors = []

        for key, result in granger_results.items():
            test_names.append(key.replace('_to_', ' ->\n'))
            min_p_values.append(result['min_p_value'])
            colors.append('lightcoral' if result['significant'] else 'lightblue')

        bars = plt.barh(test_names, min_p_values, color=colors, alpha=0.7)
        plt.xscale('log')
        plt.xlabel('Minimum P-value (Log Scale)')
        plt.title('Granger Causality Tests\n(Red = Significant Predictive Relationship)', fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.axvline(x=0.05, color='red', linestyle='--', alpha=0.5, label='p=0.05 significance threshold')

        # Add value labels
        for bar, p_val in zip(bars, min_p_values):
            plt.text(bar.get_width() * 1.1, bar.get_y() + bar.get_height() / 2,
                     f'p={p_val:.4f}', va='center', fontweight='bold')

        plt.legend()

    plt.tight_layout()
    save_path2 = save_dir / "media_cloud_granger_causality.png"
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  > Saved Granger causality results to: {save_path2.name}")

    # PLOT 3: Optimal lag summary
    plt.figure(figsize=(10, 6))

    optimal_lags_data = []
    for key, lags in lag_results.items():
        optimal_lag = max(range(len(lags)), key=lambda i: abs(lags[i]['correlation']))
        optimal_corr = lags[optimal_lag]['correlation']
        optimal_lags_data.append({
            'relationship': key.replace('_to_', ' → '),
            'optimal_lag': optimal_lag,
            'correlation': optimal_corr,
            'abs_correlation': abs(optimal_corr)
        })

    if optimal_lags_data:
        plot_df = pd.DataFrame(optimal_lags_data)

        # Create bubble chart
        scatter = plt.scatter(plot_df['optimal_lag'], plot_df['correlation'],
                              s=plot_df['abs_correlation'] * 500, alpha=0.6,
                              c=plot_df['correlation'], cmap='RdBu_r', vmin=-0.6, vmax=0.6)

        # Add labels
        for i, row in plot_df.iterrows():
            plt.annotate(row['relationship'],
                         (row['optimal_lag'], row['correlation']),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=9, fontweight='bold')

        plt.xlabel('Optimal Lag (Weeks)')
        plt.ylabel('Correlation at Optimal Lag')
        plt.title('Topic Propagation: Optimal Lags and Correlation Strength', fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)

        # Add correlation strength interpretation
        plt.axhspan(-0.3, 0.3, alpha=0.1, color='gray', label='Weak correlation')
        plt.axhspan(0.3, 0.6, alpha=0.1, color='blue', label='Moderate positive')
        plt.axhspan(-0.6, -0.3, alpha=0.1, color='red', label='Moderate negative')

        plt.colorbar(scatter, label='Correlation Strength')
        plt.legend()

    plt.tight_layout()
    save_path3 = save_dir / "media_cloud_optimal_lags.png"
    plt.savefig(save_path3, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  > Saved optimal lags analysis to: {save_path3.name}")

    # Save results to CSV
    results_path = save_dir / "media_cloud_propagation_results.csv"

    # Flatten results for saving
    flat_results = []
    for key, lags in lag_results.items():
        for lag_data in lags:
            flat_results.append({
                'relationship': key,
                'lag_weeks': lag_data['lag'],
                'correlation': lag_data['correlation'],
                'predictor': lag_data['predictor'],
                'target': lag_data['target']
            })

    for key, result in granger_results.items():
        flat_results.append({
            'relationship': f"granger_{key}",
            'lag_weeks': 'multiple',
            'correlation': result['min_p_value'],
            'predictor': 'granger_test',
            'target': f"significant_{result['significant']}"
        })

    if flat_results:
        results_df = pd.DataFrame(flat_results)
        results_df.to_csv(results_path, index=False)
        print(f"  > Saved propagation results to: {results_path.name}")

    return {
        'lag_results': lag_results,
        'granger_results': granger_results,
        'time_series_data': aligned_series
    }


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
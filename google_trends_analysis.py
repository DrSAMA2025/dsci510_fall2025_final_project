# Google Trends Analysis for MASLD Awareness Tracker

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import seaborn as sns

# Set style for publication-quality visuals
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=== MASLD AWARENESS TRACKER - GOOGLE TRENDS ANALYSIS ===")

# Read and prepare the data
df = pd.read_csv('data/google_trends_initial_data.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

print(f"\nData Overview:")
print(f"Time period: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
print(f"Total data points: {len(df)}")
print(f"Keywords: {', '.join(df.columns)}")

# Define key FDA approval dates
FDA_EVENTS = {
    'Resmetirom (Rezdiffra) Approval': '2024-03-14',
    'GLP-1 Agonists Approval': '2025-08-15'
}

# 1. MAIN TRENDS VISUALIZATION
print("\n📈 Creating main trends visualization...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('MASLD Awareness: Google Trends Analysis (2023-2025)', fontsize=16, fontweight='bold')

# Plot 1: MASLD vs NAFLD comparison
axes[0, 0].plot(df.index, df['MASLD'], label='MASLD', linewidth=2.5, color='#2E86AB', marker='o', markersize=3)
axes[0, 0].plot(df.index, df['NAFLD'], label='NAFLD', linewidth=2.5, color='#A23B72', alpha=0.8)
axes[0, 0].set_title('Terminology Transition: MASLD vs NAFLD', fontweight='bold')
axes[0, 0].set_ylabel('Search Interest Score')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Add FDA approval markers
for event, date in FDA_EVENTS.items():
    axes[0, 0].axvline(pd.to_datetime(date), color='red', linestyle='--', alpha=0.7, label=f'{event}')
    axes[0, 0].text(pd.to_datetime(date), axes[0, 0].get_ylim()[1] * 0.9, event.split(' ')[0],
                    rotation=90, verticalalignment='top')

# Plot 2: All keywords trend
for column in df.columns:
    axes[0, 1].plot(df.index, df[column], label=column, linewidth=2, alpha=0.8)
axes[0, 1].set_title('All Keywords Search Trends', fontweight='bold')
axes[0, 1].set_ylabel('Search Interest Score')
axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: MASLD/NAFLD ratio (terminology adoption)
df['MASLD_NAFLD_Ratio'] = df['MASLD'] / (df['NAFLD'] + 0.1)  # Avoid division by zero
axes[1, 0].plot(df.index, df['MASLD_NAFLD_Ratio'], linewidth=2.5, color='#F18F01')
axes[1, 0].axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Equal Interest (Ratio=1)')
axes[1, 0].set_title('MASLD/NAFLD Search Interest Ratio', fontweight='bold')
axes[1, 0].set_ylabel('Ratio (MASLD/NAFLD)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Focus on August 2025 spike
august_2025 = df['2025-08-01':'2025-09-30']
axes[1, 1].plot(august_2025.index, august_2025['MASLD'], label='MASLD', linewidth=3, color='#2E86AB', marker='o')
axes[1, 1].plot(august_2025.index, august_2025['NAFLD'], label='NAFLD', linewidth=3, color='#A23B72', marker='s')
axes[1, 1].set_title('August 2025: GLP-1 Approval Impact', fontweight='bold')
axes[1, 1].set_ylabel('Search Interest Score')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('google_trends_main_analysis.png', dpi=300, bbox_inches='tight')

# 2. CORRELATION ANALYSIS
print("Performing correlation analysis...")
correlation_matrix = df[['MASLD', 'NAFLD', 'Rezdiffra', 'Wegovy', 'Ozempic']].corr()

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.2f', cbar_kws={"shrink": .8})
plt.title('Keyword Correlation Matrix', fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('google_trends_correlation_heatmap.png', dpi=300, bbox_inches='tight')

# 3. EVENT STUDY ANALYSIS
print("Conducting event study analysis...")
event_results = {}

for event_name, event_date in FDA_EVENTS.items():
    event_date = pd.to_datetime(event_date)

    # Define event window (4 weeks before and after)
    pre_event = df[event_date - pd.Timedelta(weeks=4):event_date - pd.Timedelta(days=1)]
    post_event = df[event_date:event_date + pd.Timedelta(weeks=4)]

    if len(pre_event) > 0 and len(post_event) > 0:
        event_results[event_name] = {}
        for keyword in ['MASLD', 'NAFLD', 'Wegovy', 'Ozempic']:
            pre_mean = pre_event[keyword].mean()
            post_mean = post_event[keyword].mean()
            percent_change = ((post_mean - pre_mean) / pre_mean * 100) if pre_mean > 0 else 0

            event_results[event_name][keyword] = {
                'pre_mean': pre_mean,
                'post_mean': post_mean,
                'percent_change': percent_change
            }

# 4. STATISTICAL SUMMARY
print("Generating statistical summary...")
summary_stats = df[['MASLD', 'NAFLD', 'Rezdiffra', 'Wegovy', 'Ozempic']].describe()

# 5. OUTPUT RESULTS
print("\n" + "=" * 60)
print("KEY FINDINGS SUMMARY")
print("=" * 60)

print(f"\nTerminology Transition:")
masld_first_appearance = df[df['MASLD'] > 0].index[0]
print(f"• MASLD first appeared: {masld_first_appearance.strftime('%Y-%m-%d')}")
print(f"• Current MASLD/NAFLD ratio: {df['MASLD_NAFLD_Ratio'].iloc[-1]:.2f}")

print(f"\nPharmaceutical Trends:")
print(f"• Wegovy peak interest: {df['Wegovy'].max()} (on {df['Wegovy'].idxmax().strftime('%Y-%m-%d')})")
print(f"• Ozempic peak interest: {df['Ozempic'].max()} (on {df['Ozempic'].idxmax().strftime('%Y-%m-%d')})")

print(f"\nAugust 2025 Spike Analysis:")
august_spike = df['2025-08-01':'2025-08-31']
max_masld = august_spike['MASLD'].max()
max_nafld = august_spike['NAFLD'].max()
print(f"• MASLD peak: {max_masld} ({max_masld / df['MASLD'].mean():.1f}x average)")
print(f"• NAFLD peak: {max_nafld} ({max_nafld / df['NAFLD'].mean():.1f}x average)")

print(f"\nKey Correlations:")
print(f"• MASLD vs NAFLD: {correlation_matrix.loc['MASLD', 'NAFLD']:.3f}")
print(f"• MASLD vs Wegovy: {correlation_matrix.loc['MASLD', 'Wegovy']:.3f}")
print(f"• MASLD vs Ozempic: {correlation_matrix.loc['MASLD', 'Ozempic']:.3f}")

print(f"\nEvent Study Results:")
for event, results in event_results.items():
    print(f"\n{event}:")
    for drug, stats in results.items():
        if abs(stats['percent_change']) > 10:  # Only show significant changes
            direction = "↑" if stats['percent_change'] > 0 else "↓"
            print(f"  • {drug}: {direction} {abs(stats['percent_change']):.1f}% change")

print(f"\nAnalysis complete! Files saved:")
print(f"   • google_trends_main_analysis.png")
print(f"   • google_trends_correlation_heatmap.png")
print(f"   • Statistical summary available in variable 'summary_stats'")

# Show the plots
plt.show()

# Save summary statistics to CSV
summary_stats.to_csv('google_trends_statistical_summary.csv')
print(f"   • google_trends_statistical_summary.csv")
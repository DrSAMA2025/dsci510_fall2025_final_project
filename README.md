# MASLD Awareness Tracker - Final Project

## Project Overview
This project tracks and analyzes public and scientific awareness of Metabolic Dysfunction-Associated Steatotic Liver Disease (MASLD) in relation to major regulatory milestones to provide insights into the dynamics of health information dissemination and public engagement with emerging medical treatments.

## Task 1: Google Trends Data Collection & Analysis

### Objective
Collect and analyze historical search interest data to track public awareness of MASLD terminology adoption and monitor the impact of FDA drug approvals (Resmetirom - March 2024, GLP-1 agonists - August 2025) on public search behavior.

### Implementation
- **Script:** `google_trends.py`
- **Timeframe:** January 1, 2023 - October 28, 2025
- **Keywords:** MASLD, NAFLD, Rezdiffra, Wegovy, Ozempic

### Output
- `google_trends_initial_data.csv` - Weekly search interest data

## Google Trends Analysis - COMPLETED

### Status Update
- **Data Collection:** Successfully collected 142 weeks of search data (2023-2025)
- **Analysis:** Comprehensive trends, correlation, and FDA event study analysis completed
- **Visualizations:** Professional multi-panel charts and correlation heatmaps generated
- **Key Files:** `google_trends_analysis.py`, visualization PNGs, statistical summaries

### Key Findings
- MASLD search interest emerged June 2023, showing gradual terminology adoption
- Massive search spike in August 2025 around GLP-1 agonist approvals
- Strong correlations between MASLD awareness and GLP-1 drug searches (r=0.68)
- FDA approvals drive measurable increases in public disease awareness

### Deliverables 
- `google_trends.py` - Data collection script
- `google_trends_initial_data.csv` - Raw search data (142 data points)
- `google_trends_analysis.py` - Comprehensive analysis script
- `google_trends_main_analysis.png` - Trends visualization
- `google_trends_correlation_heatmap.png` - Keyword relationships

## 🔄 Task 2: Reddit Data Collection & Sentiment Analysis - IN PROGRESS
*Data collection and sentiment analysis pending*

## 📚 Task 3: PubMed Publications Analysis - PENDING
*Academic publication trends analysis pending*

## 🛠️ Setup Instructions
```bash
pip install pytrends pandas matplotlib seaborn

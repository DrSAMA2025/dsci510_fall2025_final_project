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
- **Google Trends Analysis - COMPLETED**

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

## 🔄 Task 2: Reddit Data Collection & Sentiment Analysis

### Objective
Collect and analyze public discussions from Reddit communities to measure sentiment and discourse patterns around MASLD, track community reactions to FDA drug approvals (Resmetirom - March 2024, GLP-1 agonists - August 2025), and compare patient versus medical professional perspectives.

### Implementation
**Script:** `reddit_data_collector.py`  
**Timeframe:** January 1, 2023 - October 28, 2025  
**Subreddits:** r/NAFLD, r/MASH, r/Ozempic, r/Wegovy, r/semaglutide, r/obesity, plus medical forums (r/medicine, r/pharmacy, r/AskDocs)  
**Search Strategy:** Comprehensive keyword targeting including disease terminology (MASLD, NAFLD, NASH, MASH) and medications (Resmetirom, Rezdiffra, Semaglutide, Ozempic, Wegovy)

### Output
`reddit_data_2023_2025_20251101_1404.csv` - 9,255 posts and comments with metadata  
**Reddit Analysis - COMPLETED**

### Status Update
**Data Collection:** Successfully collected 9,255 Reddit records (1,039 posts + 8,216 comments)  
**Analysis:** Comprehensive sentiment analysis using VADER, daily sentiment tracking, subreddit comparisons, and FDA event impact assessment  
**Visualizations:** Professional timeline charts, sentiment distributions, and community comparisons  
**Key Files:** `reddit_sentiment_analysis.py`, visualization PNGs, statistical tables, comprehensive report

### Key Findings
- **Overall Positive Sentiment:** 0.199 average score across all discussions
- **Strong Medical Professional Engagement:** r/pharmacy (2,004 records) and r/medicine (876 records) provided quality insights
- **Active Patient Communities:** r/Ozempic (1,877 records) and r/semaglutide (820 records) showed high engagement
- **Terminology Transition:** MASLD mentions emerging but NAFLD still dominant in patient discussions
- **FDA Impact Visible:** Sentiment trends show reactions around approval dates

### Deliverables
- `reddit_data_collector.py` - Comprehensive data collection script with optimized search terms
- `reddit_sentiment_analysis.py` - Complete sentiment analysis and visualization pipeline
- `tests.py` - Data quality and analysis validation tests
- `reddit_data_2023_2025_20251101_1404.csv` - Raw dataset (9,255 records)
- Multiple analysis outputs in `/analysis` folder (tables, plots, reports)

## 📚 Task 3: PubMed Publications Analysis - PENDING
*Academic publication trends analysis pending*


## 📈 Task 5: Stock Market Data Analysis 

### Objective
Analyze stock market reactions to FDA drug approvals by tracking pharmaceutical companies (Novo Nordisk - GLP-1 agonists, Madrigal Pharmaceuticals - Resmetirom) to measure industry and investor responses to regulatory milestones and correlate with public awareness trends.

### Implementation
- **Script**: `stock_data.py`
- **Timeframe**: January 3, 2023 - October 27, 2025 (707 trading days)
- **Companies**: Novo Nordisk (NVO), Madrigal Pharmaceuticals (MDGL)
- **FDA Dates**: Resmetirom (March 14, 2024), Semaglutide (August 15, 2025)

### Output
- `stock_prices.csv` - Daily stock prices, volumes, and market data
- **Stock Data Analysis - COMPLETED**

### Status Update
- **Data Collection**: Successfully downloaded 707 trading days of historical data
- **Validation**: Comprehensive data integrity tests passed
- **Coverage**: Both FDA approval dates captured in dataset
- **Analysis**: Comprehensive FDA impact study and correlation analysis completed
- **Visualizations**: Professional presentation-ready charts and correlation matrices generated
- **Key Files**: `stock_data.py`, `stock_prices.csv`, `stock_analysis.py`, enhanced visualization suite, statistical results

### Key Findings
- **FDA Approval Impact**: Madrigal -10.8% immediate drop, Novo Nordisk +2.9% gain
- **Strong Correlation**: Madrigal stock vs MASLD searches (r = 0.751) - retail investor awareness drives market performance
- **Market Sentiment**: Pure-play MASLD companies more sensitive to public disease awareness than diversified pharma
- **Data Quality**: Complete historical coverage of both FDA approval events with comprehensive validation

### Deliverables
- `stock_data.py` - Automated data collection and validation script
- `stock_analysis.py` - Enhanced analysis with presentation-ready visualizations
- `data/stock_prices.csv` - Complete historical stock data
- `analysis/stock_price_trend_presentation.png` - Professional normalized price trends
- `analysis/correlation_matrix_presentation.png` - Comprehensive correlation analysis
- `analysis/stock_analysis_results.csv` - FDA impact quantitative results
- `analysis/stock_correlation_results.csv` - Correlation statistics
- Integrated testing in `tests.py` - Data quality assurance


## 🛠️ Setup Instructions
```bash
pip install pytrends pandas matplotlib seaborn
pip install pytrends pandas matplotlib seaborn requests praw vaderSentiment

### Required Packages
- **pytrends**: Google Trends API access
- **pandas**: Data manipulation and analysis
- **matplotlib**: Data visualization and plotting
- **seaborn**: Enhanced statistical visualizations
- **requests**: HTTP requests for web APIs
- **praw**: Python Reddit API Wrapper
- **vaderSentiment**: Sentiment analysis for social media text

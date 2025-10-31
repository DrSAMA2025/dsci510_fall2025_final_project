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

## 🔄 Task 2: Reddit Data Collection & Sentiment Analysis - IN PROGRESS
*Data collection and sentiment analysis pending*

## 📚 Task 3: PubMed Publications Analysis - PENDING
*Academic publication trends analysis pending*



## 📈 Task 5: Stock Market Data Analysis 

### Objective
Analyze stock market reactions to FDA drug approvals by tracking pharmaceutical companies (Novo Nordisk - GLP-1 agonists, Madrigal Pharmaceuticals - Resmetirom) to measure industry and investor responses to regulatory milestones.

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
- **Key Files**: `stock_data.py`, `stock_prices.csv`, integrated test suite

### Key Findings
- Complete historical price data for both companies across study period
- FDA approval dates successfully captured in trading data
- Realistic price ranges verified: NVO ($45.38-$144.04), MDGL ($120.40-$458.66)
- Data quality confirmed with comprehensive validation tests

### Deliverables
- `stock_data.py` - Automated data collection and validation script
- `data/stock_prices.csv` - Complete historical stock data
- Integrated testing in `tests.py` - Data quality assurance


## 🛠️ Setup Instructions
```bash
pip install pytrends pandas matplotlib seaborn

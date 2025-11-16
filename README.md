# MASLD Awareness Tracker
*Evaluating the Impact of FDA Approvals of Resmetirom and GLP-1 Agonists on Public Awareness of Metabolic Dysfunction-Associated Stearotic Liver Disease*

## Data Sources

| Data Source | Description | Type | Fields |
|-------------|-------------|------|--------|
| Google Trends | Public search interest for MASLD, NAFLD, Rezdiffra, Wegovy, Ozempic | API | Term, date, interest index, region |
| Reddit | Public discussions from r/NAFLD, r/MASH, r/Ozempic, r/Wegovy, r/medicine, r/pharmacy | API | Subreddit, post title, text, timestamp, comments, sentiment scores |
| PubMed | Scientific publications on MASLD/NAFLD and related drug treatments | API | PubMed ID, title, abstract, publication date, journal |
| Yahoo Finance | Stock data for Novo Nordisk (NVO) and Madrigal Pharmaceuticals (MDGL) | API | Date, closing price, volume, company ticker |
| Media Cloud | News media coverage analysis from various sources | File | Article content, publication dates, sources, topics |

## Results
- **Successfully implemented multi-source data pipeline** collecting data from 5 different platforms
- **Automated data processing** with sentiment analysis for Reddit data and timeline standardization
- **Basic visualization framework** showing trends across Google Trends, Reddit sentiment, PubMed publications, stock prices, and media coverage
- **FDA event integration** with standardized event markers across all timeline analyses
- **Robust error handling** with Google Drive fallback for pre-collected datasets

## Installation

### API Keys and Environment Setup
Create a `.env` file in the `src/` directory with the following variables:

Reddit API Credentials (Required for Reddit data collection)
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here
REDDIT_USER_AGENT=dsci510_final_project_v1.0

### Pre-collected Data Access
For testing without API calls, pre-collected datasets are available via Google Drive links configured in `config.py`. This ensures project functionality even when API rate limits are encountered.

### Special Python Packages Used
- **praw**: Reddit API data collection
- **yfinance**: Stock market data for pharmaceutical companies  
- **pytrends**: Google Trends search interest data
- **biopython**: PubMed scientific literature access
- **vaderSentiment**: Reddit comment sentiment analysis
- **gdown**: Google Drive data download functionality
- **python-dotenv**: Secure environment variable management

Install all packages:
```bash
pip install -r requirements.txt
```

## Running Analysis

### Basic Execution
From the `src/` directory, run the complete pipeline:
```bash
python main.py 
```
### Advanced Options
- Use existing data only: `python main.py --analysis-only`
- Skip Reddit/Pubmed APIs: `python main.py --skip-reddit --skip-pubmed`
- Quick test run: `python main.py --quick`
- Run tests: `python tests.py`

### Testing
Run comprehensive tests to validate all data sources and API connectivity:
```bash
python src/tests.py
```
This test suite validates API connections, data quality, and Google Drive fallback functionality across all 5 data sources.

### Project Structure
- **src/**: Source code directory
  - `load.py`: Data collection from APIs
  - `process.py`: Data cleaning and processing  
  - `analyze.py`: Analysis and visualization
  - `config.py`: Configuration and constants
  - `tests.py`: Test suite validation
  - `main.py`: Main execution script
  - `results.ipynb`: Interactive analysis notebook
- **data/**: Collected datasets (excluded from Git)
- **results/**: Analysis outputs (excluded from Git) 
- **doc/**: Final project progress report
- **requirements.txt**: Python dependencies
- **README.md**: Project documentation

### Output Locations
- **Results**: All analysis outputs will be generated in organized subfolders within `/results/` when running the pipeline
- **Data**: All collected data will be stored in `/data/` folder during execution
- **Logs**: Execution details printed to console
- **Interactive Analysis**: See `src/results.ipynb` for detailed analysis and visualizations.

## (1) Google Trends Analysis
### Google Trends Analysis
- **Search Interest Tracking**: Weekly relative search volume for MASLD, NAFLD, Rezdiffra, Wegovy, and Ozempic (2023-2025)
- **FDA Event Impact Analysis**: Statistical testing of search behavior changes around Resmetirom and GLP-1 approval dates
- **Terminology Transition Monitoring**: MASLD/NAFLD ratio analysis to track disease nomenclature adoption
- **Correlation Analysis**: Relationship mapping between disease awareness and drug search interest
- **Statistical Significance Testing**: T-tests with p-values to validate observed changes in search patterns

### Summary of the Results
- **Comprehensive search interest tracking** for MASLD, NAFLD, Rezdiffra, Wegovy, and Ozempic across 148 weeks (2023-2025)
- **Basic time series visualization** showing search trends with FDA approval event markers for both Resmetirom and GLP-1 agonists
- **Advanced statistical analysis** revealing significant impact of GLP-1 approval on MASLD awareness (+0.503 points, p=0.009)
- **Market saturation effects** detected with significant decreases in Wegovy (-1.382 points, p=0.041) and Ozempic (-7.912 points, p=0.006) search interest post-approval
- **Data quality insights** showing MASLD and Rezdiffra have very low public search volume (91-93% zero values) compared to established GLP-1 drugs
- **Dual visualization approach** with both basic EDA plots and advanced statistical charts automatically saved and displayed

### How to Run Google Trends Analysis
### Data Collection
```bash
# Automatic API collection
python src/load.py          # Fetches Google Trends data via pytrends API

# Manual execution
from src.load import get_google_trends_data
trends_data = get_google_trends_data()
``` 
### Fallback Option
Pre-collected Google Trends dataset automatically downloads from Google Drive if API fails

### Analysis Execution
```bash
# Complete analysis pipeline
python src/main.py

# Google Trends specific analysis  
python src/analyze.py       # Generates basic and advanced visualizations

# Jupyter notebook exploration
jupyter notebook src/results.ipynb
```
### Output
- **Basic Analysis**: `results/google_trends/google_trends_basic_analysis.png` - Timeline with FDA events
- **Advanced Analysis**: `results/google_trends/advanced_google_trends_analysis.png` - Statistical results and correlations  
- **Statistical Summary**: Console output with p-values and significance markers for both FDA events

## (2) Reddit Analysis

### Reddit Analysis Components
- **Basic EDA**: Comprehensive exploratory data analysis including post volume trends, subreddit distribution, and text length analysis
- **Sentiment Analysis**: VADER sentiment scoring of Reddit posts and comments with FDA event tracking
- **Topic Modeling**: NMF-based topic discovery across 5 distinct discussion themes
- **Temporal Pattern Analysis**: Hourly, daily, and monthly activity patterns with FDA impact quantification
- **Network Analysis**: Subreddit relationship mapping and community detection
- **Cross-Platform Correlation**: Reddit discussion volume vs. Google Trends search interest

### Summary of the Results
- **Basic EDA insights** revealed uneven subreddit distribution with r/Supplements and r/fattyliver dominating discussion volume
- **Text analysis** showed varied post lengths with comprehensive patient discussions averaging substantial content depth
- **Comprehensive sentiment tracking** across 9,146 Reddit posts showing stable sentiment around FDA approvals (no significant changes, p>0.05)
- **Topic modeling revealed 5 key discussion themes**: Personal experiences (22.8%), Information & concerns (21.3%), Private discussions (19.8%), Liver disease discussions (13.7%), and Drug treatments (22.4%)
- **Temporal patterns identified** peak discussion times at 11:00 AM and Fridays, with significant GLP-1 approval impact (+82.3% discussion increase)
- **Network analysis showed fully connected MASLD community** with 12 subreddits forming a unified discussion ecosystem (density=1.000)
- **Strong cross-platform correlation** between Reddit discussions and MASLD searches (r=0.522), indicating coordinated online engagement
- **Drug treatment discussions dominated** with 2,972 posts, reflecting high community interest in pharmaceutical interventions

### How to Run Reddit Analysis

### Data Collection
```bash
# Automatic API collection
python src/load.py          # Fetches Reddit data via Pushshift API

# Manual execution
from src.load import get_reddit_data
reddit_data = get_reddit_data()
```
### Fallback Option
Pre-collected Reddit dataset automatically loads from local storage if API fails

### Analysis Execution
```bash
# Complete analysis pipeline
python src/main.py

# Reddit specific analysis
python src/analyze.py       # Generates sentiment, topic, temporal, and network analyses

# Jupyter notebook exploration
jupyter notebook src/results.ipynb
```

### Output
- **Basic EDA**: `results/reddit/reddit_basic_analysis.png` - Post volume trends, subreddit distribution, and text statistics
- **Sentiment Analysis**: `results/reddit/sentiment_analysis.png` - Sentiment timeline with FDA events
- **Topic Modeling**: `results/reddit/topic_modeling_results.png` - 5-topic distribution and key terms
- **Temporal Patterns**: `results/reddit/temporal_patterns.png` - Activity heatmaps and FDA impact charts
- **Network Analysis**: `results/reddit/network_analysis.png` - Subreddit relationship graph
- **Cross-Platform Correlation**: `results/reddit/correlation_analysis.png` - Reddit vs. Google Trends relationship

## (3) PubMed Analysis

### PubMed Analysis Components
- **Publication Trend Analysis**: Monthly publication counts for MASLD research with FDA event tracking
- **Research Focus Tracking**: Disease-drug combination analysis (MASLD + Resmetirom vs MASLD + GLP-1)
- **Journal Distribution Analysis**: Top publishing journals and research outlets
- **FDA Approval Impact Assessment**: Statistical testing of publication volume changes before/after drug approvals
- **Statistical Significance Testing**: Fisher's exact tests with p-values and odds ratios for FDA event impacts

### Summary of the Results
- **Comprehensive publication analysis** of 620 MASLD-related research articles (2023-2025)
- **Basic timeline visualization** showing disease-drug combination trends with FDA approval markers
- **Advanced statistical analysis** revealing Resmetirom approval created entirely new research area (0 publications before → 133 after, clinically significant)
- **GLP-1 research maturity** demonstrated with established publication volume pre-approval (159 publications) and statistically significant decline post-approval (odds ratio: 3.615, p=0.0000)
- **Research focus distribution** showing 213 MASLD+GLP-1 publications vs 133 MASLD+Resmetirom publications
- **Top journal identification** with leading publications in MASLD research landscape
- **Multi-plot visualization** with separate timeline, focus areas, journal distribution, and FDA impact charts

### How to Run PubMed Analysis

### Data Collection
```bash
# Automatic API collection
python src/load.py          # Fetches PubMed data via Biopython Entrez API

# Manual execution
from src.load import get_pubmed_data
pubmed_data = get_pubmed_data()
```
### Fallback Option
Pre-collected PubMed dataset automatically downloads from Google Drive if API fails

### Analysis Execution
```bash
# Complete analysis pipeline
python src/main.py

# PubMed specific analysis  
python src/analyze.py       # Generates basic and advanced visualizations

# Jupyter notebook exploration
jupyter notebook src/results.ipynb
```
### Output
- **Basic Analysis**: `results/pubmed/pubmed_drug_comparison_timeline.png` - Disease-drug combination timeline
- **Advanced Analysis**: 
  - `results/pubmed/advanced_pubmed_timeline.png` - Publication trends with FDA events
  - `results/pubmed/advanced_pubmed_focus_areas.png` - Research focus distribution
  - `results/pubmed/advanced_pubmed_top_journals.png` - Top publishing journals
  - `results/pubmed/advanced_pubmed_fda_impact.png` - FDA approval impact with significance markers
- **Statistical Summary**: Console output with p-values, odds ratios, and comprehensive summary table
- **Data Export**: `results/pubmed/advanced_pubmed_summary_table.csv` - Complete analysis results

## (4) Stock Data Analysis

### Stock Data Analysis
- **Price Movement Tracking**: Daily closing prices for Novo Nordisk (NVO) and Madrigal Pharmaceuticals (MDGL) from 2023-2025
- **FDA Event Impact Analysis**: Event study methodology to measure stock price reactions around Resmetirom and GLP-1 approval dates  
- **Cross-Platform Correlation**: Statistical analysis connecting stock returns with Google Trends search interest and Reddit sentiment
- **Volatility Analysis**: Rolling volatility measurements to assess market uncertainty around key regulatory events
- **Statistical Significance Testing**: T-tests and F-tests to validate observed changes in returns and volatility patterns

### Summary of the Results
- **Comprehensive stock price tracking** for NVO and MDGL across 705 trading days (2023-2025) with dual-axis visualization
- **Event study analysis** revealing statistically significant market reactions to FDA approvals with proper pre/post-event windows
- **Cross-platform correlation analysis** showing very weak relationships between stock returns and external data sources
- **Volatility pattern identification** with rolling 5-day volatility measurements around key regulatory events
- **Statistical validation** of all findings with p-values and significance markers for robust conclusions
- **Automated visualization pipeline** generating event study charts, correlation heatmaps, and volatility trend plots

### How to Run Stock Data Analysis

### Data Collection
```bash
# Automatic API collection
python src/load.py          # Fetches stock data via yfinance API

# Manual execution
from src.load import get_stock_data
stock_data = get_stock_data()
```
### Fallback Option
Pre-collected stock dataset automatically downloads from Google Drive if API fails

### Analysis Execution
```bash
# Complete analysis pipeline
python src/main.py

# Stock specific analysis
from src.analyze import analyze_stock_and_events, advanced_stock_analysis
analyze_stock_and_events(processed_stocks)

# Cross-platform correlation
from src.analyze import cross_platform_correlation_analysis
correlation_results = cross_platform_correlation_analysis(processed_data)

# Jupyter notebook exploration
jupyter notebook src/results.ipynb
```
### Output
- **Basic Analysis**: `results/stock_analysis/stock_vs_events_timeline.png` - Price movements with FDA event markers
- **Advanced Analysis**: `results/stock_analysis/advanced_stock_comparison.png` - Statistical event study results
- **Volatility Analysis**: `results/stock_analysis/advanced_stock_volatility_comparison.png` - Market uncertainty patterns
- **Cross-Platform Correlation**: `results/stock_analysis/cross_platform_trends_correlation.png` - Heatmap of stock-search relationships
- **Statistical Summary**: Console output with correlation coefficients and significance testing for all analyses

## (5) Media Cloud Analysis

### Media Cloud Analysis
- **Coverage Volume Tracking**: Daily article counts across disease-focused, Resmetirom-focused, and GLP-1-focused media coverage (2023-2025)
- **FDA Event Impact Analysis**: Interrupted time series analysis with statistical testing of coverage changes around Resmetirom and GLP-1 approval dates
- **Media Concentration Analysis**: Gini coefficient and Herfindahl-Hirschman Index calculations to measure coverage concentration across media outlets
- **Topic Propagation Analysis**: Time-lagged correlations and Granger causality tests to identify predictive relationships between coverage types
- **Source Network Analysis**: Jaccard similarity and source overlap quantification across coverage datasets
- **Statistical Significance Testing**: T-tests, Mann-Whitney U tests, and Granger causality with p-values to validate observed patterns

### Summary of the Results
- **Comprehensive media coverage tracking** across 1,036 days with 21,905 disease articles, 145 Resmetirom articles, and 15,263 GLP-1 articles
- **Advanced statistical event analysis** revealing significant spillover effects - Resmetirom approval increased GLP-1 coverage by 55% (p=0.0114)
- **Extreme media concentration** detected in Resmetirom coverage (Gini: 0.763, HHI: 5016) with Benzinga dominating 70.3% of coverage
- **Topic propagation insights** showing Resmetirom coverage predicts GLP-1 coverage with 3-4 week lead time (Granger causality p=0.0062)
- **Statistical validation** of all concentration differences between datasets (p<0.0001) and predictive relationships
- **Multi-faceted visualization approach** generating event impact charts, concentration metrics, Lorenz curves, and causality test results

### How to Run Media Cloud Analysis
### Data Collection
```bash
# Automatic data download
python src/load.py          # Downloads pre-collected Media Cloud data from Google Drive

# Manual execution
from src.load import get_media_cloud_data
media_data_available = get_media_cloud_data()
```
### Fallback Option
Pre-collected Media Cloud datasets automatically download from Google Drive with comprehensive coverage across three focus areas

### Analysis Execution
```bash
# Complete analysis pipeline
python src/main.py

# Media Cloud specific analyses
from src.analyze import advanced_media_cloud_event_analysis
media_cloud_results = advanced_media_cloud_event_analysis(notebook_plot=True)

from src.analyze import advanced_media_cloud_concentration_analysis  
concentration_results = advanced_media_cloud_concentration_analysis(notebook_plot=True)

from src.analyze import advanced_media_cloud_topic_propagation
propagation_results = advanced_media_cloud_topic_propagation(notebook_plot=True)

# Jupyter notebook exploration
jupyter notebook src/results.ipynb
```
### Output
- **Event Impact Analysis**: `results/media_cloud/media_cloud_event_impact_barchart.png` - FDA approval impacts with statistical significance
- **Time Series Trends**: `results/media_cloud/media_cloud_timeseries_trends.png` - Coverage trends with event markers
- **Concentration Metrics**: `results/media_cloud/media_cloud_concentration_metrics.png` - Gini coefficients and HHI across datasets
- **Source Analysis**: `results/media_cloud/media_cloud_source_analysis.png` - Top sources and overlap visualization
- **Topic Propagation**: `results/media_cloud/media_cloud_granger_causality.png` - Predictive relationship testing
- **Statistical Summary**: Console output with p-values, effect sizes, and significance markers for all analyses
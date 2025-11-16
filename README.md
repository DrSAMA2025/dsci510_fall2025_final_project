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

## (1) Google Trends
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
# MASLD Awareness Tracker
*Evaluating the Impact of FDA Approvals of Resmetirom and GLP-1 Agonists on Public Awareness of Metabolic Dysfunction-Associated Stearotic Liver Disease*

## Data Sources

| Data Source | Description | Type | Fields |
|-------------|-------------|---|--------|
| Google Trends | Public search interest for MASLD, NAFLD, Rezdiffra, Wegovy, Ozempic | API + G-Drive Fallback | Term, date, interest index, region |
| Reddit | Public discussions from r/NAFLD, r/MASH, r/Ozempic, r/Wegovy, r/medicine, r/pharmacy | API + G-Drive Fallback | Subreddit, post title, text, timestamp, comments, sentiment scores |
| PubMed | Scientific publications on MASLD/NAFLD and related drug treatments | API + G-Drive Fallback | PubMed ID, title, abstract, publication date, journal |
| Yahoo Finance | Stock data for Novo Nordisk (NVO) and Madrigal Pharmaceuticals (MDGL) | API + G-Drive Fallback | Date, closing price, volume, company ticker |
| Media Cloud | News media coverage analysis from various sources | File + G-Drive Fallback | Article content, publication dates, sources, topics |

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
For testing without API calls, pre-collected datasets are available via Google Drive links. The data loading system in `load.py` automatically uses pre-collected data when API credentials are unavailable or rate limits are encountered, ensuring project functionality under all conditions.

### Special Python Packages Used
#### Core Data Manipulation & Analysis
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing and array operations
- **scipy**: Statistical analysis and scientific computing
- **statsmodels**: Advanced statistical modeling and hypothesis testing

#### Data Collection & APIs
- **praw**: Reddit API data collection
- **yfinance**: Stock market data for pharmaceutical companies  
- **pytrends**: Google Trends search interest data
- **biopython**: PubMed scientific literature access
- **requests**: HTTP requests for web data collection

#### Machine Learning & NLP
- **scikit-learn**: NMF topic modeling and machine learning
- **vaderSentiment**: Reddit comment sentiment analysis
- **nltk**: Natural language processing toolkit
- **networkx**: Network analysis and community detection

#### Visualization
- **matplotlib**: Comprehensive plotting and visualization
- **seaborn**: Statistical data visualization

#### Infrastructure & Utilities
- **gdown**: Google Drive data download functionality
- **python-dotenv**: Secure environment variable management
- **beautifulsoup4**: Web scraping and HTML parsing

Install all packages:
```bash
pip install -r requirements.txt
```

## How to Run

### Reproducing the Analysis
1. Clone the repository and navigate to the project directory
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables: Copy `src/.env.example` to `src/.env` and add your API credentials (see instructions below)
4. Navigate to the `src/` directory: `cd src`
5. Run the complete pipeline: `python main.py`
6. For interactive exploration, open `results.ipynb` in Jupyter

### Environment Setup
Create `src/.env` file with the following structure (using the template from `src/.env.example`):
REDDIT_CLIENT_ID=""
REDDIT_CLIENT_SECRET=""
REDDIT_USER_AGENT="dsci510_final_project_v1.0"

**Note:** The project includes pre-collected datasets and will automatically use Google Drive fallback if API credentials are not available.

### Data Setup
**Option A: Automatic Download (Recommended)**
Run the pipeline once and it will automatically:
1. Create the `data/` folder
2. Download pre-collected datasets from Google Drive (if needed)
3. Proceed with analysis

```bash
cd src
python main.py
```
**Option B: Manual Download**
If you prefer manual control:
1. Download data files from Google Drive:
   - **Google Trends**: [google_trends_initial_data.csv](https://drive.google.com/file/d/1lrov39Ww1Zp2kJTu4zb1yr3Q2j69H1rX/view?usp=drive_link)
   - **Stock Data**: [stock_prices.csv](https://drive.google.com/file/d/1hHWJ85BtGBP0aJBkhgr0wjjbPWvWqZYX/view?usp=drive_link)
   - **Reddit Data**: [reddit_data_*.csv](https://drive.google.com/file/d/1atMK_8axChUJMtzw8e7iv46tEehPTSsK/view?usp=drive_link)
   - **PubMed Data**: [pubmed_masld_articles_*.csv](https://drive.google.com/file/d/1HyNZl8yxF3U9ooj9reF48MlRHheFTPEt/view?usp=drive_link)
   - **Media Cloud Data**: [media_cloud_*.csv](https://drive.google.com/file/d/1MfP0OizpCSUrqjZmMLwrCj3vS4KI0KA3/view?usp=drive_link)
2. Run `python main.py` once - it will create the `data/` folder
3. Place downloaded files in the `data/` folder
4. Run the analysis pipeline again

### Basic Execution
# From project root
python src/main.py

# Or from src directory
cd src
python main.py

### Command Line Parameters
# Use existing data only (skip API calls)
python src/main.py --analysis-only

# Skip specific APIs
python src/main.py --skip-reddit --skip-pubmed

# Quick test run with minimal data
python src/main.py --quick

# Skip system tests
python src/main.py --skip-tests

# Skip data loading (use existing data only)
python src/main.py --skip-load

### Testing
# Run comprehensive test suite
python src/tests.py

# Or run tests through main.py
python src/main.py --skip-load  # Tests will run automatically

### Interactive Analysis
jupyter notebook src/results.ipynb

### Data Pipeline
The project follows this data flow:
1. **Data Collection** (`load.py`): Fetches data from APIs with Google Drive fallback
2. **Data Processing** (`process.py`): Cleans and standardizes data across platforms
3. **Analysis** (`analyze.py`): Performs statistical analysis and generates visualizations
4. **Results** (`results.ipynb`): Interactive exploration of findings

All data loading and processing is reproducible. The pipeline will automatically use pre-collected data from Google Drive if API credentials are not available.

## Running Analysis

### Basic Execution
From the `src/` directory, run the complete pipeline:
```bash
cd src
python main.py 
```
### Advanced Options
From the `src/` directory:
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
  - `load.py`: Data collection from APIs and Google Drive fallback
  - `process.py`: Data cleaning and processing  
  - `analyze.py`: Analysis and visualization
  - `config.py`: Configuration and constants
  - `utils.py`: Helper functions and utilities
  - `tests.py`: Test suite validation
  - `main.py`: Main execution script
  - `results.ipynb`: Interactive analysis notebook
- **data/**: Collected datasets (excluded from Git)
- **results/**: Analysis outputs (excluded from Git) 
- **doc/**: Final project progress report and presentation
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
- **Gold-Standard Time Series Analysis**: Interrupted Time Series (ITS) modeling using SARIMAX to account for autocorrelation and temporal dynamics

### Summary of the Results
- **Comprehensive search interest tracking** for MASLD, NAFLD, Rezdiffra, Wegovy, and Ozempic across 148 weeks (2023-2025)
- **Basic time series visualization** showing search trends with FDA approval event markers for both Resmetirom and GLP-1 agonists
- **Data Quality Validation**: MASLD showed reliable search volume (16.9% zeros), while Rezdiffra was excluded from statistical analysis due to insufficient search volume (66.9% zeros)
- **Data quality insights** showing Rezdiffra had minimal search volume (67% zeros, mean=0.3) compared to established terms
- **Advanced statistical analysis** revealing significant increases after GLP-1 approvals: MASLD (+14.4 points, p=0.000), NAFLD (+31.5 points, p=0.001), and Wegovy (+21.8 points, p=0.000)
- **Resmetirom approval impact**: Significant increases in MASLD (+4.4 points, p=0.000) and NAFLD (+1.1 points, p=0.030) search interest
- **Strong GLP-1 impact** with Wegovy showing dramatic search increases post-approval (+21.8 points, p=0.000) while Ozempic remained stable (p=0.246)
- **Methodological rigor**: Data quality checks ensured statistical analysis focused only on reliable search terms with sufficient public interest
- **Dual visualization approach** with both basic EDA plots and advanced statistical charts automatically saved and displayed

### Gold-Standard Interrupted Time Series Findings
- **GLP-1 Agonists Drove MASLD Awareness**: MASLD showed +17.1 point immediate increase (p=0.0000) after GLP-1 approvals
- **Wegovy Demonstrated Strongest Impact**: +25.6 point surge (p=0.0011) after GLP-1 approvals, with additional +7.8 point spillover (p=0.0053) from Resmetirom approval
- **Ozempic Showed Complex Patterns**: Significant level changes after both Resmetirom (+19.6 points, p=0.0000) and GLP-1 approvals (+19.4 points, p=0.0063)
- **Resmetirom Impact More Nuanced**: No significant immediate level change for MASLD (p=0.3835) but positive slope trend (+0.032, p=0.0000)
- **Critical Insight**: GLP-1 approvals accelerated MASLD terminology adoption more effectively than the dedicated MASLD drug itself
- **Methodological Rigor**: ITS analysis provided gold-standard validation of search trend impacts

### Advanced Statistical Validation
- **Assumption Testing**: Stationarity (ADF tests), normality (Shapiro-Wilk), and variance equality (Levene's tests) validated
- **Non-Parametric Confirmation**: Mann-Whitney U tests confirmed significant findings (MASLD: p=0.0000, NAFLD: p=0.0043, Wegovy: p=0.0000)
- **Seasonal Decomposition**: Weekly pattern analysis addressing time series non-stationarity
- **Methodological Rigor**: Consistent results across parametric and non-parametric methods demonstrate robust findings
- **Gold-Standard Validation**: Interrupted Time Series (ITS) analysis using SARIMAX models provides academic-level rigor
- **Correlation Validation**: Exceptionally strong MASLD-Wegovy correlation (r=0.93) confirmed disease-drug awareness relationship

### Data Quality & Methodological Improvements
- **Data Quality Assessment**: Automated zero-analysis identified terms with insufficient search volume for reliable statistical analysis
- **Selective Statistical Testing**: Only terms with <50% zero values included in advanced statistical models (MASLD: 16.9% zeros, Rezdiffra: 66.9% zeros excluded)
- **Robust Analysis Pipeline**: Data quality checks prevent statistical artifacts and ensure scientifically valid findings
- **Transparent Limitations**: Clear acknowledgment of emerging terminology search volume limitations

### Multi-Layered Statistical Approach
- **Traditional Methods**: T-tests and correlation analysis for initial insights
- **Robust Validation**: Non-parametric tests and assumption checking
- **Gold Standard**: Interrupted Time Series (ITS) for rigorous causal inference
- **Comprehensive Validation**: Multiple statistical methods ensure robust findings

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
- **Advanced Timeline**: `results/google_trends/advanced_google_trends_timeline.png` - Search trends with statistical annotations
- **Impact Analysis**: `results/google_trends/advanced_google_trends_impact.png` - Bar chart of significant changes
- **Correlation Matrix**: `results/google_trends/advanced_google_trends_correlation.png` - Search term relationships
- **Statistical Table**: `results/google_trends/advanced_google_trends_statistical_table.png` - Comprehensive results summary
- **Seasonal Decomposition**: `results/google_trends/google_trends_seasonal_decomposition.png` - Advanced time series analysis
- **Gold-Standard ITS Analysis**: `results/google_trends/google_trends_its_analysis.png` - Interrupted Time Series visualization
- **ITS Summary**: `results/google_trends/its_analysis_summary.csv` - Complete ITS coefficients and p-values
- **Statistical Summary**: Console output with p-values and significance markers for both FDA events

## (2) Reddit Analysis

### Reddit Analysis Components
- **Basic EDA**: Comprehensive exploratory data analysis including post volume trends, subreddit distribution, and text length analysis
- **Sentiment Analysis**: VADER sentiment scoring of Reddit posts and comments with FDA event tracking
- **Topic Modeling**: NMF-based topic discovery across 5 distinct discussion themes
- **Temporal Pattern Analysis**: Hourly, daily, and monthly activity patterns with FDA impact quantification
- **Enhanced statistical validation** including normality tests, autocorrelation checks, Bonferroni correction, and effect size calculations (Cohen's d)
- **Network Analysis**: Subreddit relationship mapping and community detection
- **Cross-Platform Correlation**: Reddit discussion volume vs. Google Trends search interest
- **Methodological rigor**: All analyses include statistical assumption validation, multiple comparison correction, and effect size reporting

### Data Processing Note
Reddit data collection via Pushshift API included historical posts (some dating to 2011). Our processing pipeline in `process.py` filters all data to the study period (2023-01-01 to 2025-10-28) before analysis. This ensures only temporally relevant data is included in FDA event impact studies.

### Summary of the Results
- **Basic EDA insights** revealed pharmacy (1,993 posts), Ozempic (1,872 posts), and MASH (1,089 posts) as the most active subreddits
- **Text analysis** showed varied post lengths with comprehensive patient discussions averaging substantial content depth
- **Comprehensive sentiment tracking** across 9,097 Reddit posts (filtered to 2023-2025) showing stable sentiment around FDA approvals (no statistically significant changes, p>0.05) with small negative effect size for Resmetirom (Cohen's d = -0.305) and negligible effect for GLP-1 (Cohen's d = -0.012)
- **Topic modeling revealed 5 key discussion themes**: Personal experiences (44.0%), Drug treatments (32.7%), Liver disease discussions (13.8%), Information seeking (4.9%), and Private discussions (2.5%)
- **Temporal patterns identified** peak discussion times at 11:00 AM and Fridays, with significant GLP-1 approval impact (+82.3% discussion increase)
- **Resmetirom approval impact** showed -30.8% decrease in discussion volume (39 → 27 posts), indicating muted community response to MASLD-specific treatment
- **Network analysis showed fully connected MASLD community** with all 12 subreddits equally central in a unified discussion ecosystem (density=1.000, single community detected)
- **Limited cross-platform alignment** revealed weak correlations between Reddit discussions and search interest (MASLD r=0.344, Wegovy r=0.312, Rezdiffra r=0.272), indicating semi-independent platform behaviors
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
- **Sentiment Timeline**: `results/reddit/reddit_sentiment_with_confidence_intervals.png` - Weekly sentiment trends with FDA events
- **Advanced Sentiment Analysis**: 
  - `results/reddit/reddit_sentiment_timeline.png` - Daily sentiment with statistical events
  - `results/reddit/reddit_subreddit_sentiment.png` - Sentiment comparison across subreddits
  - `results/reddit/reddit_sentiment_distribution.png` - Sentiment score histogram
  - `results/reddit/reddit_fda_impact.png` - FDA event statistical impacts
- **Topic Modeling**: `results/reddit/reddit_topic_analysis.png` - 5-topic distribution and key terms
- **Temporal Patterns**: `results/reddit/reddit_temporal_patterns.png` - Hourly, daily, monthly activity patterns
- **Network Analysis**: `results/reddit/reddit_network_analysis.png` - Subreddit relationship graph and centrality
- **Cross-Platform Correlation**: `results/reddit/reddit_trends_correlation.png` - Reddit vs. Google Trends relationships

## (3) PubMed Analysis

### PubMed Analysis Components
- **Publication Trend Analysis**: Monthly publication counts for MASLD research with FDA event tracking
- **Research Focus Tracking**: Disease-drug combination analysis (MASLD + Resmetirom vs MASLD + GLP-1)
- **Journal Distribution Analysis**: Top publishing journals and research outlets
- **FDA Approval Impact Assessment**: Statistical testing of publication volume changes before/after drug approvals
- **Statistical Significance Testing**: Fisher's exact tests with p-values and odds ratios for FDA event impacts

### Summary of the Results
- **Comprehensive publication analysis** of 618 MASLD-related research articles (2023-2025)
- **Basic timeline visualization** showing disease-drug combination trends with FDA approval markers
- **Advanced statistical analysis** revealing Resmetirom approval created entirely new research area (0 publications before → 132 after, clinically significant)
- **GLP-1 research maturity** demonstrated with established publication volume pre-approval (157 publications) and statistically significant increase post-approval (odds ratio: 3.557, p=0.0000)
- **Research focus distribution** showing 211 MASLD+GLP-1 publications vs 132 MASLD+Resmetirom publications
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
# MASLD Awareness Tracker
*Evaluating the Impact of FDA Approvals of Resmetirom and GLP-1 Agonists on Public Awareness of Metabolic Dysfunction-Associated Stearotic Liver Disease*

## Data Sources

| Data Source | Description | Type | Fields |
|-------------|-------------|------|--------|
| Google Trends | Public search interest for MASLD, NAFLD, Rezdiffra, Wegovy, Ozempic | API | Term, date, interest index, region |
| Reddit | Public discussions from r/NAFLD, r/MASH, r/Ozempic, r/Wegovy, r/medicine, r/pharmacy | API | Subreddit, post title, text, timestamp, comments |
| PubMed | Scientific publications on MASLD/NAFLD and related drug treatments | API | PubMed ID, title, abstract, publication date, journal |
| Yahoo Finance | Stock data for Novo Nordisk (NVO) and Madrigal Pharmaceuticals (MDGL) | API | Date, closing price, volume, company ticker |
| Media Cloud | News media coverage analysis from various sources | File | Article content, publication dates, sources, topics |

## Results

### Google Trends Analysis (148 data points)
- **Search Patterns**: MASLD average interest score of 9.1 (range: 0-44), NAFLD average 26.6 (range: 16-100)
- **Strong Correlations**: MASLD vs NAFLD (0.83), MASLD vs Wegovy (0.93), MASLD vs Ozempic (0.46)
- **Public Awareness**: MASLD search interest emerged June 2023, showing gradual terminology adoption

### PubMed Scientific Literature (1,344 publications)
- **Terminology Transition**: MASLD mentioned in 400 publications (29.8%) vs NAFLD in 175 (13.0%)
- **Research Focus**: GLP-1 agonists featured in 600 publications (44.6% of total)
- **Drug Integration**: 100 publications link MASLD + Resmetirom, 94 link MASLD + GLP-1
- **Publication Growth**: 1,161 publications in 2025 vs 135 in 2024, showing exponential research interest
- **Journal Distribution**: Research spread across 620 unique journals, led by diabetes and hepatology specialists

### Reddit Community Analysis (9,255 posts/comments)
- **Overall Sentiment**: Positive average sentiment of 0.199 (53.8% positive, 26.5% negative)
- **Active Communities**: r/pharmacy (2,004 records), r/Ozempic (1,877), r/MASH (1,132), r/medicine (876)
- **Highest Sentiment**: r/MASLD (0.469), r/semaglutide (0.341), r/Ozempic (0.312)
- **Key Mentions**: Semaglutide (1,380 mentions, 14.9% of total), MASLD (786 mentions, 8.5%)

### Stock Market Analysis (707 trading days)
- **FDA Approval Impact**: Madrigal -10.77% after Resmetirom approval, Novo Nordisk +2.87% after Semaglutide approval
- **Strong Correlation**: Madrigal stock vs MASLD searches (r = 0.751)
- **Market Dynamics**: Pure-play MASLD companies more sensitive to public disease awareness than diversified pharma

### Media Cloud News Analysis (39,376 articles)
- **Coverage Volume**: 23,534 disease-focused articles vs 15,697 GLP-1 articles vs 145 Resmetirom articles
- **FDA Impact**: Resmetirom coverage increased 1,100% after approval, GLP-1 coverage +8.5% after approval
- **Top Sources**: Benzinga.com dominates all categories (2,073 disease, 1,811 GLP-1, 102 Resmetirom articles)
- **Media Focus**: Significant coverage spikes around both FDA approval events

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
- **Results**: All analysis outputs in `/results/` folder
- **Data**: All collected data in `/data/` folder
- **Logs**: Execution details printed to console

**Interactive Analysis**: See `src/results.ipynb` for detailed analysis and visualizations.


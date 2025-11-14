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


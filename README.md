# dsci510_fall2025_final_project
A final project to track public and scientific awareness of MASLD

# Task 1: Google Trends Data Collection & Analysis
**Objective: 
Collect and analyze historical search interest data to track public awareness of MASLD terminology adoption and monitor the impact of FDA drug approvals (Resmetirom - March 2024, GLP-1 agonists - August 2025) on public search behavior.

**Research Questions:
- How has search interest shifted from NAFLD to MASLD terminology following the nomenclature change?
- Did FDA drug approvals trigger significant changes in public search behavior?
- What is the comparative public interest in MASLD/NAFLD versus related pharmaceuticals?

**Data Collection:
>Keywords Tracked
- Disease Terminology: MASLD, NAFLD
- Pharmaceuticals: Rezdiffra, Wegovy, Ozempic

>Technical Implementation
- Time Period: January 1, 2023 - October 28, 2025
- Geographic Scope: Global search data
- Data Points: ~5,000 weekly interest scores
- Output Format: CSV time-series data

>Code Features
- Key configuration:
keywords = ['MASLD', 'NAFLD', 'Rezdiffra', 'Wegovy', 'Ozempic']
timeframe = '2023-01-01 2025-10-28'
- Custom HTTP headers to avoid rate limiting
- Automated handling of partial data markers
- Error handling for Google Trends API limitations

**How to Reproduce:
>Prerequisites:
pip install pytrends pandas

>Execution:
python google_trends.py

>Output:
- File: google_trends_initial_data.csv
- Format: Weekly relative search interest (0-100 scale)
- Columns: Date, MASLD, NAFLD, Rezdiffra, Wegovy, Ozempic

**Planned Analysis
- This data will enable:
- Time-series analysis of terminology transition (MASLD vs NAFLD)
- Event study analysis around FDA approval dates
- Correlation analysis between drug-related and disease-related searches
- Comparative visualization of public interest trends

**Connection to Overall Project
This task establishes the public awareness baseline that will be correlated with:
- Reddit discussion sentiment (Task 2)
- Academic publication trends (Task 3)

**Files
- google_trends.py - Data collection script
- google_trends_initial_data.csv - Raw search interest data
- tests.py - Code tesing

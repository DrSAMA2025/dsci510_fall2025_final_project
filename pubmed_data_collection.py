# pubmed_data_collection.py
import requests
import pandas as pd
from datetime import datetime
import time
import xml.etree.ElementTree as ET
import os


def setup_folders():
    """Create necessary folders for data storage"""
    if not os.path.exists('data'):
        os.makedirs('data')


def get_pubmed_search_queries():
    """Define multiple smaller search queries with proper PubMed syntax"""
    queries = [
        # Main MASLD/NAFLD terms - simplified
        'MASLD OR NAFLD OR NASH OR MASH',

        # Resmetirom terms
        'Resmetirom OR Rezdiffra',

        # GLP-1 terms
        'Semaglutide OR Ozempic OR Wegovy OR "GLP-1"',

        # Combined searches
        '(MASLD OR NAFLD) AND (Resmetirom OR Rezdiffra)',
        '(MASLD OR NAFLD) AND (Semaglutide OR Ozempic OR Wegovy)'
    ]
    return queries


def build_pubmed_query(search_term):
    """Build PubMed API query with proper date range syntax"""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    # PubMed uses "2023/01/01"[PDAT] for publication date filtering
    params = {
        'db': 'pubmed',
        'term': f'({search_term}) AND (2023/01/01[PDAT] : 2025/10/28[PDAT])',
        'retmax': 500,  # Start smaller
        'retmode': 'json'
    }

    return base_url, params


def fetch_pubmed_ids_for_query(query_url, params, query_description, search_term, max_retries=3):
    """Fetch PubMed article IDs for a single query"""
    for attempt in range(max_retries):
        try:
            print(f"Attempting: {query_description}")
            response = requests.get(query_url, params=params, timeout=30)

            if response.status_code == 400:
                print(f"Bad request - trying simpler approach...")
                # Try without date filter
                params_simple = params.copy()
                params_simple['term'] = f'({search_term})'
                response = requests.get(query_url, params=params_simple, timeout=30)

            response.raise_for_status()

            data = response.json()
            article_ids = data.get('esearchresult', {}).get('idlist', [])
            total_count = int(data.get('esearchresult', {}).get('count', 0))

            print(f"Found {total_count} articles for: {query_description}")
            return article_ids, total_count

        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed for {query_description}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                print(f"All attempts failed for: {query_description}")
                return [], 0


def check_pubmed_connection():
    """Test if we can connect to PubMed with a simple query"""
    print("Testing PubMed connection...")
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        'db': 'pubmed',
        'term': 'COVID-19',
        'retmax': 5,
        'retmode': 'json'
    }

    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        count = data.get('esearchresult', {}).get('count', 0)
        print(f"✓ PubMed connection successful. Found {count} COVID-19 articles.")
        return True
    except Exception as e:
        print(f"✗ PubMed connection failed: {e}")
        return False


def fetch_all_pubmed_ids():
    """Fetch article IDs from all search queries"""
    # First test connection
    if not check_pubmed_connection():
        return [], {}

    queries = get_pubmed_search_queries()
    all_article_ids = set()
    query_results = {}

    for i, search_term in enumerate(queries):
        query_description = f"Query {i + 1}: {search_term}"
        query_url, params = build_pubmed_query(search_term)

        article_ids, count = fetch_pubmed_ids_for_query(query_url, params, query_description, search_term)
        query_results[f"query_{i + 1}"] = {
            'search_term': search_term,
            'count': count,
            'article_ids': article_ids
        }

        # Add to combined set
        all_article_ids.update(article_ids)

        # Be respectful to PubMed API
        time.sleep(1)

    print(f"\nCombined unique articles across all searches: {len(all_article_ids)}")
    return list(all_article_ids), query_results


def fetch_article_details(article_ids, batch_size=100):
    """Fetch detailed information for PubMed articles"""
    if not article_ids:
        return []

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    all_articles = []

    # Process in batches to avoid overloading API
    for i in range(0, len(article_ids), batch_size):
        batch_ids = article_ids[i:i + batch_size]
        print(
            f"Fetching batch {i // batch_size + 1}/{(len(article_ids) - 1) // batch_size + 1} ({len(batch_ids)} articles)")

        params = {
            'db': 'pubmed',
            'id': ','.join(batch_ids),
            'retmode': 'xml'
        }

        try:
            response = requests.get(base_url, params=params, timeout=60)
            response.raise_for_status()

            articles = parse_pubmed_xml(response.content)
            all_articles.extend(articles)

            print(f"  Successfully processed {len(articles)} articles in this batch")

            # Be respectful to PubMed API
            time.sleep(1)

        except requests.exceptions.RequestException as e:
            print(f"Error fetching batch: {e}")
            continue

    return all_articles


def parse_pubmed_xml(xml_content):
    """Parse PubMed XML response and extract relevant fields"""
    articles = []

    try:
        root = ET.fromstring(xml_content)

        for article in root.findall('.//PubmedArticle'):
            article_data = extract_article_data(article)
            if article_data:
                articles.append(article_data)

    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")

    return articles


def extract_article_data(article_element):
    """Extract specific data from PubMed article XML"""
    try:
        # PubMed ID
        pmid_element = article_element.find('.//PMID')
        pmid = pmid_element.text if pmid_element is not None else 'N/A'

        # Article title
        title_element = article_element.find('.//ArticleTitle')
        title = title_element.text if title_element is not None else 'N/A'

        # Abstract
        abstract_text = ''
        abstract_elements = article_element.findall('.//AbstractText')
        for elem in abstract_elements:
            if elem.text:
                abstract_text += elem.text + ' '

        # Publication date - try multiple date formats
        pub_date = 'N/A'
        pub_year = 'N/A'
        pub_month = '01'

        # Try PubDate first
        year_elem = article_element.find('.//PubDate/Year')
        if year_elem is not None:
            pub_year = year_elem.text
            month_elem = article_element.find('.//PubDate/Month')
            if month_elem is not None:
                pub_month = month_elem.text

        # Journal
        journal_element = article_element.find('.//Journal/Title')
        journal = journal_element.text if journal_element is not None else 'N/A'

        # Authors
        author_list = article_element.findall('.//Author')
        authors = []
        for author in author_list:
            last_name = author.find('LastName')
            fore_name = author.find('ForeName')
            if last_name is not None and fore_name is not None:
                authors.append(f"{fore_name.text} {last_name.text}")

        article_data = {
            'pubmed_id': pmid,
            'title': title,
            'abstract': abstract_text.strip(),
            'publication_year': pub_year,
            'publication_month': pub_month,
            'journal': journal,
            'authors': '; '.join(authors),
            'abstract_length': len(abstract_text),
            'has_abstract': len(abstract_text.strip()) > 50  # More realistic threshold
        }

        return article_data

    except Exception as e:
        print(f"Error extracting article data: {e}")
        return None


def save_pubmed_data(articles, filename=None):
    """Save PubMed data to CSV file"""
    if not filename:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        filename = f'data/pubmed_masld_articles_{timestamp}.csv'

    df = pd.DataFrame(articles)

    # Create publication date column for analysis
    if 'publication_year' in df.columns:
        df['publication_date'] = pd.to_datetime(
            df['publication_year'] + '-' + df['publication_month'],
            errors='coerce'
        )

    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"Saved {len(df)} articles to {filename}")
    return filename


def main():
    """Main function to coordinate PubMed data collection"""
    print("Starting PubMed Data Collection for MASLD Project")
    print("=" * 50)

    # Setup folders
    setup_folders()

    print("Searching PubMed from 2023-2025 for MASLD and related terms...")

    # Fetch article IDs from all queries
    all_article_ids, query_results = fetch_all_pubmed_ids()

    if not all_article_ids:
        print("No articles found. This could be due to:")
        print("1. Network connectivity issues")
        print("2. PubMed API temporary unavailability")
        print("3. Overly restrictive search terms")
        print("Trying alternative approach...")
        return

    print(f"Successfully retrieved {len(all_article_ids)} unique article IDs")

    # Fetch detailed article information
    print("Fetching article details...")
    articles = fetch_article_details(all_article_ids)

    if articles:
        # Save data
        filename = save_pubmed_data(articles)

        # Print summary
        print("\n=== DATA COLLECTION SUMMARY ===")
        print(f"Total articles collected: {len(articles)}")
        print(f"Articles with abstracts: {sum(1 for a in articles if a['has_abstract'])}")

        # Yearly distribution
        years = [a['publication_year'] for a in articles if a['publication_year'] != 'N/A']
        if years:
            year_counts = pd.Series(years).value_counts().sort_index()
            print("\nArticles by year:")
            for year, count in year_counts.items():
                print(f"  {year}: {count} articles")

        print(f"\nData saved to: {filename}")
        print("You can now run pubmed_analysis.py to analyze this data.")

    else:
        print("No articles were successfully downloaded.")


if __name__ == "__main__":
    main()
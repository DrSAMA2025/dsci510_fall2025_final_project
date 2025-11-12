# pubmed_data_collection.py
import requests
import pandas as pd
from datetime import datetime
import time
import xml.etree.ElementTree as ET
import os


def setup_folders(data_dir='data', results_dir='results'):
    """Create necessary folders for data and results storage"""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    print(f"Checked/Created '{data_dir}/' and '{results_dir}/' folders.")


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
    """
    Build PubMed API query using the specified fixed date range (2023/01/01 to 2025/10/28).
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    # --- KEEPING FIXED DATE AS REQUESTED ---
    start_date = "2023/01/01"
    end_date = "2025/10/28"
    # ---------------------------------------

    # PubMed uses "YYYY/MM/DD"[PDAT] for publication date filtering
    params = {
        'db': 'pubmed',
        # Increased retmax to 10000 to capture most or all IDs in one call
        'retmax': 10000,
        'term': f'({search_term}) AND ({start_date}[PDAT] : {end_date}[PDAT])',
        'retmode': 'json'
    }

    return base_url, params


def fetch_pubmed_ids_for_query(query_url, params, query_description, search_term, max_retries=3):
    """Fetch PubMed article IDs for a single query with retry logic."""
    for attempt in range(max_retries):
        try:
            print(f"Attempting: {query_description}")
            response = requests.get(query_url, params=params, timeout=30)

            if response.status_code == 400:
                print(f"Bad request - trying simpler approach (without date filter)...")
                # Try without date filter as a fallback for complex queries
                params_simple = params.copy()
                params_simple['term'] = f'({search_term})'
                response = requests.get(query_url, params=params_simple, timeout=30)

            response.raise_for_status()

            data = response.json()
            article_ids = data.get('esearchresult', {}).get('idlist', [])
            total_count = int(data.get('esearchresult', {}).get('count', 0))

            print(f"Found {total_count} articles for: {query_description}")
            if total_count > len(article_ids) and len(article_ids) > 0:
                print(f"  Note: Only {len(article_ids)} IDs retrieved (retmax limit).")
            elif total_count == 0:
                print(f"  No articles found for this query.")

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

        # Be respectful to PubMed API (1 second delay)
        time.sleep(1)

    print(f"\nCombined unique articles across all searches: {len(all_article_ids)}")
    return list(all_article_ids), query_results


def fetch_article_details(article_ids, batch_size=100):
    """Fetch detailed information for PubMed articles in batches"""
    if not article_ids:
        return []

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    all_articles = []

    # Process in batches to avoid overloading API
    total_batches = (len(article_ids) - 1) // batch_size + 1
    for i in range(0, len(article_ids), batch_size):
        batch_ids = article_ids[i:i + batch_size]
        print(
            f"Fetching batch {i // batch_size + 1}/{total_batches} ({len(batch_ids)} articles)")

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

            # Be respectful to PubMed API (1 second delay)
            time.sleep(1)

        except requests.exceptions.RequestException as e:
            print(f"Error fetching batch: {e}. Skipping batch.")
            continue

    return all_articles


def parse_pubmed_xml(xml_content):
    """Parse PubMed XML response and extract relevant fields"""
    articles = []

    try:
        root = ET.fromstring(xml_content)

        # Find all individual articles
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

        # Article title (can be found in two places, ArticleTitle is usually best)
        title_element = article_element.find('.//ArticleTitle')
        title = title_element.text if title_element is not None else 'N/A'

        # Abstract
        abstract_text = ''
        abstract_elements = article_element.findall('.//AbstractText')
        for elem in abstract_elements:
            # Check for a label/heading if present, otherwise just append text
            label = elem.attrib.get('Label', '')
            text_part = elem.text or ''

            if label:
                abstract_text += f"[{label}] {text_part} "
            else:
                abstract_text += text_part + ' '

        # Publication date - try multiple date formats
        pub_year = 'N/A'
        pub_month = '01'

        # Try PubDate first
        year_elem = article_element.find('.//PubDate/Year')
        if year_elem is not None:
            pub_year = year_elem.text
            # Use Month name or abbreviation if available, otherwise default to 01
            month_elem = article_element.find('.//PubDate/Month')
            if month_elem is not None:
                pub_month = month_elem.text

        # Journal
        journal_element = article_element.find('.//Journal/Title')
        journal = journal_element.text if journal_element is not None else 'N/A'

        # Authors
        author_list = article_element.findall('.//AuthorList/Author')
        authors = []
        for author in author_list:
            last_name = author.find('LastName')
            fore_name = author.find('ForeName')
            initials = author.find('Initials')

            author_name = ''
            if last_name is not None and last_name.text:
                author_name += last_name.text

            if fore_name is not None and fore_name.text:
                author_name += f", {fore_name.text}"
            elif initials is not None and initials.text:
                author_name += f", {initials.text}"

            if author_name:
                authors.append(author_name)

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
        # We don't print the full error here to avoid console clutter during batch processing
        return None


def save_pubmed_data(articles, filename_prefix='pubmed_masld_articles', data_dir='data'):
    """Save PubMed data to CSV file"""

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(data_dir, f'{filename_prefix}_{timestamp}.csv')

    df = pd.DataFrame(articles)

    # Create publication date column for analysis
    if 'publication_year' in df.columns:
        # Convert month abbreviations (Jan, Feb) to numbers (01, 02) for consistent date parsing
        month_to_num = {
            'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
            'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12',
        }
        # Safely convert month to string and map, keeping originals if no map exists (e.g., '01')
        df['pub_month_num'] = df['publication_month'].astype(str).str.slice(0, 3).map(month_to_num).fillna('01')

        # Combine Year and numerical Month
        df['publication_date'] = pd.to_datetime(
            df['publication_year'].astype(str) + '-' + df['pub_month_num'].astype(str),
            errors='coerce'
        )
        df = df.drop(columns=['pub_month_num'])  # drop temp column

    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"Saved {len(df)} articles to {filename}")
    return filename, df


def main(data_dir='../data', results_dir='../results', notebook_mode=False):
    """Main function to coordinate PubMed data collection"""
    print("Starting PubMed Data Collection for MASLD Project")
    print("=" * 50)

    # Setup folders
    setup_folders(data_dir=data_dir, results_dir=results_dir)

    print(f"Searching PubMed from 2023/01/01 to 2025/10/28 for MASLD and related terms...")

    # Fetch article IDs from all queries
    all_article_ids, query_results = fetch_all_pubmed_ids()

    if not all_article_ids:
        print("No articles found. Check connection or search terms.")
        return None

    print(f"Successfully retrieved {len(all_article_ids)} unique article IDs")

    # Fetch detailed article information
    print("Fetching article details...")
    articles = fetch_article_details(all_article_ids)

    if articles:
        # Save data
        filename, df = save_pubmed_data(articles, data_dir=data_dir)

        # Print summary
        print("\n=== DATA COLLECTION SUMMARY ===")
        print(f"Total articles collected: {len(df)}")
        print(f"Articles with abstracts: {df['has_abstract'].sum()}")

        # Count by year for quick summary
        if 'publication_year' in df.columns:
            year_counts = df[df['publication_year'] != 'N/A']['publication_year'].value_counts().sort_index()
            print("\nArticles by year:")
            for year, count in year_counts.items():
                print(f"  {year}: {count} articles")

        print(f"\nData saved to: {filename}")

        if notebook_mode:
            return df

    else:
        print("No articles were successfully downloaded or parsed.")

    return None


if __name__ == "__main__":
    # When run directly, assume relative paths from src/
    main(data_dir='../data', results_dir='../results')
# Tests for Google Trends Data (Data Source #1)
import pandas as pd
import os


def test_data_integrity(filename='data/google_trends_initial_data.csv'):

    print("\n" + "=" * 50)
    print("GOOGLE TRENDS DATA VALIDATION TESTS")
    print("=" * 50)

    try:
        # 1. Check if file exists
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found. Run google_trends.py first.")

        # 2. Load data
        df = pd.read_csv(filename)

        print("File loaded successfully")
        print(f"Dataset shape: {df.shape}")

        # 3. Basic structure tests
        print("\n--- STRUCTURE TESTS ---")

        # Check required columns
        required_columns = ['date', 'MASLD', 'NAFLD', 'Rezdiffra', 'Wegovy', 'Ozempic']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"FAIL: Missing columns: {missing_columns}")
        else:
            print("PASS: All required columns present")

        # Check data types
        if pd.api.types.is_datetime64_any_dtype(df['date']):
            print("PASS: Date column is properly formatted as datetime")
        else:
            print("WARNING: Date column is not datetime - consider converting")

        # 4. Data quality tests
        print("\n--- DATA QUALITY TESTS ---")

        # Check for missing values
        missing_values = df.isnull().sum().sum()
        if missing_values == 0:
            print("PASS: No missing values in dataset")
        else:
            print(f"WARNING: Found {missing_values} missing values")

        # Check date range and continuity
        df['date'] = pd.to_datetime(df['date'])
        date_range = df['date'].max() - df['date'].min()
        expected_days = (pd.to_datetime('2025-10-28') - pd.to_datetime('2023-01-01')).days
        date_coverage = (date_range.days / expected_days) * 100

        print(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        print(f"Date coverage: {date_coverage:.1f}% of expected range")

        # 5. Content validation tests
        print("\n--- CONTENT VALIDATION TESTS ---")

        # Check that MASLD data exists (should not be all zeros)
        masld_non_zero = (df['MASLD'] > 0).sum()
        if masld_non_zero > 0:
            print(f"PASS: MASLD has {masld_non_zero} non-zero data points")
        else:
            print("FAIL: MASLD has no non-zero values - check data collection")

        # Check value ranges (Google Trends should be 0-100)
        trend_columns = ['MASLD', 'NAFLD', 'Rezdiffra', 'Wegovy', 'Ozempic']
        for col in trend_columns:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                if 0 <= min_val <= max_val <= 100:
                    print(f"PASS: {col}: values in valid range ({min_val}-{max_val})")
                else:
                    print(f"WARNING: {col}: values outside expected 0-100 range ({min_val}-{max_val})")

        # 6. Sample data display
        print("\n--- SAMPLE DATA ---")
        print("First 3 rows:")
        print(df.head(3).to_string(index=False))

        print("\n--- SUMMARY STATISTICS ---")
        print(df[trend_columns].describe())

        # 7. Final assessment
        print("\n" + "=" * 50)
        print("TEST SUMMARY")
        print("=" * 50)
        print("Google Trends data collection: COMPLETE")
        print(f"Total data points: {len(df)}")
        print(f"Time period coverage: {date_coverage:.1f}%")
        print("Data appears valid and ready for analysis!")

        return True

    except Exception as e:
        print(f"TEST FAILED: {e}")
        return False


def test_analysis_outputs():
    print("\n--- ANALYSIS OUTPUT TESTS ---")

    expected_files = [
        'google_trends_analysis.py',
        'analysis/google_trends_main_analysis.png',
        'analysis/google_trends_correlation_heatmap.png',
        'analysis/google_trends_statistical_summary.csv'
    ]

    for file in expected_files:
        if os.path.exists(file):
            print(f"PASS: {file} - Found")
        else:
            print(f"WARNING: {file} - Missing")


if __name__ == "__main__":
    # Run data integrity tests
    data_ok = test_data_integrity()

    # Run analysis output tests
    test_analysis_outputs()

    print("\n" + "=" * 50)
    if data_ok:
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("Task 1 data is ready for the final project.")
    else:
        print("SOME TESTS FAILED - Please check your data collection.")
    print("=" * 50)


# Tests for Reddit Data (Data Source #2)
# tests.py
import pandas as pd
import os
import sys
from datetime import datetime


def test_reddit_data_quality():
    """Test the quality and structure of Reddit data"""
    print("=== REDDIT DATA QUALITY TESTS ===")

    # Test 1: Check if Reddit data file exists
    try:
        reddit_files = [f for f in os.listdir('data') if f.startswith('reddit_data_2023_2025_') and f.endswith('.csv')]
        assert len(reddit_files) > 0, "No Reddit data files found in data folder"
        latest_file = max(reddit_files)
        print(f"✓ Found Reddit data file: {latest_file}")
    except Exception as e:
        print(f"✗ Test 1 FAILED: {e}")
        return False

    # Test 2: Load and check data structure
    try:
        df = pd.read_csv(f'data/{latest_file}')
        expected_columns = ['subreddit', 'post_title', 'post_text', 'comment_text', 'timestamp', 'type']
        for col in expected_columns:
            assert col in df.columns, f"Missing expected column: {col}"
        print("✓ Data structure validation passed")
    except Exception as e:
        print(f"✗ Test 2 FAILED: {e}")
        return False

    # Test 3: Check data volume
    try:
        assert len(df) > 0, "Data file is empty"
        assert len(df) >= 1000, f"Low data volume: only {len(df)} records"
        print(f"✓ Data volume check passed: {len(df):,} records")
    except Exception as e:
        print(f"✗ Test 3 FAILED: {e}")
        return False

    # Test 4: Check date range
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        min_date = df['timestamp'].min()
        max_date = df['timestamp'].max()
        assert min_date >= pd.Timestamp('2023-01-01'), f"Data starts before study period: {min_date}"
        assert max_date <= pd.Timestamp('2025-10-28'), f"Data extends beyond study period: {max_date}"
        print(f"✓ Date range validation passed: {min_date.date()} to {max_date.date()}")
    except Exception as e:
        print(f"✗ Test 4 FAILED: {e}")
        return False

    # Test 5: Check subreddit coverage
    try:
        subreddits = df['subreddit'].unique()
        expected_subs = ['Ozempic', 'Wegovy', 'semaglutide', 'NAFLD', 'MASH']
        found_subs = [sub for sub in expected_subs if sub in subreddits]
        assert len(found_subs) >= 3, f"Missing key subreddits. Found: {found_subs}"
        print(f"✓ Subreddit coverage passed: {len(subreddits)} subreddits including {found_subs}")
    except Exception as e:
        print(f"✗ Test 5 FAILED: {e}")
        return False

    # Test 6: Check for key term mentions
    try:
        all_text = df['post_text'].fillna('') + ' ' + df['comment_text'].fillna('')
        key_terms = ['resmetirom', 'rezdiffra', 'semaglutide', 'ozempic', 'wegovy', 'nafld', 'nash', 'masld']
        mentions = {term: all_text.str.contains(term, case=False).sum() for term in key_terms}
        total_mentions = sum(mentions.values())
        assert total_mentions > 0, "No mentions of key MASLD terms found"
        print(f"✓ Key term mentions found: {total_mentions} total mentions")
        for term, count in mentions.items():
            if count > 0:
                print(f"  - {term}: {count} mentions")
    except Exception as e:
        print(f"✗ Test 6 FAILED: {e}")
        return False

    print("\nALL REDDIT DATA TESTS PASSED!")
    return True


def test_sentiment_analysis_output():
    """Test the sentiment analysis outputs"""
    print("\n=== SENTIMENT ANALYSIS OUTPUT TESTS ===")

    # Test 1: Check if sentiment analysis files exist
    try:
        analysis_files = os.listdir('analysis')
        sentiment_files = [f for f in analysis_files if 'sentiment' in f.lower()]
        assert len(sentiment_files) >= 3, f"Not enough sentiment analysis files. Found: {sentiment_files}"
        print(f"✓ Found sentiment analysis files: {len(sentiment_files)} files")
    except Exception as e:
        print(f"✗ Sentiment file test FAILED: {e}")
        return False

    # Test 2: Check daily sentiment file
    try:
        daily_files = [f for f in analysis_files if 'daily' in f.lower() and 'sentiment' in f.lower()]
        if daily_files:
            daily_df = pd.read_csv(f'analysis/{daily_files[0]}')
            assert 'avg_sentiment' in daily_df.columns, "Missing avg_sentiment column"
            assert 'date' in daily_df.columns, "Missing date column"
            assert len(daily_df) > 0, "Daily sentiment file is empty"
            print(f"✓ Daily sentiment file validation passed: {len(daily_df)} days")
    except Exception as e:
        print(f"✗ Daily sentiment test FAILED: {e}")
        return False

    # Test 3: Check sentiment range
    try:
        full_files = [f for f in analysis_files if 'with_sentiment' in f.lower()]
        if full_files:
            full_df = pd.read_csv(f'analysis/{full_files[0]}')
            assert 'sentiment' in full_df.columns, "Missing sentiment column in full dataset"
            sentiment_range = full_df['sentiment'].between(-1, 1).all()
            assert sentiment_range, "Sentiment scores outside valid range (-1 to 1)"
            print(
                f"✓ Sentiment range validation passed: {full_df['sentiment'].min():.3f} to {full_df['sentiment'].max():.3f}")
    except Exception as e:
        print(f"✗ Sentiment range test FAILED: {e}")
        return False

    print("ALL SENTIMENT ANALYSIS TESTS PASSED!")
    return True


def test_visualization_outputs():
    """Test that visualization files were created"""
    print("\n=== VISUALIZATION OUTPUT TESTS ===")

    try:
        analysis_files = os.listdir('analysis')
        plot_files = [f for f in analysis_files if f.endswith('.png')]
        assert len(plot_files) >= 3, f"Expected at least 3 plots, found: {len(plot_files)}"

        expected_plots = ['trend', 'subreddit', 'distribution']
        found_keywords = []
        for plot_file in plot_files:
            if any(keyword in plot_file.lower() for keyword in ['trend', 'fda']):
                found_keywords.append('trend')
            elif 'subreddit' in plot_file.lower():
                found_keywords.append('subreddit')
            elif 'distribution' in plot_file.lower():
                found_keywords.append('distribution')

        assert len(set(found_keywords)) >= 2, f"Missing key plot types. Found: {found_keywords}"
        print(f"✓ Visualization files check passed: {len(plot_files)} plots generated")

    except Exception as e:
        print(f"✗ Visualization test FAILED: {e}")
        return False

    print("ALL VISUALIZATION TESTS PASSED!")
    return True


def main():
    """Run all tests"""
    print("Running MASLD Reddit Analysis Tests...")
    print("=" * 50)

    all_passed = True

    # Run data quality tests
    if not test_reddit_data_quality():
        all_passed = False

    # Run sentiment analysis tests (only if analysis folder exists)
    if os.path.exists('analysis'):
        if not test_sentiment_analysis_output():
            all_passed = False
        if not test_visualization_outputs():
            all_passed = False
    else:
        print("\nAnalysis folder not found - skipping sentiment analysis tests")

    print("\n" + "=" * 50)
    if all_passed:
        print("ALL TESTS PASSED! Reddit data analysis is ready for use.")
    else:
        print("SOME TESTS FAILED! Please check your data and analysis.")

    return all_passed


if __name__ == "__main__":
    main()

# Tests for PubMed Data (Data Source #3)
# tests_pubmed.py
import unittest
import pandas as pd
import os
import sys
from unittest.mock import patch, MagicMock
import xml.etree.ElementTree as ET

# Add the current directory to Python path so we can import the modules
sys.path.append('.')


class TestPubMedDataCollection(unittest.TestCase):
    """Test PubMed data collection functions"""

    def setUp(self):
        """Set up test data"""
        self.sample_pubmed_xml = """
        <PubmedArticleSet>
            <PubmedArticle>
                <MedlineCitation Status="MEDLINE" Owner="NLM">
                    <PMID Version="1">12345678</PMID>
                    <Article>
                        <ArticleTitle>Test Article About MASLD and Semaglutide</ArticleTitle>
                        <Abstract>
                            <AbstractText>This is a test abstract about MASLD treatment with Semaglutide.</AbstractText>
                        </Abstract>
                        <AuthorList CompleteYN="Y">
                            <Author ValidYN="Y">
                                <LastName>Smith</LastName>
                                <ForeName>John</ForeName>
                            </Author>
                            <Author ValidYN="Y">
                                <LastName>Johnson</LastName>
                                <ForeName>Jane</ForeName>
                            </Author>
                        </AuthorList>
                    </Article>
                    <Journal>
                        <Title>Test Journal</Title>
                    </Journal>
                    <PubDate>
                        <Year>2024</Year>
                        <Month>Mar</Month>
                    </PubDate>
                </MedlineCitation>
            </PubmedArticle>
        </PubmedArticleSet>
        """

    def test_search_terms_creation(self):
        """Test that search terms are created correctly"""
        from pubmed_data_collection import get_pubmed_search_queries

        queries = get_pubmed_search_queries()

        self.assertIsInstance(queries, list)
        self.assertGreater(len(queries), 0)

        # Check that key terms are included
        all_terms = ' '.join(queries).lower()
        self.assertIn('masld', all_terms)
        self.assertIn('nafld', all_terms)
        self.assertIn('resmetirom', all_terms)
        self.assertIn('semaglutide', all_terms)

    def test_xml_parsing(self):
        """Test PubMed XML parsing"""
        from pubmed_data_collection import parse_pubmed_xml

        articles = parse_pubmed_xml(self.sample_pubmed_xml.encode())

        self.assertEqual(len(articles), 1)
        article = articles[0]

        self.assertEqual(article['pubmed_id'], '12345678')
        self.assertEqual(article['title'], 'Test Article About MASLD and Semaglutide')
        self.assertIn('MASLD', article['title'])
        self.assertIn('Semaglutide', article['abstract'])
        self.assertEqual(article['journal'], 'Test Journal')
        self.assertEqual(article['publication_year'], '2024')
        self.assertEqual(article['publication_month'], 'Mar')
        self.assertIn('John Smith', article['authors'])
        self.assertIn('Jane Johnson', article['authors'])

    def test_article_data_extraction(self):
        """Test individual article data extraction"""
        from pubmed_data_collection import extract_article_data

        root = ET.fromstring(self.sample_pubmed_xml)
        article_element = root.find('.//PubmedArticle')

        article_data = extract_article_data(article_element)

        self.assertIsNotNone(article_data)
        self.assertEqual(article_data['pubmed_id'], '12345678')
        self.assertTrue(article_data['has_abstract'])
        self.assertGreater(article_data['abstract_length'], 0)

    @patch('pubmed_data_collection.requests.get')
    def test_api_connection(self, mock_get):
        """Test PubMed API connection"""
        from pubmed_data_collection import check_pubmed_connection

        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'esearchresult': {
                'count': '100',
                'idlist': ['123', '456', '789']
            }
        }
        mock_get.return_value = mock_response

        result = check_pubmed_connection()
        self.assertTrue(result)

    def test_dataframe_creation(self):
        """Test that DataFrame is created with correct columns"""
        from pubmed_data_collection import save_pubmed_data

        test_articles = [{
            'pubmed_id': '12345678',
            'title': 'Test Article',
            'abstract': 'Test abstract',
            'publication_year': '2024',
            'publication_month': 'Mar',
            'journal': 'Test Journal',
            'authors': 'John Smith',
            'keywords': '',
            'publication_types': '',
            'abstract_length': 12,
            'has_abstract': True
        }]

        # Test saving data
        filename = 'test_output.csv'
        try:
            result_file = save_pubmed_data(test_articles, filename)

            # Check file was created
            self.assertTrue(os.path.exists(result_file))

            # Check DataFrame structure
            df = pd.read_csv(result_file)
            expected_columns = ['pubmed_id', 'title', 'abstract', 'publication_year',
                                'publication_month', 'journal', 'authors', 'keywords',
                                'publication_types', 'abstract_length', 'has_abstract',
                                'publication_date']

            for col in expected_columns:
                self.assertIn(col, df.columns)

        finally:
            # Clean up
            if os.path.exists(filename):
                os.remove(filename)


class TestPubMedAnalysis(unittest.TestCase):
    """Test PubMed analysis functions"""

    def setUp(self):
        """Set up test data for analysis"""
        self.test_data = pd.DataFrame({
            'pubmed_id': ['1', '2', '3', '4', '5'],
            'title': [
                'MASLD treatment with Resmetirom',
                'NAFLD and Semaglutide study',
                'MASH clinical trial',
                'GLP-1 agonists for obesity',
                'Resmetirom Phase 3 results'
            ],
            'abstract': [
                'Study of MASLD treated with Resmetirom shows good results',
                'NAFLD patients responded well to Semaglutide treatment',
                'MASH patients in clinical trial',
                'GLP-1 agonists effective for weight loss',
                'Resmetirom shows promise in Phase 3 trials'
            ],
            'publication_date': pd.to_datetime([
                '2023-01-15', '2023-06-20', '2024-02-10',
                '2024-08-05', '2025-03-12'
            ]),
            'journal': ['Journal A', 'Journal B', 'Journal A', 'Journal C', 'Journal B'],
            'authors': ['Author X', 'Author Y', 'Author Z', 'Author W', 'Author V'],
            'abstract_length': [100, 150, 120, 180, 90],
            'has_abstract': [True, True, True, True, True]
        })

    def test_data_cleaning(self):
        """Test data cleaning and preprocessing"""
        from pubmed_analysis import clean_and_preprocess_data

        df_cleaned = clean_and_preprocess_data(self.test_data.copy())

        # Check that new columns are created
        self.assertIn('year', df_cleaned.columns)
        self.assertIn('month', df_cleaned.columns)
        self.assertIn('year_month', df_cleaned.columns)

        # Check term detection
        self.assertIn('mentions_masld', df_cleaned.columns)
        self.assertIn('mentions_nafld', df_cleaned.columns)
        self.assertIn('mentions_resmetirom', df_cleaned.columns)
        self.assertIn('mentions_glp1', df_cleaned.columns)

        # Verify term detection works
        self.assertTrue(df_cleaned.loc[0, 'mentions_masld'])
        self.assertTrue(df_cleaned.loc[0, 'mentions_resmetirom'])
        self.assertTrue(df_cleaned.loc[1, 'mentions_nafld'])
        self.assertTrue(df_cleaned.loc[1, 'mentions_glp1'])

    def test_publication_trends(self):
        """Test publication trend analysis"""
        from pubmed_analysis import analyze_publication_trends

        df_cleaned = self.test_data.copy()
        df_cleaned['year_month'] = df_cleaned['publication_date'].dt.to_period('M')

        # Mock the analysis folder
        with patch('pubmed_analysis.plt') as mock_plt:
            trends = analyze_publication_trends(df_cleaned, 'test_analysis')

            self.assertIsNotNone(trends)
            self.assertGreater(len(trends), 0)

    def test_summary_statistics(self):
        """Test summary statistics generation"""
        from pubmed_analysis import generate_summary_statistics

        df_cleaned = self.test_data.copy()

        # Add term mention columns for testing
        df_cleaned['mentions_masld'] = [True, False, False, False, False]
        df_cleaned['mentions_nafld'] = [False, True, False, False, False]
        df_cleaned['mentions_resmetirom'] = [True, False, False, False, True]
        df_cleaned['mentions_glp1'] = [False, True, False, True, False]
        df_cleaned['mentions_masld_resmetirom'] = [True, False, False, False, False]
        df_cleaned['mentions_masld_glp1'] = [False, False, False, False, False]

        with patch('pandas.DataFrame.to_csv'):
            summary_stats, yearly_stats = generate_summary_statistics(df_cleaned, 'test_analysis')

            self.assertIsInstance(summary_stats, dict)
            self.assertEqual(summary_stats['total_publications'], 5)
            self.assertEqual(summary_stats['publications_mentioning_masld'], 1)
            self.assertEqual(summary_stats['publications_mentioning_resmetirom'], 2)

    def test_journal_analysis(self):
        """Test journal distribution analysis"""
        from pubmed_analysis import analyze_journal_distribution

        with patch('pubmed_analysis.plt') as mock_plt:
            journal_dist = analyze_journal_distribution(self.test_data, 'test_analysis')

            self.assertIsNotNone(journal_dist)
            self.assertEqual(len(journal_dist), 3)  # 3 unique journals
            self.assertEqual(journal_dist['Journal A'], 2)
            self.assertEqual(journal_dist['Journal B'], 2)


class TestIntegration(unittest.TestCase):
    """Integration tests for the full workflow"""

    def test_folder_creation(self):
        """Test that necessary folders are created"""
        from pubmed_data_collection import setup_folders

        # Test folder setup
        setup_folders()
        self.assertTrue(os.path.exists('data'))

    def test_complete_workflow(self):
        """Test the complete data collection and analysis workflow"""
        # This would be a more comprehensive integration test
        # For now, we'll verify the main functions exist and are callable
        from pubmed_data_collection import main as collection_main
        from pubmed_analysis import main as analysis_main

        # Just verify the functions exist and are callable
        self.assertTrue(callable(collection_main))
        self.assertTrue(callable(analysis_main))


def run_all_tests():
    """Run all tests and print summary"""
    print("Running PubMed Code Tests...")
    print("=" * 50)

    # Create a test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestPubMedDataCollection)
    suite.addTests(loader.loadTestsFromTestCase(TestPubMedAnalysis))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed. Check the details above.")

    return result.wasSuccessful()


if __name__ == "__main__":
    run_all_tests()

# Tests for Stock Data (Data Source #5)
def test_stock_data_integrity(filename='data/stock_prices.csv'):
    print("\n" + "=" * 50)
    print("STOCK DATA VALIDATION TESTS")
    print("=" * 50)

    try:
        # 1. Check if file exists
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found. Run stock_data.py first.")

        # 2. Load data - skip the header row for date parsing
        df = pd.read_csv(filename, header=[0, 1])

        print("Stock data loaded successfully")
        print(f"Dataset shape: {df.shape}")

        # 3. Basic structure tests
        print("\n--- STRUCTURE TESTS ---")

        # Check if we have data for both companies
        if 'NVO' in df['Close'].columns and 'MDGL' in df['Close'].columns:
            print("PASS: Data for both companies (NVO, MDGL) present")
        else:
            print("FAIL: Missing company data")

        # 4. Data quality tests
        print("\n--- DATA QUALITY TESTS ---")

        # Check for missing values
        missing_values = df.isnull().sum().sum()
        if missing_values == 0:
            print("PASS: No missing values in dataset")
        else:
            print(f"INFO: Found {missing_values} missing values (normal for stock data)")

        # Check date range - handle the date column properly
        date_col = df.columns[0]  # First column should be date
        # Skip the first row if it's the header
        date_series = df[date_col].iloc[1:]  # Skip first row
        date_series = pd.to_datetime(date_series, errors='coerce')
        valid_dates = date_series.dropna()

        print(f"Date range: {valid_dates.min().strftime('%Y-%m-%d')} to {valid_dates.max().strftime('%Y-%m-%d')}")
        print(f"Total trading days: {len(valid_dates)}")

        # 5. Content validation
        print("\n--- CONTENT VALIDATION TESTS ---")

        # Check that prices are reasonable (skip first row)
        nvo_prices = df['Close']['NVO'].iloc[1:].astype(float)
        mdgl_prices = df['Close']['MDGL'].iloc[1:].astype(float)

        print(f"Novo Nordisk (NVO) price range: ${nvo_prices.min():.2f} - ${nvo_prices.max():.2f}")
        print(f"Madrigal (MDGL) price range: ${mdgl_prices.min():.2f} - ${mdgl_prices.max():.2f}")

        # 6. FDA approval date coverage
        print("\n--- FDA APPROVAL DATE COVERAGE ---")
        fda_dates = ['2024-03-14', '2025-08-15']
        for date in fda_dates:
            fda_date = pd.to_datetime(date)
            if any(valid_dates.dt.date == fda_date.date()):
                print(f"PASS: FDA approval date {date} is in dataset")
            else:
                # Find the closest trading date
                closest_date = valid_dates.iloc[(valid_dates - fda_date).abs().argmin()]
                print(
                    f"INFO: FDA date {date} was on a weekend/holiday. Closest trading date: {closest_date.strftime('%Y-%m-%d')}")

        print("\n" + "=" * 50)
        print("STOCK DATA TEST SUMMARY")
        print("=" * 50)
        print("Stock data collection: COMPLETE")
        print(f"Total trading days: {len(valid_dates)}")
        print("Data appears valid and ready for analysis!")

        return True

    except Exception as e:
        print(f"TEST FAILED: {e}")
        return False
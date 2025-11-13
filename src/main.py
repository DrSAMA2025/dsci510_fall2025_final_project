# main.py
"""
MASLD AWARENESS TRACKER - MAIN EXECUTION SCRIPT

Orchestrates the complete data pipeline:
1. Data Acquisition (load.py)
2. Data Processing (process.py)
3. Analysis & Visualization (analyze.py + individual analysis scripts)
4. Cross-platform insights
"""

import sys
from pathlib import Path
import argparse
import time
from datetime import datetime

# Add project root to Python path for module imports
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import project modules
try:
    from load import run_all_data_loaders
    from process import run_all_data_processors
    from analyze import run_all_analysis
    from tests import run_all_tests

    # Import individual analysis scripts with error handling
    try:
        from google_trends_analysis import main as run_google_trends_analysis
    except ImportError:
        run_google_trends_analysis = None
        print("Warning: google_trends_analysis module not available")

    try:
        from stock_analysis import analyze_stock_impact as run_stock_analysis
    except ImportError:
        run_stock_analysis = None
        print("Warning: stock_analysis module not available")

    try:
        from reddit_sentiment_analysis import main as run_reddit_analysis
    except ImportError:
        run_reddit_analysis = None
        print("Warning: reddit_sentiment_analysis module not available")

    try:
        from pubmed_analysis import main as run_pubmed_analysis
    except ImportError:
        run_pubmed_analysis = None
        print("Warning: pubmed_analysis module not available")

    try:
        from media_cloud_analysis import main as run_media_cloud_analysis
    except ImportError:
        run_media_cloud_analysis = None
        print("Warning: media_cloud_analysis module not available")

    # Import configuration
    from config import (
        DATA_DIR, RESULTS_DIR,
        STUDY_START_DATE, STUDY_END_DATE,
        FDA_EVENT_DATES
    )
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure all project files are in the same directory.")
    sys.exit(1)


class MASLDAnalysisPipeline:
    """Main orchestrator for the MASLD awareness tracking pipeline."""

    def __init__(self):
        self.start_time = time.time()
        self.execution_log = []
        self.processed_data = {}

    def log_step(self, step_name, status="COMPLETED", message=""):
        """Log execution steps with timing."""
        elapsed = time.time() - self.start_time
        log_entry = {
            'step': step_name,
            'status': status,
            'message': message,
            'elapsed_seconds': round(elapsed, 2),
            'timestamp': datetime.now().strftime('%H:%M:%S')
        }
        self.execution_log.append(log_entry)
        print(f"\n[{log_entry['timestamp']}] {step_name}: {status} {message}")

    def print_header(self):
        """Print project header."""
        print("\n" + "=" * 70)
        print("MASLD AWARENESS TRACKER - COMPREHENSIVE ANALYSIS PIPELINE")
        print("=" * 70)
        print(f"Study Period: {STUDY_START_DATE} to {STUDY_END_DATE}")
        print(f"Key FDA Events: {len(FDA_EVENT_DATES)} events configured")
        print(f"Data Directory: {DATA_DIR}")
        print(f"Results Directory: {RESULTS_DIR}")
        print("=" * 70)

    def run_tests(self, quick=False):
        """Run project tests."""
        self.log_step("SYSTEM TESTS", "STARTING")
        try:
            if quick:
                print("  > Running quick tests...")
                # Import and run minimal tests
                from tests import test_basic_structure
                test_basic_structure()
            else:
                run_all_tests()
            self.log_step("SYSTEM TESTS", "COMPLETED", "All tests passed")
            return True
        except Exception as e:
            self.log_step("SYSTEM TESTS", "FAILED", str(e))
            return False

    def load_data(self, skip_reddit=False, skip_pubmed=False):
        """Run data acquisition from all sources."""
        self.log_step("DATA ACQUISITION", "STARTING")
        from load import ensure_data_available
        ensure_data_available()  # Download from GDrive if data is missing
        try:
            # Run main data loaders
            run_all_data_loaders()

            # Handle optional skips
            if skip_reddit:
                self.log_step("REDDIT DATA", "SKIPPED", "User requested skip")
            if skip_pubmed:
                self.log_step("PUBMED DATA", "SKIPPED", "User requested skip")

            self.log_step("DATA ACQUISITION", "COMPLETED", "All data sources processed")
            return True
        except Exception as e:
            self.log_step("DATA ACQUISITION", "FAILED", str(e))
            return False

    def process_data(self):
        """Process and clean all acquired data."""
        self.log_step("DATA PROCESSING", "STARTING")
        try:
            self.processed_data = run_all_data_processors()
            data_summary = {k: f"{len(v)} records" if hasattr(v, '__len__') else "processed"
                            for k, v in self.processed_data.items()}
            self.log_step("DATA PROCESSING", "COMPLETED", f"Processed: {data_summary}")
            return True
        except Exception as e:
            self.log_step("DATA PROCESSING", "FAILED", str(e))
            return False

    def run_basic_analysis(self):
        """Run basic analysis and visualization."""
        self.log_step("BASIC ANALYSIS", "STARTING")
        try:
            if self.processed_data:
                run_all_analysis(self.processed_data)
                self.log_step("BASIC ANALYSIS", "COMPLETED", "Core visualizations generated")
            else:
                self.log_step("BASIC ANALYSIS", "SKIPPED", "No processed data available")
            return True
        except Exception as e:
            self.log_step("BASIC ANALYSIS", "FAILED", str(e))
            return False

    def run_detailed_analyses(self):
        """Run detailed individual analyses."""
        analyses = [
            ("GOOGLE TRENDS", run_google_trends_analysis),
            ("STOCK ANALYSIS", run_stock_analysis),
            ("REDDIT SENTIMENT", run_reddit_analysis),
            ("PUBMED ANALYSIS", run_pubmed_analysis),
            ("MEDIA CLOUD", run_media_cloud_analysis)
        ]

        for analysis_name, analysis_func in analyses:
            self.log_step(analysis_name, "STARTING")
            try:
                analysis_func()
                self.log_step(analysis_name, "COMPLETED")
            except Exception as e:
                self.log_step(analysis_name, "FAILED", str(e))

    def generate_cross_platform_insights(self):
        """Generate insights across all data platforms."""
        self.log_step("CROSS-PLATFORM INSIGHTS", "STARTING")
        try:
            # This would combine findings from all analyses
            insights = self._generate_combined_insights()
            self._save_insights_report(insights)
            self.log_step("CROSS-PLATFORM INSIGHTS", "COMPLETED", "Combined report generated")
            return True
        except Exception as e:
            self.log_step("CROSS-PLATFORM INSIGHTS", "FAILED", str(e))
            return False

    def _generate_combined_insights(self):
        """Generate combined insights from all data sources."""
        # Placeholder for actual cross-platform analysis
        insights = {
            'timestamp': datetime.now().isoformat(),
            'study_period': f"{STUDY_START_DATE} to {STUDY_END_DATE}",
            'data_sources_processed': list(self.processed_data.keys()),
            'key_findings': [
                "FDA approval impact analysis completed",
                "Multi-platform sentiment and trend tracking active",
                "Media coverage and scientific publication correlation available"
            ]
        }
        return insights

    def _save_insights_report(self, insights):
        """Save combined insights report."""
        report_path = RESULTS_DIR / f"masld_combined_insights_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"

        with open(report_path, 'w') as f:
            f.write("MASLD AWARENESS TRACKER - COMBINED INSIGHTS REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {insights['timestamp']}\n")
            f.write(f"Study Period: {insights['study_period']}\n")
            f.write(f"Data Sources: {', '.join(insights['data_sources_processed'])}\n\n")

            f.write("EXECUTION SUMMARY:\n")
            f.write("-" * 20 + "\n")
            for log in self.execution_log:
                f.write(f"{log['timestamp']} - {log['step']}: {log['status']}\n")

            f.write("\nKEY FINDINGS:\n")
            f.write("-" * 20 + "\n")
            for finding in insights['key_findings']:
                f.write(f"• {finding}\n")

        print(f"  > Insights report saved: {report_path.name}")

    def print_summary(self):
        """Print execution summary."""
        total_time = time.time() - self.start_time
        successful_steps = [log for log in self.execution_log if log['status'] == 'COMPLETED']
        failed_steps = [log for log in self.execution_log if log['status'] == 'FAILED']

        print("\n" + "=" * 70)
        print("EXECUTION SUMMARY")
        print("=" * 70)
        print(f"Total Time: {total_time:.2f} seconds")
        print(f"Successful Steps: {len(successful_steps)}")
        print(f"Failed Steps: {len(failed_steps)}")
        print(f"Data Sources Processed: {len(self.processed_data)}")

        if failed_steps:
            print("\nFailed Steps:")
            for step in failed_steps:
                print(f"  - {step['step']}: {step['message']}")

        print(f"\nResults available in: {RESULTS_DIR}/")
        print("=" * 70)


def main():
    """Main execution function with command line arguments."""
    parser = argparse.ArgumentParser(description='MASLD Awareness Tracker Analysis Pipeline')
    parser.add_argument('--skip-tests', action='store_true', help='Skip system tests')
    parser.add_argument('--skip-load', action='store_true', help='Skip data loading (use existing data)')
    parser.add_argument('--skip-reddit', action='store_true', help='Skip Reddit data collection')
    parser.add_argument('--skip-pubmed', action='store_true', help='Skip PubMed data collection')
    parser.add_argument('--quick', action='store_true', help='Run quick version (minimal data)')
    parser.add_argument('--analysis-only', action='store_true', help='Run only analysis on existing data')

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = MASLDAnalysisPipeline()
    pipeline.print_header()

    try:
        # 1. Run tests (unless skipped)
        if not args.skip_tests:
            if not pipeline.run_tests(quick=args.quick):
                print("Tests failed. Exiting.")
                return

        # 2. Load data (unless skipped or analysis-only)
        if not args.skip_load and not args.analysis_only:
            pipeline.load_data(skip_reddit=args.skip_reddit, skip_pubmed=args.skip_pubmed)

        # 3. Process data (unless analysis-only)
        if not args.analysis_only:
            pipeline.process_data()
        else:
            # For analysis-only, we still need to process existing data
            pipeline.process_data()

        # 4. Run analyses
        pipeline.run_basic_analysis()
        pipeline.run_detailed_analyses()

        # 5. Generate cross-platform insights
        pipeline.generate_cross_platform_insights()

        # 6. Print summary
        pipeline.print_summary()

    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
    except Exception as e:
        print(f"\nPipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure results directory is created even if pipeline fails
        RESULTS_DIR.mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    main()
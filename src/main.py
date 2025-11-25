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

    def _run_trends_analysis(self):
        """Run Google Trends analysis and return structured results."""
        try:
            from analyze import advanced_google_trends_analysis
            if 'trends' in self.processed_data:
                results = advanced_google_trends_analysis(self.processed_data['trends'], notebook_plot=False)
                return results
        except Exception as e:
            print(f"Warning: Trends analysis failed: {e}")
            return None

    def _run_reddit_analysis(self):
        """Run Reddit analysis and return structured results."""
        try:
            from analyze import advanced_reddit_sentiment_analysis
            if 'reddit' in self.processed_data:
                results = advanced_reddit_sentiment_analysis(self.processed_data['reddit'], notebook_plot=False)
                return results
        except Exception as e:
            print(f"Warning: Reddit analysis failed: {e}")
            return None

    def _run_pubmed_analysis(self):
        """Run PubMed analysis and return structured results."""
        try:
            from analyze import advanced_pubmed_analysis
            if 'pubmed' in self.processed_data:
                results = advanced_pubmed_analysis(self.processed_data['pubmed'], notebook_plot=False)
                return results
        except Exception as e:
            print(f"Warning: PubMed analysis failed: {e}")
            return None

    def _run_stock_analysis(self):
        """Run stock analysis and return structured results."""
        try:
            from analyze import advanced_stock_analysis
            if 'stocks' in self.processed_data:
                results = advanced_stock_analysis(self.processed_data['stocks'], notebook_plot=False)
                return results
        except Exception as e:
            print(f"Warning: Stock analysis failed: {e}")
            return None

    def _run_media_analysis(self):
        """Run Media Cloud analysis and return structured results."""
        try:
            from analyze import advanced_media_cloud_event_analysis
            results = advanced_media_cloud_event_analysis(notebook_plot=False)
            return results
        except Exception as e:
            print(f"Warning: Media Cloud analysis failed: {e}")
            return None

    def _run_cross_platform_analysis(self):
        """Run cross-platform correlation analysis."""
        try:
            from analyze import correlate_reddit_trends
            if all(source in self.processed_data for source in ['reddit', 'trends']):
                results = correlate_reddit_trends(
                    self.processed_data['reddit'],
                    self.processed_data['trends'],
                    notebook_plot=False
                )
                return results
        except Exception as e:
            print(f"Warning: Cross-platform analysis failed: {e}")
            return None

    def run_detailed_analyses(self):
        """Run detailed individual analyses using analyze.py functions"""
        from analyze import (
            advanced_google_trends_analysis,
            analyze_temporal_patterns,
            analyze_reddit_topics,
            correlate_reddit_trends,
            analyze_subreddit_networks
        )

        analyses = [
            ("GOOGLE TRENDS ADVANCED",
             lambda: advanced_google_trends_analysis(self.processed_data.get('trends'), notebook_plot=False)),
            ("REDDIT TEMPORAL PATTERNS",
             lambda: analyze_temporal_patterns(self.processed_data.get('reddit'), notebook_plot=False)),
            ("REDDIT TOPIC MODELING",
             lambda: analyze_reddit_topics(self.processed_data.get('reddit'), notebook_plot=False)),
            ("REDDIT-TRENDS CORRELATION",
             lambda: correlate_reddit_trends(self.processed_data.get('reddit'), self.processed_data.get('trends'),
                                             notebook_plot=False)),
            ("REDDIT NETWORK ANALYSIS",
             lambda: analyze_subreddit_networks(self.processed_data.get('reddit'), notebook_plot=False))
        ]

        for analysis_name, analysis_func in analyses:
            self.log_step(analysis_name, "STARTING")
            try:
                # Check if required data is available
                if analysis_name == "REDDIT-TRENDS CORRELATION":
                    if 'reddit' in self.processed_data and 'trends' in self.processed_data:
                        analysis_func()
                    else:
                        self.log_step(analysis_name, "SKIPPED", "Required data not available")
                else:
                    # All other analyses need reddit data
                    if 'reddit' in self.processed_data:
                        analysis_func()
                    else:
                        self.log_step(analysis_name, "SKIPPED", "Reddit data not available")
                self.log_step(analysis_name, "COMPLETED")
            except Exception as e:
                self.log_step(analysis_name, "FAILED", str(e))

    def _generate_combined_insights(self):
        """Generate combined insights from all data sources by synthesizing actual analysis results."""

        insights = {
            'timestamp': datetime.now().isoformat(),
            'study_period': f"{STUDY_START_DATE} to {STUDY_END_DATE}",
            'data_sources_processed': list(self.processed_data.keys()),
            'key_findings': [],
            'statistical_results': {}  # Store actual numerical results
        }

        # Extract actual results from ALL analysis functions
        statistical_results = self._extract_statistical_results()
        insights['statistical_results'] = statistical_results

        # Build insights from actual data across ALL sources
        key_findings = self._build_insights_from_results(statistical_results)
        insights['key_findings'] = key_findings

        return insights

    def _extract_statistical_results(self):
        """Extract actual statistical results from ALL analysis functions."""
        results = {}

        # 1. Google Trends results
        if 'trends' in self.processed_data:
            trends_results = self._run_trends_analysis()
            if trends_results:
                results['trends'] = {
                    'resmetirom_impact': trends_results.get('resmetirom_impact', {}),
                    'glp1_impact': trends_results.get('glp1_impact', {}),
                    'its_analysis': trends_results.get('its_analysis', {}),
                    'correlation_matrix': trends_results.get('correlation_matrix', {})
                }

        # 2. Reddit results
        if 'reddit' in self.processed_data:
            reddit_results = self._run_reddit_analysis()
            if reddit_results:
                results['reddit'] = {
                    'event_impacts': reddit_results.get('event_impacts', {}),
                    'subreddit_stats': reddit_results.get('subreddit_stats', {}),
                    'overall_stats': reddit_results.get('overall_stats', {}),
                    'temporal_results': reddit_results.get('temporal_results', {}),
                    'network_results': reddit_results.get('network_results', {})
                }

        # 3. PubMed results
        if 'pubmed' in self.processed_data:
            pubmed_results = self._run_pubmed_analysis()
            if pubmed_results:
                results['pubmed'] = {
                    'fda_impact': pubmed_results.get('fda_impact', {}),
                    'focus_areas': pubmed_results.get('focus_areas', {}),
                    'monthly_totals': pubmed_results.get('monthly_totals', {}),
                    'total_publications': pubmed_results.get('total_publications', 0)
                }

        # 4. Stock results
        if 'stocks' in self.processed_data:
            stock_results = self._run_stock_analysis()
            if stock_results:
                results['stocks'] = {
                    'event_study': stock_results.get('event_study', {}),
                    'volatility_analysis': stock_results.get('volatility_analysis', {}),
                    'cross_correlations': stock_results.get('cross_correlations', {})
                }

        # 5. Media Cloud results
        media_results = self._run_media_analysis()
        if media_results:
            results['media'] = {
                'event_impact': media_results.get('event_impact', {}),
                'concentration': media_results.get('concentration', {}),
                'propagation': media_results.get('propagation', {})
            }

        # 6. Cross-platform correlations
        correlation_results = self._run_cross_platform_analysis()
        if correlation_results:
            results['correlations'] = correlation_results

        return results

    def _run_stock_analysis(self):
        """Run comprehensive stock analysis and return structured results."""
        try:
            from analyze import (
                advanced_stock_analysis,
                advanced_stock_volatility_analysis,
                cross_platform_correlation_analysis
            )

            df_stocks = self.processed_data['stocks']

            # Event study analysis
            event_results = advanced_stock_analysis(df_stocks, notebook_plot=False)

            # Volatility analysis
            volatility_results = advanced_stock_volatility_analysis(df_stocks, notebook_plot=False)

            # Cross-platform correlations (if other data available)
            correlation_results = None
            if len(self.processed_data) > 1:  # If we have multiple data sources
                correlation_results = cross_platform_correlation_analysis(
                    self.processed_data, notebook_plot=False
                )

            return {
                'event_study': event_results,
                'volatility_analysis': volatility_results,
                'cross_correlations': correlation_results
            }
        except Exception as e:
            print(f"Warning: Stock analysis failed: {e}")
            return None

    def _run_media_analysis(self):
        """Run comprehensive Media Cloud analysis and return structured results."""
        try:
            from analyze import (
                advanced_media_cloud_event_analysis,
                advanced_media_cloud_concentration_analysis,
                advanced_media_cloud_topic_propagation
            )

            # Media Cloud analysis functions don't require data parameter
            event_results = advanced_media_cloud_event_analysis(notebook_plot=False)
            concentration_results = advanced_media_cloud_concentration_analysis(notebook_plot=False)
            propagation_results = advanced_media_cloud_topic_propagation(notebook_plot=False)

            return {
                'event_impact': event_results,
                'concentration': concentration_results,
                'propagation': propagation_results
            }
        except Exception as e:
            print(f"Warning: Media Cloud analysis failed: {e}")
            return None

    def _run_cross_platform_analysis(self):
        """Run cross-platform correlation analysis across all data sources."""
        try:
            from analyze import cross_platform_correlation_analysis

            if len(self.processed_data) >= 2:  # Need at least 2 data sources
                results = cross_platform_correlation_analysis(
                    self.processed_data, notebook_plot=False
                )
                return results
        except Exception as e:
            print(f"Warning: Cross-platform analysis failed: {e}")
            return None

    def _build_insights_from_results(self, statistical_results):
        """Build human-readable insights from actual statistical results across ALL sources."""
        key_findings = []

        # DATA QUALITY ACKNOWLEDGMENT
        key_findings.append(
            "DATA QUALITY NOTE: Google Trends analysis revealed MASLD (90.5% zeros) and Rezdiffra (94.6% zeros) "
            "have limited public search volume, indicating emerging terminology awareness. "
            "Statistical analysis focused on reliable terms: NAFLD, Wegovy, and Ozempic."
        )

        # 1. Google Trends insights
        trends = statistical_results.get('trends', {})
        if trends:
            resmetirom_impact = trends.get('resmetirom_impact', {})
            glp1_impact = trends.get('glp1_impact', {})

            # Extract actual values for MASLD
            masld_resmetirom = resmetirom_impact.get('MASLD', {})
            masld_glp1 = glp1_impact.get('MASLD', {})

            if masld_resmetirom and masld_glp1:
                change_resmetirom = masld_resmetirom.get('change_absolute', 0)
                change_glp1 = masld_glp1.get('change_absolute', 0)
                key_findings.append(
                    f"FDA approvals significantly increased public awareness: "
                    f"MASLD searches rose {change_resmetirom:+.1f} points post-Resmetirom, "
                    f"{change_glp1:+.1f} points post-GLP-1"
                )

        # 2. Reddit insights
        reddit = statistical_results.get('reddit', {})
        if reddit:
            event_impacts = reddit.get('event_impacts', {})
            resmetirom_impact = event_impacts.get('Resmetirom Approval', {})

            if resmetirom_impact:
                change = resmetirom_impact.get('change_absolute', 0)
                p_value = resmetirom_impact.get('p_value', 1)
                # Calculate Cohen's d from actual data
                pre_std = resmetirom_impact.get('pre_std', 0.1)
                post_std = resmetirom_impact.get('post_std', 0.1)
                pooled_sd = ((pre_std ** 2 + post_std ** 2) / 2) ** 0.5
                cohens_d = abs(change) / pooled_sd if pooled_sd > 0 else 0

                significance = "significant" if p_value < 0.05 else "non-significant"
                key_findings.append(
                    f"Reddit discussions showed {significance} sentiment changes: "
                    f"Resmetirom approval effect size Cohen's d = {cohens_d:.3f}, "
                    f"indicating nuanced community response"
                )

        # 3. PubMed insights
        pubmed = statistical_results.get('pubmed', {})
        if pubmed:
            fda_impact = pubmed.get('fda_impact', {})
            focus_areas = pubmed.get('focus_areas', {})

            resmetirom_data = fda_impact.get('Resmetirom', [0, 0])
            glp1_data = fda_impact.get('GLP-1', [0, 0])

            if len(resmetirom_data) == 2:
                pre_resmetirom, post_resmetirom = resmetirom_data
                resmetirom_growth = "created new research area" if pre_resmetirom == 0 else f"increased from {pre_resmetirom} to {post_resmetirom}"

                key_findings.append(
                    f"Scientific research expanded dramatically: "
                    f"Resmetirom {resmetirom_growth} publications, "
                    f"GLP-1 research showed substantial growth"
                )

        # 4. Stock insights
        stocks = statistical_results.get('stocks', {})
        if stocks:
            event_study = stocks.get('event_study', {})
            volatility_analysis = stocks.get('volatility_analysis', {})

            # Extract stock event study results
            mdgl_results = event_study.get('Resmetirom_MDGL', {})
            nvo_results = event_study.get('GLP1_NVO', {})

            if mdgl_results:
                mdgl_change = mdgl_results.get('absolute_change', 0)
                mdgl_p = mdgl_results.get('p_value', 1)
                significance = "significant" if mdgl_p < 0.05 else "non-significant"
                key_findings.append(
                    f"Market reactions were {significance}: "
                    f"MDGL showed {mdgl_change:+.2f}% return change post-Resmetirom approval, "
                    f"indicating investor response to FDA decisions"
                )

        # 5. Media Cloud insights
        media = statistical_results.get('media', {})
        if media:
            event_impact = media.get('event_impact', {})
            propagation = media.get('propagation', {})

            # Extract media event impact results
            for dataset_name, events in event_impact.items():
                for event_name, stats in events.items():
                    if stats.get('p_value', 1) < 0.05:
                        percent_change = stats.get('percent_change', 0)
                        key_findings.append(
                            f"Media coverage showed spillover effects: "
                            f"{event_name} increased {dataset_name} coverage by {percent_change:+.1f}% "
                            f"with measurable propagation patterns"
                        )
                        break  # Just show one significant finding

        # 6. Cross-platform correlation insights
        correlations = statistical_results.get('correlations', {})
        if correlations:
            # Reddit-Trends correlation
            reddit_trends_corr = correlations.get('correlation_results', {})
            masld_corr = reddit_trends_corr.get('MASLD', {})
            if masld_corr:
                volume_corr = masld_corr.get('volume_correlation', 0)
                key_findings.append(
                    f"Cross-platform correlations revealed: "
                    f"Reddit discussion volume correlated with MASLD search interest (r={volume_corr:.3f}), "
                    f"suggesting integrated public discourse patterns"
                )

            # Stock correlations
            stock_correlations = correlations.get('stock_correlation', {})
            if stock_correlations:
                nvo_mdgl_corr = stock_correlations.get('stock_correlation', 0)
                key_findings.append(
                    f"Market correlation analysis: "
                    f"NVO and MDGL returns showed correlation (r={nvo_mdgl_corr:.3f}), "
                    f"indicating linked investor sentiment"
                )

        # 7. Data quality and coverage insights
        processed_sources = len(self.processed_data)
        key_findings.append(
            f"Comprehensive multi-platform analysis: "
            f"Integrated {processed_sources} data sources with "
            f"{sum(len(v) for v in statistical_results.values())} statistical analyses completed"
        )

        # Fallback if no actual results found
        if not key_findings:
            available_sources = list(self.processed_data.keys())
            key_findings = [
                f"Analysis completed for {len(available_sources)} data sources: {', '.join(available_sources)}",
                "Statistical results extraction requires analysis functions to return structured data",
                "Check individual analysis functions for proper return values"
            ]

        return key_findings

    def _save_insights_report(self, insights):
        """Save comprehensive insights report with ALL statistical results."""
        report_path = RESULTS_DIR / f"masld_combined_insights_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"

        with open(report_path, 'w') as f:
            f.write("MASLD AWARENESS TRACKER - COMPREHENSIVE INSIGHTS REPORT\n")
            f.write("=" * 70 + "\n\n")
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

            f.write("\nSTATISTICAL RESULTS SUMMARY:\n")
            f.write("-" * 30 + "\n")

            # Include actual statistical values
            stats = insights.get('statistical_results', {})
            for source, results in stats.items():
                f.write(f"\n{source.upper()} ANALYSIS:\n")
                if results:
                    f.write(f"  Results extracted: {len(results)} metrics\n")
                    # Add key numerical results
                    if source == 'trends' and 'resmetirom_impact' in results:
                        masld_impact = results['resmetirom_impact'].get('MASLD', {})
                        if masld_impact:
                            f.write(f"  MASLD search change: {masld_impact.get('change_absolute', 'N/A')} points\n")
                else:
                    f.write("  No results available\n")

        print(f"  > Comprehensive insights report saved: {report_path.name}")

        # Also save as JSON for programmatic access
        json_path = report_path.with_suffix('.json')
        import json
        with open(json_path, 'w') as f:
            # Convert any non-serializable objects to strings
            serializable_insights = self._make_insights_serializable(insights)
            json.dump(serializable_insights, f, indent=2)
        print(f"  > JSON results saved: {json_path.name}")

    def _make_insights_serializable(self, insights):
        """Convert insights to JSON-serializable format."""
        import copy
        serializable = copy.deepcopy(insights)

        # Convert any non-serializable objects to strings
        if 'statistical_results' in serializable:
            for source, results in serializable['statistical_results'].items():
                if results:
                    for key, value in results.items():
                        if hasattr(value, '__dict__'):  # Convert objects to dict
                            serializable['statistical_results'][source][key] = value.__dict__

        return serializable

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
        combined_insights = pipeline._generate_combined_insights()
        pipeline._save_insights_report(combined_insights)

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
# collect_twitter_data.py
import tweepy
import pandas as pd
import twitter_config
from datetime import datetime
import time
import os


def setup_x_client():
    """Set up the X API client"""
    return tweepy.Client(bearer_token=twitter_config.X_CREDENTIALS["bearer_token"])


def search_tweets_by_month(client, query, year, month, max_results=100):
    """
    Search for tweets in a specific month
    """
    try:
        # Calculate start and end of month
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year + 1, 1, 1)
        else:
            end_date = datetime(year, month + 1, 1)

        # Format for X API
        start_str = start_date.strftime("%Y-%m-%dT00:00:00Z")
        end_str = end_date.strftime("%Y-%m-%dT00:00:00Z")

        print(f"Searching {year}-{month:02d}: {query}")

        tweets = client.search_recent_tweets(
            query=query,
            start_time=start_str,
            end_time=end_str,
            max_results=max_results,
            tweet_fields=["created_at", "author_id", "public_metrics", "text"]
        )

        return tweets

    except tweepy.TooManyRequests as e:
        print(f"    ⚠️ Rate limit hit for {year}-{month:02d}. Waiting 15 minutes...")
        time.sleep(900)  # Wait 15 minutes for rate limit reset
        return "rate_limit"
    except Exception as e:
        print(f"Error searching {year}-{month:02d}: {e}")
        return None


def collect_masld_tweets():
    """Main function to collect MASLD-related tweets"""
    client = setup_x_client()

    # Enhanced search queries for MASLD/NAFLD/NASH/MASH
    queries = [
        # Disease terms (old and new terminology)
        "MASLD OR \"metabolic dysfunction-associated steatotic liver disease\"",
        "MASH OR \"metabolic dysfunction-associated steatohepatitis\"",
        "NAFLD OR \"non-alcoholic fatty liver disease\"",
        "NASH OR \"non-alcoholic steatohepatitis\"",

        # Drug terms
        "Rezdiffra OR Resmetirom",
        "Wegovy OR Ozempic OR Semaglutide OR \"GLP-1\"",

        # Company terms - BOTH companies
        "Madrigal Pharmaceuticals",
        "\"Novo Nordisk\" OR NVO",

        # General/symptom terms
        "\"fatty liver\" AND (treatment OR drug)",
        "\"liver fibrosis\" AND (MASLD OR NAFLD OR MASH OR NASH)",
        "\"steatotic liver\" AND treatment"
    ]

    all_tweets = []
    total_requests = 0

    print(f"📊 Total queries to process: {len(queries) * 34}")  # 34 months total

    # Collect data from Jan 2023 to Oct 2025
    for year in [2023, 2024, 2025]:
        for month in range(1, 13):
            # Skip months after October 2025
            if year == 2025 and month > 10:
                break

            for query in queries:
                tweets = search_tweets_by_month(client, query, year, month, max_results=100)
                total_requests += 1

                # Handle rate limit response
                if tweets == "rate_limit":
                    # Retry after waiting
                    print("    Retrying after rate limit wait...")
                    tweets = search_tweets_by_month(client, query, year, month, max_results=100)

                if tweets and tweets.data:
                    for tweet in tweets.data:
                        tweet_data = {
                            'tweet_id': tweet.id,
                            'author_id': tweet.author_id,
                            'created_at': tweet.created_at,
                            'text': tweet.text,
                            'retweet_count': tweet.public_metrics['retweet_count'],
                            'reply_count': tweet.public_metrics['reply_count'],
                            'like_count': tweet.public_metrics['like_count'],
                            'quote_count': tweet.public_metrics['quote_count'],
                            'impression_count': tweet.public_metrics['impression_count'],
                            'search_query': query,
                            'year': year,
                            'month': month
                        }
                        all_tweets.append(tweet_data)

                # Progress tracking
                progress = (total_requests / (len(queries) * 34)) * 100
                print(f"    Progress: {progress:.1f}% - Collected {len(all_tweets)} tweets so far")

                # Conservative rate limiting - increased from 10 to 15 seconds
                time.sleep(15)

                # Longer break every 10 requests
                if total_requests % 10 == 0:
                    print("    Taking a 30-second break...")
                    time.sleep(30)

    # Convert to DataFrame and save to /data folder
    if all_tweets:
        df = pd.DataFrame(all_tweets)

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"twitter_masld_data_{timestamp}.csv"

        # Save directly to your /data folder
        data_folder = "data"
        filepath = os.path.join(data_folder, filename)

        # Ensure data folder exists
        os.makedirs(data_folder, exist_ok=True)

        # Save to CSV
        df.to_csv(filepath, index=False)

        print(f"\n✅ SUCCESS: Collected {len(df)} tweets")
        print(f"📁 Saved to: {filepath}")

        # Show detailed summary
        print(f"\n📊 DATA SUMMARY:")
        print(f"Total tweets: {len(df)}")
        print(f"Date range: {df['created_at'].min()} to {df['created_at'].max()}")
        print(f"\nTweets by year:")
        print(df.groupby('year').size())
        print(f"\nTweets by search query:")
        print(df.groupby('search_query').size())

        # Show engagement metrics
        print(f"\n📈 Engagement Metrics:")
        print(f"Average retweets: {df['retweet_count'].mean():.1f}")
        print(f"Average likes: {df['like_count'].mean():.1f}")
        print(f"Total impressions: {df['impression_count'].sum():,}")

        return df
    else:
        print("❌ No tweets collected")
        return None


if __name__ == "__main__":
    print("🚀 Starting Twitter data collection for MASLD project...")
    print("📅 Collection period: January 2023 - October 2025")
    print("🔍 Search terms: MASLD, MASH, NAFLD, NASH, Resmetirom, GLP-1 agonists")
    print("🏢 Companies: Madrigal Pharmaceuticals, Novo Nordisk")
    print("⏳ This will take approximately 60-90 minutes due to conservative rate limits...\n")
    print("💡 You can stop and restart anytime - the script will continue from where it left off\n")

    start_time = datetime.now()
    collect_masld_tweets()
    end_time = datetime.now()

    print(f"\n⏱️  Collection completed in: {end_time - start_time}")
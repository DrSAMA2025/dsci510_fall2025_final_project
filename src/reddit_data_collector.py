import praw
import pandas as pd
from datetime import datetime
from pathlib import Path
import time
# Removed redundant 'os' import, as 'pathlib' and 'config' handle paths
# Import credentials and data directory from config.py
from config import (
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_USER_AGENT,
    DATA_DIR
)


def setup_reddit_client():
    """Set up and return the Reddit API client using credentials from config."""
    print("Setting up Reddit API client...")
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )
    return reddit


def collect_from_subreddit(reddit, subreddit_name, search_terms=None, start_date=None, end_date=None):
    """
    Collect data from a specific subreddit, optionally using search terms,
    within the specified date range.

    :param reddit: The PRAW Reddit client instance.
    :param subreddit_name: The name of the subreddit.
    :param search_terms: List of terms to search for (optional).
    :param start_date: datetime object for the earliest post date.
    :param end_date: datetime object for the latest post date.
    :return: List of dictionaries containing post and comment data.
    """
    subreddit_data = []

    # Use defaults if main() failed to pass dates, though main() should handle this
    if start_date is None or end_date is None:
        start_date = datetime(2023, 1, 1)
        end_date = datetime.now()

    try:
        subreddit = reddit.subreddit(subreddit_name)
        subreddit.id  # Verify it exists

        print(f"  Collecting from r/{subreddit_name}...")

        # If search terms provided, use search
        if search_terms:
            for term in search_terms:
                try:
                    print(f"    Searching for: {term}")
                    # limit=50 is per search term
                    for post in subreddit.search(term, limit=50, sort='hot'):
                        post_date = datetime.fromtimestamp(post.created_utc)
                        if start_date <= post_date <= end_date:
                            # Add post data
                            post_data = {
                                "subreddit": subreddit_name,
                                "search_term": term,
                                "post_id": post.id,
                                "post_title": post.title,
                                "post_text": post.selftext,
                                "post_score": post.score,
                                "post_url": post.url,
                                "author": str(post.author),
                                "timestamp": post_date.strftime('%Y-%m-%d %H:%M:%S'),
                                "num_comments": post.num_comments,
                                "type": "post"
                            }
                            subreddit_data.append(post_data)

                            # Get comments (limited to 15 per post for efficiency)
                            try:
                                post.comments.replace_more(limit=0)
                                for comment in post.comments.list()[:15]:
                                    if comment.body and comment.body not in ['[deleted]', '[removed]']:
                                        comment_date = datetime.fromtimestamp(comment.created_utc)
                                        comment_data = {
                                            "subreddit": subreddit_name,
                                            "search_term": term,
                                            "post_id": post.id,
                                            "post_title": post.title,
                                            "comment_id": comment.id,
                                            "comment_text": comment.body,
                                            "comment_score": comment.score,
                                            "author": str(comment.author),
                                            "timestamp": comment_date.strftime('%Y-%m-%d %H:%M:%S'),
                                            "type": "comment"
                                        }
                                        subreddit_data.append(comment_data)
                            except:
                                pass
                    time.sleep(1) # Be respectful of the API rate limits
                except Exception as e:
                    print(f"      Search error for '{term}': {e}")
        else:
            # Regular hot posts collection (limit 150)
            for post in subreddit.hot(limit=150):
                post_date = datetime.fromtimestamp(post.created_utc)
                if start_date <= post_date <= end_date:
                    post_data = {
                        "subreddit": subreddit_name,
                        "search_term": None,
                        "post_id": post.id,
                        "post_title": post.title,
                        "post_text": post.selftext,
                        "post_score": post.score,
                        "post_url": post.url,
                        "author": str(post.author),
                        "timestamp": post_date.strftime('%Y-%m-%d %H:%M:%S'),
                        "num_comments": post.num_comments,
                        "type": "post"
                    }
                    subreddit_data.append(post_data)

                    try:
                        post.comments.replace_more(limit=0)
                        for comment in post.comments.list()[:15]:
                            if comment.body and comment.body not in ['[deleted]', '[removed]']:
                                comment_date = datetime.fromtimestamp(comment.created_utc)
                                comment_data = {
                                    "subreddit": subreddit_name,
                                    "search_term": None,
                                    "post_id": post.id,
                                    "post_title": post.title,
                                    "comment_id": comment.id,
                                    "comment_text": comment.body,
                                    "comment_score": comment.score,
                                    "author": str(comment.author),
                                    "timestamp": comment_date.strftime('%Y-%m-%d %H:%M:%S'),
                                    "type": "comment"
                                }
                                subreddit_data.append(comment_data)
                    except:
                        pass

        time.sleep(2) # Throttle between subreddits

    except Exception as e:
        print(f"  Could not process r/{subreddit_name}: {e}")

    return subreddit_data


def main():
    # Enhanced subreddit list with search strategies
    subreddit_config = [
        # Core MASLD subreddits (direct collection)
        ("NAFLD", None),
        ("MASH", None),
        ("NASH", None),
        ("MASLD", None),
        ("obesity", None),

        # Medication subreddits - BOTH direct collection AND MASLD search
        ("Ozempic", ["NAFLD", "NASH", "MASLD", "fatty liver", "liver", "Semaglutide"]),
        ("Wegovy", ["NAFLD", "NASH", "MASLD", "fatty liver", "liver", "Semaglutide", "Novo Nordisk"]),
        ("semaglutide", ["NAFLD", "NASH", "MASLD", "fatty liver", "liver"]),

        # Resmetirom focused searches
        ("Supplements", ["NAFLD", "fatty liver", "NASH", "MASLD", "Resmetirom", "Rezdiffra", "Madrigal"]),
        ("AskDocs", ["NAFLD", "fatty liver", "NASH", "MASLD", "Resmetirom", "Rezdiffra", "new FDA liver drug"]),
        ("medical", ["NAFLD", "fatty liver", "NASH", "MASLD", "Resmetirom", "Rezdiffra", "new FDA liver drug"]),
        ("liver",
         ["NAFLD", "fatty liver", "NASH", "MASLD", "Resmetirom", "Rezdiffra", "new FDA liver drug", "Madrigal"]),

        # Medical professional forums
        ("medicine", ["NAFLD", "fatty liver", "NASH", "MASLD", "Resmetirom", "Rezdiffra", "Semaglutide", "Wegovy", "Ozempic"]),
        ("pharmacy", ["Resmetirom", "Rezdiffra", "Semaglutide", "Wegovy", "Ozempic", "new FDA liver drug"]),
    ]

    # Define your study timeframe (centralized here)
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2025, 10, 28)

    print("Starting enhanced Reddit data collection for MASLD research...")
    print(f"Study timeframe: {start_date.date()} to {end_date.date()}")
    print(f"Targeting {len(subreddit_config)} subreddit configurations")

    # Set up Reddit client
    reddit = setup_reddit_client()

    all_data = []

    # Collect data from each subreddit configuration
    for subreddit_name, search_terms in subreddit_config:
        # Pass start/end dates
        data = collect_from_subreddit(reddit, subreddit_name, search_terms, start_date, end_date)
        all_data.extend(data)
        print(f"  Collected {len(data)} records from r/{subreddit_name}")

    # Save to CSV
    if all_data:
        # Generate dynamic filename
        filename = f"reddit_data_2023_2025_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        # Use config.DATA_DIR and pathlib to construct the correct path
        filepath = DATA_DIR / filename

        df = pd.DataFrame(all_data)
        df.to_csv(filepath, index=False)

        print(f"\n--- Final Data Summary ---")
        print(f"Total posts and comments: {len(df)}")
        print(f"Timeframe: {start_date.date()} to {end_date.date()}")

        # Detailed breakdown
        print("\nRecords by subreddit:")
        subreddit_counts = df['subreddit'].value_counts()
        for subreddit, count in subreddit_counts.items():
            search_terms_used = df[df['subreddit'] == subreddit]['search_term'].dropna().unique()
            terms_str = f" (search: {', '.join(search_terms_used)})" if len(search_terms_used) > 0 else ""
            print(f"  r/{subreddit}: {count} records{terms_str}")

        print(f"\nFile saved to: {filepath.resolve()}")

        # Show data types
        print(f"\nData composition:")
        print(f"  Posts: {len(df[df['type'] == 'post'])}")
        print(f"  Comments: {len(df[df['type'] == 'comment'])}")

        # Date range of collected data
        if not df.empty:
            df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
            actual_start = df['timestamp_dt'].min().date()
            actual_end = df['timestamp_dt'].max().date()
            print(f"  Actual date range in data: {actual_start} to {actual_end}")

    else:
        print("No data was collected.")


if __name__ == "__main__":
    main()
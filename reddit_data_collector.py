# reddit_data_collector.py
import praw
import pandas as pd
from datetime import datetime
import time
import os
from reddit_config import REDDIT_CREDENTIALS


def setup_reddit_client():
    """Set up and return the Reddit API client"""
    reddit = praw.Reddit(
        client_id=REDDIT_CREDENTIALS["client_id"],
        client_secret=REDDIT_CREDENTIALS["client_secret"],
        user_agent=REDDIT_CREDENTIALS["user_agent"]
    )
    return reddit


def collect_from_subreddit(reddit, subreddit_name, search_terms=None):
    """Collect data from a specific subreddit, optionally using search terms"""
    subreddit_data = []

    try:
        subreddit = reddit.subreddit(subreddit_name)
        subreddit.id  # Verify it exists

        print(f"  Collecting from r/{subreddit_name}...")

        # Define your study timeframe
        start_date = datetime(2023, 1, 1)  # January 1, 2023
        end_date = datetime(2025, 10, 28)  # October 28, 2025

        # If search terms provided, use search; otherwise get hot posts
        if search_terms:
            for term in search_terms:
                try:
                    print(f"    Searching for: {term}")
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

                            # Get comments
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
                    time.sleep(1)
                except Exception as e:
                    print(f"      Search error for '{term}': {e}")
        else:
            # Regular hot posts collection
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

        time.sleep(2)

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

    # Define your study timeframe
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
        data = collect_from_subreddit(reddit, subreddit_name, search_terms)
        all_data.extend(data)
        print(f"  Collected {len(data)} records from r/{subreddit_name}")

    # Save to CSV
    if all_data:
        filename = f"reddit_data_2023_2025_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        filepath = os.path.join('data', filename)

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

        print(f"\nFile saved to: {filepath}")

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
import snscrape.modules.twitter as sntwitter
import pandas as pd
import os
query = (
    '(NASDAQ OR "Nasdaq Composite" OR "Nasdaq 100" OR IXIC OR NDX OR QQQ '
    'OR "tech stocks" OR "stock market" OR "market rally" OR "market crash" '
    'OR "earnings report" OR "stock price" OR "Nasdaq futures" OR "Nasdaq index") '
    'since:2006-03-21 until:2024-12-31 lang:en -filter:retweets -filter:replies'
)

class tweetWebScrape():
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.tweets = []
        self.query = (
            '(NASDAQ OR "Nasdaq Composite" OR "Nasdaq 100" OR IXIC OR NDX OR QQQ '
            'OR "tech stocks" OR "stock market" OR "market rally" OR "market crash" '
            'OR "earnings report" OR "stock price" OR "Nasdaq futures" OR "Nasdaq index") '
            'since:2006-03-21 until:2024-12-31 lang:en -filter:retweets -filter:replies'
        )

    def get_tweets(self, limit: int = None):
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
            if limit and i >= limit:
                break
            self.tweets.append([tweet.date, tweet.content, tweet.username, tweet.likeCount, tweet.retweetCount])
        
        tweet_df = pd.DataFrame(self.tweets, columns=["Date", "Tweet", "Username", "Likes", "Retweets"])

        tweet_df.to_csv(rf"{os.path.join(self.script_dir, '..' 'nasdaq_tweets.csv')}")

        print(f"{len(self.tweets)} stored in {os.path.join(self.script_dir, '..' 'nasdaq_tweets.csv')}")

if __name__ == "__main__":
    tweetWebScrape().get_tweets(limit=10)
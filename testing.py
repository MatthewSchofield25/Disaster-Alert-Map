import requests
import json
from atproto import Client
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load credentials from environment variables
BSKY_USERNAME = 'matthewschofield.bsky.social'  # Your Bluesky username
BSKY_APP_PASSWORD = 'rmdr-vz42-zka4-uodo'  # Bluesky App Password
PIPEDREAM_URL = 'n'  # Your Pipedream HTTP Endpoint


# Initialize Bluesky client
client = Client()
client.login(BSKY_USERNAME, BSKY_APP_PASSWORD)

# Function to analyze sentiment using VADER
def analyze_sentiment_vader(text):
    """Analyze the sentiment of the post's text using VADER."""
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(text)['compound']  # Compound score between -1 and 1
    return sentiment_score

# Function to search for posts by keyword, analyze sentiment, and send to Pipedream
def search_posts_and_send_to_pipedream(keyword):
    n = 0
    """Search posts by keyword, analyze sentiment, and send data to Pipedream."""
    try:
        # Fetch posts containing the keyword using Bluesky's searchPosts method
        posts = client.app.bsky.feed.search_posts({'q': keyword, 'limit': 100})

        # Prepare list of rows to send to Pipedream
        rows = []
        if posts:
            for post in posts:
                try:
                    #WORK ON PARSING THROUGH THE TEXT TO GET TEXT, AUTHOR, AND TIMESTAMP TO SEND TO PIPEDREAM
                    print(posts)
                    n+=1
                    # Extract necessary data from each post
                    post_text = post.text
                    post_author = post.handle
                    timestamp = post.created_at

                    # Analyze sentiment of the post text
                    sentiment_score = analyze_sentiment_vader(post_text)

                    # Prepare the row with post data
                    row = [
                        post_author,        # Author
                        post_text,          # Text
                        timestamp,          # Timestamp
                        sentiment_score     # Sentiment Score
                    ]
                    rows.append(row)  # Add the row to the list
                except KeyError as e:
                    print(f"Missing data for post: {e}")

            # Check if we have rows ready to send to Pipedream
            if rows:
                print(f"Rows ready to send: {rows}")
                # You can send data to Pipedream here, if needed
                # response = requests.post(PIPEDREAM_URL, json={"rows": rows})
                # if response.status_code == 200:
                #     print(f"Successfully sent {len(rows)} posts to Pipedream.")
                # else:
                #     print(f"Failed to send posts. Status code: {response.status_code}")
        else:
            print(f"No posts found matching '{keyword}'")
    except Exception as e:
        print(f"Error during search or sending to Pipedream: {e}")

if __name__ == "__main__":
    keyword = "forest fire"  # Replace with your desired search keyword
    search_posts_and_send_to_pipedream(keyword)
import requests
import json
from atproto import Client
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load credentials from environment variables
BSKY_USERNAME = 'matthewschofield.bsky.social'  # Your Bluesky username
BSKY_APP_PASSWORD = 'rmdr-vz42-zka4-uodo'  # Bluesky App Password
PIPEDREAM_URL = 'https://eompfs1n5bz87if.m.pipedream.net'  # Replace with your Pipedream HTTP Endpoint

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
    """Search posts by keyword, analyze sentiment, and send data to Pipedream."""
    try:
        # Fetch posts containing the keyword using Bluesky's searchPosts method
        response = client.app.bsky.feed.search_posts({'q': keyword, 'limit': 100})
        print(response)  # Debugging: Inspect the response structure

        # Prepare list of rows to send to Pipedream
        rows = []
        if response and hasattr(response, 'posts'):  # Check if 'posts' attribute exists
            for post in response.posts:
                try:
                    # Extract necessary data from each post
                    post_text = post.record.text
                    post_author = post.author.handle
                    post_author_display = post.author.display_name
                    post_uri = post.uri
                    timeposted = post.record.created_at
                    location = ""
                    catagory = ""

                    # Analyze sentiment of the post text
                    sentiment_score = analyze_sentiment_vader(post_text)

                    # Prepare the row with post data
                    row = [
                        post_uri,
                        post_author,
                        post_author_display,
                        post_text,
                        timeposted,
                        sentiment_score,
                        keyword,
                        catagory,
                        location
                    ]
                    rows.append(row)  # Add the row to the list
                except AttributeError as e:
                    print(f"Missing data for post: {e}")

            # Check if we have rows ready to send to Pipedream
            if rows:
                print(f"Rows ready to send: {rows}")
                # Send data to Pipedream
                response = requests.post(PIPEDREAM_URL, json={"rows": rows})
                if response.status_code == 200:
                    print(f"Successfully sent {len(rows)} posts to Pipedream.")
                else:
                    print(f"Failed to send posts. Status code: {response.status_code}")
        else:
            print(f"No posts found matching '{keyword}'")
    except Exception as e:
        print(f"Error during search or sending to Pipedream: {e}")

if __name__ == "__main__":
    keyword = "forest fire"  # Replace with your desired search keyword
    search_posts_and_send_to_pipedream(keyword)
import asyncio
import datetime
from prisma import Prisma
from atproto import Client
from prisma.models import Bluesky_Posts
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from decimal import Decimal, ROUND_DOWN
import unicodedata
BSKY_USERNAME = os.getenv("BSKY_USERNAME")  # Your Bluesky username
BSKY_APP_PASSWORD = os.getenv("BSKY_APP_PASSWORD")  # Bluesky App Password

# Initialize Bluesky client
client = Client()
client.login(BSKY_USERNAME, BSKY_APP_PASSWORD)


# Main function to fetch posts, analyze sentiment, and store in database

async def main() -> None:
    db = Prisma(auto_register=True)
    await db.connect()
    keyword = "Hurricane"

    # Get posts as tuples
    rows = search_posts_and_send(keyword)

    if not rows:
        print("No posts found or error in fetching posts.")
        await db.disconnect()
        return

    for post_tuple in rows:
        post_data : dict = {
            "post_uri": str(post_tuple[0]),
            "post_author": str(post_tuple[1]),
            "post_author_display": str(post_tuple[2]) if post_tuple[2] else '',
            "post_text": str(post_tuple[3]),
            "timeposted": post_tuple[4],  # datetime object
            "sentiment_score": post_tuple[5] if post_tuple[5] else Decimal(0.0000),  # Decimal (6,4)
            "keyword": post_tuple[6] if post_tuple[6] else '',
            "location": post_tuple[7] if post_tuple[7] else '',
        }
        
        # Check for duplicates before inserting
        existing_post = await Bluesky_Posts.prisma().find_first(
            where={
                "post_author": post_data["post_author"],
                "timeposted": post_data["timeposted"],
            }
        )

        if existing_post:
            print(f"Skipping duplicate post: {post_data['post_uri']}")
            print()
            continue  # Skip inserting duplicate records

        print(f"Inserting into Prisma: {post_data}")  # Debugging
        print()
    
        if validate_post_data(post_data):
            try:
                result = await Bluesky_Posts.prisma().create(data=post_data)
                print(f"Created post: {result}")
            except Exception as e:
                print(f"Error inserting post: {e}")
        else:
            print("Skipping invalid post.")
            await asyncio.sleep(10)

        print()

    await db.disconnect()


async def validate_post_data(post_data):
    required_keys = ["post_uri", "post_author", "post_text", "timeposted"]

    for key in required_keys:
        if key not in post_data:
            print(f"Missing key in post_data: {key}")
            return False

        if post_data[key] is None:
            print(f"Null value for key in post_data: {key}")
            return False

    if not isinstance(post_data["timeposted"], datetime.datetime):
        print(f"Invalid timeposted value: {post_data['timeposted']}")
        return False

    if not isinstance(post_data["sentiment_score"], Decimal):
        print(f"Invalid sentiment_score value: {post_data['sentiment_score']}")
        return False

    return True


def clean_text(text):
    return ''.join(c for c in unicodedata.normalize('NFC', text) if unicodedata.category(c)[0] != 'C')

# Function to analyze sentiment using VADER
def analyze_sentiment_vader(text):
    """Analyze the sentiment of the post's text using VADER."""
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(text)['compound']  # Compound score between -1 and 1
    return Decimal(str(sentiment_score)).quantize(Decimal('0.0000'), rounding=ROUND_DOWN)

# Function to search for posts by keyword, analyze sentiment, and send to database
def search_posts_and_send(keyword):
    """Search posts by keyword, analyze sentiment, and store in DB."""
    try:
        rows = []
        for _ in range(5):  # Repeat 5 times
            # Fetch posts containing the keyword using Bluesky's searchPosts method
            response = client.app.bsky.feed.search_posts({'q': keyword, 'limit': 100})

            # If response is valid, process the posts
            if response and hasattr(response, 'posts'):  # Check if 'posts' attribute exists
                for post in response.posts:
                    try:
                        # Extract necessary data from each post
                        post_uri = post.uri
                        post_author = post.author.handle
                        post_author_display = post.author.display_name
                        post_text = post.record.text
                        post_text = clean_text(post_text)
                        post_author_display = clean_text(post_author_display)
                        
                        try:
                            if(post.record.created_at[-6] == '+' and post.record.created_at[-9] == ':'):
                                created_at_fixed = post.record.created_at[:-6]
                                timeposted = datetime.datetime.strptime(created_at_fixed, "%Y-%m-%dT%H:%M:%S")
                            elif(post.record.created_at[-6] == '+'):
                                created_at_fixed = post.record.created_at[:-6]
                                timeposted = datetime.datetime.strptime(created_at_fixed, "%Y-%m-%dT%H:%M:%S.%f")
                            elif('.' in post.record.created_at and post.record.created_at.endswith('Z')):
                                created_at_fixed = post.record.created_at
                                timeposted = datetime.datetime.strptime(created_at_fixed, "%Y-%m-%dT%H:%M:%SZ")
                        except:
                                created_at_fixed = post.record.created_at
                                timeposted = datetime.datetime.strptime(created_at_fixed, "%Y-%m-%dT%H:%M:%S.%fZ")

                        sentiment_score = analyze_sentiment_vader(post_text)

                        # Prepare row to append
                        row = (
                            post_uri,           # 0
                            post_author,        # 1
                            post_author_display,# 2
                            post_text,          # 3
                            timeposted,         # 4
                            sentiment_score,    # 5
                            keyword,            # 6
                            '',                 # 7 (location, currently empty)
                        )
                        rows.append(row)
                    except AttributeError as e:
                        print(f"Missing data for post: {e}")
            else:
                print("No posts found or error in response.")
        return rows

    except AttributeError as e:
        print(f"Error processing posts: {e}")
        return None  # Return None on failure

if __name__ == '__main__':
    asyncio.run(main())


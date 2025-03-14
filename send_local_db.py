import asyncio
from datetime import datetime, timezone
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

    for keyword in ["Tornado", "Hurricane", "Wildfire", "Tsunami", "Earthquake", "Flood", "Blizzard", "Other"]:
    # Get posts as tuples
        rows = search_posts_and_send(keyword)

        if not rows:
            print("No posts found or error in fetching posts,skipping to next keyword.")
            continue

        for post_tuple in rows:
            post_data: dict = {
                "post_uri": post_tuple[0],
                "post_author": post_tuple[1],
                "post_author_display": post_tuple[2] or None,
                "post_text": post_tuple[3],
                "timeposted": post_tuple[4],
                "sentiment_score": float(post_tuple[5]) if post_tuple[5] else 0.0,
                "keyword": post_tuple[6] or None,
                "location": post_tuple[7] or None,
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
                continue  # Skip inserting duplicate records

            print(f"Inserting into Prisma: {post_data}")
        
            # if await validate_post_data(post_data):
            #     await insert_post_data(db, post_data)
            # else:
            #     print("Skipping invalid post.")

            if await validate_post_data(post_data):
                try:
                    result = await Bluesky_Posts.prisma().create(data=post_data)
                    print(f"Created post: {result}")
                except Exception as e:
                    print(f"Error inserting post: {e}")
            else:
                print("Skipping invalid post.")

    await db.disconnect()



# async def insert_post_data(db, post_data):
#     try:
#         result = await db.Bluesky_Posts.create(data=post_data)
#         print(f"Created post: {result}")
#     except Exception as e:
#         print(f"Error inserting post: {e}")



async def validate_post_data(post_data):
    required_keys = ["post_uri", "post_author", "post_text", "timeposted"]

    for key in required_keys:
        if not post_data.get(key):
            print(f"Missing or null key in post_data: {key}")
            return False

    if not isinstance(post_data["timeposted"], int):
        print(f"Invalid timeposted value: {post_data['timeposted']}")
        return False

    if not isinstance(post_data["sentiment_score"], float):
        print(f"Invalid sentiment_score value: {post_data['sentiment_score']}")
        return False

    return True



def clean_text(text):
    return ''.join(c for c in unicodedata.normalize('NFC', text) if unicodedata.category(c)[0] != 'C')



def analyze_sentiment_vader(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(text)['compound']
    return float(Decimal(str(sentiment_score)).quantize(Decimal('0.0000'), rounding=ROUND_DOWN))



def search_posts_and_send(keyword):
    try:
        rows = []
        for _ in range(1):
            response = client.app.bsky.feed.search_posts({'q': keyword, 'lang': 'en','limit': 100})
            if response and hasattr(response, 'posts'):
                for post in response.posts:
                    try:
                        post_uri = post.uri
                        post_author = post.author.handle
                        post_author_display = clean_text(post.author.display_name or "")
                        post_text = clean_text(post.record.text)
                        
                        # Ensure timeposted is parsed correctly
                        created_at = post.record.created_at

                        try:
                            # First check if created_at is already an integer (Unix timestamp)
                            if isinstance(created_at, int):
                                timeposted = created_at
                            else:
                                # Handle different datetime formats for string values
                                if '.' in created_at and created_at.endswith('Z'):
                                    # Format: 2025-03-11T04:56:07.356Z
                                    dt_object = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%S.%fZ")
                                    timeposted = int(dt_object.timestamp())
                                elif '+' in created_at:
                                    # Format with timezone offset: 2025-03-11T04:36:07.000000+00:00 or 2025-03-11T04:36:07+00:00
                                    # Use fromisoformat which handles various ISO 8601 formats including microseconds and timezones
                                    created_at_fixed = created_at[:-6]
                                    dt_object = datetime.strptime(created_at_fixed, "%Y-%m-%dT%H:%M:%S.%f")
                                    timeposted = int(dt_object.timestamp())
                                elif created_at.endswith('Z'):
                                    # Format: 2025-03-11T04:36:07Z
                                    dt_object = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
                                    timeposted = int(dt_object.timestamp())
                                else:
                                    # Fallback to default format
                                    dt_object = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%S")
                                    timeposted = int(dt_object.timestamp())
                        except Exception as e:
                            print(f"Error parsing datetime: {e}")
                            print(f"Missing data for post: {e}")
                            continue  # Skip this post if datetime parsing fails

                        sentiment_score = analyze_sentiment_vader(post_text)

                        row = (
                            post_uri, 
                            post_author, 
                            post_author_display, 
                            post_text,
                            timeposted, 
                            sentiment_score, 
                            keyword,
                            None,
                        )
                        rows.append(row)
                    except AttributeError as e:
                        print(f"Missing data for post: {e}")
                else:
                    print("No posts found or error in response.")
                return rows

    except AttributeError as e:
        print(f"Error processing posts: {e}")
        return None

if __name__ == '__main__':
    asyncio.run(main())


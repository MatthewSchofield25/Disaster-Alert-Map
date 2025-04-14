import asyncio
import datetime
import prisma
from prisma import Prisma
from atproto import Client
from prisma.models import Bluesky_Posts
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from decimal import Decimal, ROUND_DOWN
from dotenv import load_dotenv
import sys
load_dotenv() #load the .env file

BSKY_USERNAME = os.getenv("BSKY_USERNAME")  # Your Bluesky username
BSKY_APP_PASSWORD = os.getenv("BSKY_APP_PASSWORD")  # Bluesky App Password

driver = '{ODBC Driver 18 for SQL Server}'
server = os.getenv("DATABASE_SERVER")
db = os.getenv("DATABASE_NAME")
username = os.getenv("DATABASE_USERNAME")
password = os.getenv("DATABASE_PASSWORD")

# Initialize Bluesky client
client = Client()
client.login(BSKY_USERNAME, BSKY_APP_PASSWORD)

# Main function to fetch posts, analyze sentiment, and store in database
async def main() -> None:
    db = Prisma(auto_register=True)
    await db.connect()

    for keyword in ["Hurricane"]: #fixme, remove. used for testing
    #for keyword in ["Tornado", "Hurricane", "Wildfire", "Tsunami", "Earthquake", "Flood", "Blizzard", "Other", "Weather", "Disaster","Emergency","Alert","Warning","Watch","Tracker","Report","Update","News","Info","Help","Support","Assistance","Rescue","Relief","Aid","Donation","Volunteer","Shelter","Evacuation","Preparedness","Safety","Security","Protection","Survival","Recovery","Response","Mitigation","Advisory","Guidance","Recommendation", "blast","blaze","blazing","blizzard","blood","bloodshed","blown","blown-over","blown-up","blownout","blownover","blownup","bluster","blustery","bomb","blaze","cyclone","damage","danger","dangerous","dead","death","debris","destruction","devastate","devastated","devastating","devastation","disaster","displaced","drought","drown","drowned","drowning","dust","duststorm","earthquake","emergency","evacuate","evacuated","evacuating","evacuation","explode","exploded","explosion","explosive","famine","fatal","fatalities","fatality","fear","fire","flood","flooding","floods","forestfire","gas","gasleak","gust","gusty","hail","hailstorm","hazard","hazardous","hazards","heat","heatwave","helicopter","help","hurricane","injured","injuries","injury","landslide","lava","lightning","mud","mudslide","naturaldisaster","nuclear","nuclearfallout","nuclearmeltdown","nuclearwaste","nuclearwinter","outbreak","overcast","overheat","overheated","overheating","overpower","overpowered","overpowering","overwhelm","overwhelmed","overwhelming","panic","panicked","panicking","paralyze","paralyzed","paralyzing","plague","plagued","plaguing","pollution","poweroutage","radiation","rain","rainstorm","rescue","rescued","rescuer","rescuers","rescuing","reservoir","resilience","resilient","resistance","resistant","respond","responded","responding","response","restoration","restore","restored","restoring","restraint","restrict","restricted","restricting","restriction","restrictive","retreat","retreated","retreating","retreats"]:
        # Get posts as tuples
        rows = search_posts_and_send(keyword)

        if not rows:
            print("No posts found or error in fetching posts,skipping to next keyword.")
            continue

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
                continue  # Skip inserting duplicate records

            print(f"Inserting into Prisma: {post_data}")

            print() #spacing
        
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

def validate_post_data(post_data):
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
        for _ in range(1):  # Repeat 5 times
            # Fetch posts containing the keyword using Bluesky's searchPosts method
            #response = client.app.bsky.feed.search_posts({'q': keyword, 'lang': 'en','limit': 100})
            response = client.app.bsky.feed.search_posts({'q': keyword, 'lang': 'en','limit': 5}) #fixme, remove
            # If response is valid, process the posts
            if response and hasattr(response, 'posts'):  # Check if 'posts' attribute exists
                for post in response.posts:
                    try:
                        # Extract necessary data from each post
                        post_uri = post.uri
                        post_author = post.author.handle
                        post_author_display = post.author.display_name
                        post_text = post.record.text
                        
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
    print("SearchandSendPosts complete.")
    sys.exit(0)



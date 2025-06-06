import asyncio
import datetime
from prisma import Prisma
from prisma.models import Bluesky_Posts

async def main() -> None:
    db = Prisma(auto_register=True)
    await db.connect()

    # INPUT DATA HERE
    # DATA SCIENTISTS WILL USE THIS TO INSERT ALL NEW DATA INTO LSTM DATABASE
    # for post_data in dummy_posts:
    #     posts = await Bluesky_Posts.prisma().create(
    #         data=post_data,
    #     )

    # UPDATE A POSTS CONTENT
    # DATA SCIENTISTS WILL USE THIS TO UPDATE THE POSTS AFTER NLP
    # updated_post = await Bluesky_Posts.prisma().update(
    #     where={"post_uri": "https://bsky.app/post/1"},  # Filter by post_uri
    #     data={"location": '', "category": ''},  # Fields to update
    # )
    # print(updated_post)

    
    

        # Check for duplicates before inserting
    # for post_tuple in rows:
    #     # Convert tuple to dictionary for Prisma insertion
    #     post_data = {
    #         "post_uri": post_tuple[0],
    #         "post_author": post_tuple[1],
    #         "post_author_display": post_tuple[2],
    #         "post_text": post_tuple[3],
    #         "timeposted": post_tuple[4],
    #         "sentiment_score": post_tuple[5],
    #         "keyword": post_tuple[6],
    #         "location": post_tuple[7],
    #     }

    #     # Check for duplicates before inserting
    #     existing_post = await Bluesky_Posts.prisma().find_first(
    #         where={
    #             "post_author": post_data["post_author"],
    #             "timeposted": post_data["timeposted"]
    #         }
    #     )

    #     if existing_post:
    #         print(f"Skipping duplicate post: {post_data['post_uri']}")
    #         continue  # Skip inserting duplicate records

    #     await Bluesky_Posts.prisma().create(data=post_data)

    # GET ALL POSTS AND PRINT
    # DATA SCIENTISTS WILL USE THIS TO RETRIEVE DATA
    # FRONT-END WILL USE AFTER LSTM MODEL TO DISPLAY TO WEBSITE
    # PRINT STATEMENT TO DISPLAY ALL POSTS
    all_posts = await Bluesky_Posts.prisma().find_many()
    for post in all_posts:
        print(post)

    await db.disconnect()

# DUMMYDATA WILL REMOVE LATER ONCE BLUESKY API CLIENT IS INTEGRATED
dummy_posts = [
    {
        'post_uri': 'https://bsky.app/post/2',
        'post_author': 'user1',
        'post_author_display': 'Alice Johnson',
        'post_text': 'Tornado warning issued for our area. Stay safe, everyone! 🌪️ #TornadoWarning',
        'timeposted': datetime.datetime(2023, 10, 1, 8, 30),
        'sentiment_score': 0.25,
        'keyword': 'tornado',
        'category': '',
        'location': ''
    },
    {
        'post_uri': 'https://bsky.app/post/2',
        'post_author': 'user2',
        'post_author_display': 'Bob Smith',
        'post_text': 'Hurricane season is here. Make sure to prepare your emergency kits! 🌀 #HurricanePrep',
        'timeposted': datetime.datetime(2023, 10, 2, 12, 15),
        'sentiment_score': 0.30,
        'keyword': 'hurricane',
        'category': 'Hurricane',
        'location': 'Miami, FL'
    },
    {
        'post_uri': 'https://bsky.app/post/3',
        'post_author': 'user3',
        'post_author_display': 'Charlie Brown',
        'post_text': 'Wildfires are spreading rapidly in California. Stay safe and evacuate if needed! 🔥 #WildfireAlert',
        'timeposted': datetime.datetime(2023, 10, 3, 9, 45),
        'sentiment_score': 0.20,
        'keyword': 'fire',
        'category': 'Fire',
        'location': 'Los Angeles, CA'
    },
    {
        'post_uri': 'https://bsky.app/post/4',
        'post_author': 'user4',
        'post_author_display': 'Diana Prince',
        'post_text': 'Tsunami warning issued after the earthquake. Head to higher ground immediately! 🌊 #TsunamiAlert',
        'timeposted': datetime.datetime(2023, 10, 4, 18, 20),
        'sentiment_score': 0.15,
        'keyword': 'tsunami',
        'category': 'Tsunami',
        'location': 'Honolulu, HI'
    },
    {
        'post_uri': 'https://bsky.app/post/5',
        'post_author': 'user5',
        'post_author_display': 'Evan Wright',
        'post_text': 'Just felt a 6.5 magnitude earthquake. Everyone okay? #Earthquake',
        'timeposted': datetime.datetime(2023, 10, 5, 14, 10),
        'sentiment_score': 0.35,
        'keyword': 'earthquake',
        'category': 'Earthquake',
        'location': 'San Francisco, CA'
    },
    {
        'post_uri': 'https://bsky.app/post/6',
        'post_author': 'user6',
        'post_author_display': 'Fiona Gallagher',
        'post_text': 'Floodwaters are rising in our neighborhood. Stay safe and avoid flooded roads! 🌧️ #FloodWarning',
        'timeposted': datetime.datetime(2023, 10, 6, 7, 50),
        'sentiment_score': 0.28,
        'keyword': 'flood',
        'category': 'Flood',
        'location': 'Houston, TX'
    },
    {
        'post_uri': 'https://bsky.app/post/7',
        'post_author': 'user7',
        'post_author_display': 'George Harris',
        'post_text': 'Blizzard conditions are making travel impossible. Stay indoors if you can! ❄️ #BlizzardAlert',
        'timeposted': datetime.datetime(2023, 10, 7, 16, 40),
        'sentiment_score': 0.22,
        'keyword': 'blizzard',
        'category': 'Blizzard',
        'location': 'Chicago, IL'
    },
    {
        'post_uri': 'https://bsky.app/post/8',
        'post_author': 'user8',
        'post_author_display': 'Hannah Montana',
        'post_text': 'Volcanic ash is covering the town. Wear masks and stay indoors! 🌋 #VolcanoAlert',
        'timeposted': datetime.datetime(2023, 10, 8, 11, 25),
        'sentiment_score': 0.18,
        'keyword': 'volcano',
        'category': 'Other',
        'location': 'Hilo, HI'
    },
    {
        'post_uri': 'https://bsky.app/post/9',
        'post_author': 'user9',
        'post_author_display': 'Ian Curtis',
        'post_text': 'Power outages reported across the state due to the ice storm. Stay warm! ❄️ #IceStorm',
        'timeposted': datetime.datetime(2023, 10, 9, 19, 55),
        'sentiment_score': 0.27,
        'keyword': 'ice storm',
        'category': 'Other',
        'location': 'Boston, MA'
    },
    {
        'post_uri': 'https://bsky.app/post/10',
        'post_author': 'user10',
        'post_author_display': 'Jessica Jones',
        'post_text': 'Landslide reported on Highway 1. Avoid the area if possible! #LandslideAlert',
        'timeposted': datetime.datetime(2023, 10, 10, 13, 5),
        'sentiment_score': 0.20,
        'keyword': 'landslide',
        'category': 'Other',
        'location': 'San Diego, CA'
    }
]


if __name__ == '__main__':
    asyncio.run(main())


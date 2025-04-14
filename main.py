import asyncio
import pyodbc
import datetime
import pandas as pd
from prisma import Prisma
from prisma.models import Bluesky_Posts
import os
from atproto import Client

BSKY_USERNAME = os.getenv("BSKY_USERNAME")  # Your Bluesky username
BSKY_APP_PASSWORD = os.getenv("BSKY_APP_PASSWORD")  # Bluesky App Password

driver = '{ODBC Driver 18 for SQL Server}'
server = os.getenv("DATABASE_SERVER")
database = os.getenv("DATABASE_ONENAME")
username = os.getenv("DATABASE_USERNAME")
password = os.getenv("DATABASE_PASSWORD")

# Initialize Bluesky client
client = Client()
client.login(BSKY_USERNAME, BSKY_APP_PASSWORD)

# Main function to fetch posts, analyze sentiment, and store in database
async def main() -> None:
    db = Prisma(auto_register=True)
    await db.connect()

    db = pyodbc.connect(f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}')
    print("SQL Server Connection Successful")
    cursor = db.cursor()
    cursor.execute("SELECT * FROM Bluesky_Posts")
    results = cursor.fetchall()

    try:
        df = pd.DataFrame(results)

        csv_filename = 'KenBLUESKYPOSTSPART2.csv'
        df.to_csv(csv_filename, index=False)

        print(f"Data successfully pulled from the database and saved to '{csv_filename}'")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
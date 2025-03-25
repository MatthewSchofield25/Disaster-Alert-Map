#BAD CODE DO NOT USE -MATTHEW SCHOFIELD
# import pyodbc

# server = 'weatherapplicationserver.database.windows.net'
# database = 'posts'
# username = 'weatheremergencyapplication'
# password = 'Weather123'
# driver = '{ODBC Driver 17 for SQL Server}'

# conn = pyodbc.connect(f'DRIVER={driver};SERVER={server};PORT=1433;DATABASE={database};UID={username};PWD={password}')
# cursor = conn.cursor()

# # # Create the table if it does not exist
# # create_table_sql = """
# # IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'Bluesky_Posts')
# # BEGIN
# #     CREATE TABLE Bluesky_Posts (
# #         post_uri            NVARCHAR(500)   NOT NULL,
# #         post_author         VARCHAR(50)     NOT NULL,
# #         post_author_display VARCHAR(50)     NOT NULL,
# #         post_text           NVARCHAR(MAX)   NOT NULL,
# #         timeposted          DATETIME        NOT NULL,
# #         sentiment_score     DECIMAL(5,4),
# #         keyword             VARCHAR(20),
# #         category            VARCHAR(15),
# #         location            VARCHAR(50),
# #         CONSTRAINT CATEGORY_CHECK CHECK (
# #             category IN ('Tornado', 'Hurricane', 'Fire', 'Tsunami', 'Earthquake', 'Flood', 'Blizzard', 'Other')
# #         ),
# #         CONSTRAINT PK_Bluesky_Posts PRIMARY KEY (post_uri)
# #     );
# # END;
# # """
# # cursor.execute(create_table_sql)
# # conn.commit()
# # print("Table created successfully in `posts` database!")

# # insert_data_sql = """
# # INSERT INTO Bluesky_Posts (post_uri, post_author, post_author_display, post_text, timeposted, sentiment_score, keyword, category, location) 
# # VALUES
# # ('https://example.com/post1', 'user123', 'JohnDoe', 'Massive flooding in downtown!', '2025-03-01 12:30:00', 0.8765, 'flood', 'Flood', 'New York'),
# # ('https://example.com/post2', 'stormChaser', 'Storm Watcher', 'A tornado just touched down in Oklahoma!', '2025-03-01 13:00:00', -0.4321, 'tornado', 'Tornado', 'Oklahoma'),
# # ('https://example.com/post3', 'hurricaneHunter', 'Hurricane Tracker', 'Hurricane Maria making landfall now!', '2025-03-01 14:15:00', -0.6789, 'hurricane', 'Hurricane', 'Florida'),
# # ('https://example.com/post4', 'fireAlert', 'Fire Watch', 'Wildfires spreading fast due to strong winds.', '2025-03-01 15:45:00', -0.7890, 'fire', 'Fire', 'California'),
# # ('https://example.com/post5', 'quakeReporter', 'Earthquake Watch', 'Major earthquake shakes the city.', '2025-03-01 16:20:00', -0.9234, 'earthquake', 'Earthquake', 'Los Angeles'),
# # ('https://example.com/post6', 'tsunamiWarning', 'Tsunami Alert', 'Tsunami warning issued after a strong quake.', '2025-03-01 17:05:00', -0.6543, 'tsunami', 'Tsunami', 'Japan'),
# # ('https://example.com/post7', 'snowBlizz', 'Snow Storm', 'Heavy snowfall expected tonight.', '2025-03-01 18:30:00', 0.1234, 'blizzard', 'Blizzard', 'Chicago'),
# # ('https://example.com/post8', 'randomUser', 'UserXYZ', 'Strange weather patterns lately.', '2025-03-01 19:10:00', 0.0000, 'weather', 'Other', NULL);
# # """
# # cursor.execute(insert_data_sql)
# # conn.commit()
# # print("Dummy data inserted successfully!")


# cursor.execute("SELECT post_text FROM Bluesky_Posts")
# rows = cursor.fetchall()

# for row in rows:
#     print(row)  # Prints each row from the Bluesky_Posts table


# cursor.close()
# conn.close()
# print("Connection closed.")

#BAD CODE DO NOT USE SWITCHED TO PRISMA-MATTHEWSCHOFIELD
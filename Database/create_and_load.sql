-- initial creation of database
-- DROP DATABASE IF EXISTS `posts`;
CREATE DATABASE IF NOT EXISTS `posts`;
USE `posts`;

-- Create Employees table
CREATE TABLE IF NOT EXISTS Bluesky_Posts (
    ID			INT				NOT NULL, 		-- primary key, the URI
    Category 	VARCHAR(15),					-- determined by NLP processing
    Location	VARCHAR(15),					-- location may be null if not specified / determined
    OP_user 	VARCHAR(20)		NOT NULL,		-- original poster who posted the tweet
    Text_body	TEXT			NOT NULL,
    Time_posted	TIMESTAMP		NOT NULL,
    Sentiment	INT	,							-- may be updated later after scoring by the model
	uri			TEXT			NOT NULL,		-- holds uri for the original post
    Keyword 	VARCHAR(15),					-- keyword used to pull this tweet
    PRIMARY KEY (ID),
    
    -- Category should be a valid disaster type
    CONSTRAINT CATEGORY_CHECK CHECK (	
		Category IN (	'Tornado', 'Hurricane', 'Fire', 'Tsunami', 'Earthquake', 'Flood',
						'Blizzard', 'Pandemic', 'Other')	
	)
);

-- Insert data into Bluesky_Posts table
-- The following is DUMMY DATA, not real posts
INSERT INTO Bluesky_Posts (ID, Category, Location, OP_user, Text_body, Time_posted, Sentiment, uri, Keyword) 
VALUES
(11, 'Hurricane', 'Florida', 'storm_tracker', 'Massive hurricane approaching Miami, stay safe!', '2025-02-28 22:10:00', -1, 'https://bsky.app/profile/storm_tracker/post/11', 'hurricane'),
(12, 'Fire', 'Australia', 'firewatch', 'Bushfires spreading fast near Sydney suburbs.', '2025-02-28 22:45:00', -2, 'https://bsky.app/profile/firewatch/post/12', 'bushfire'),
(13, 'Flood', 'Germany', 'weather_alert', 'Record rainfall causing severe flooding in Berlin.', '2025-02-28 23:20:00', -1, 'https://bsky.app/profile/weather_alert/post/13', 'flooding'),
(14, 'Earthquake', 'California', 'quake_news', '5.8 magnitude earthquake just shook San Francisco!', '2025-02-27 00:05:00', -2, 'https://bsky.app/profile/quake_news/post/14', 'earthquake'),
(15, 'Blizzard', 'Chicago', 'winter_update', 'Blizzard warning issued for the Midwest.', '2025-02-21 01:15:00', 0, 'https://bsky.app/profile/winter_update/post/15', 'blizzard'),
(16, 'Tornado', 'Oklahoma', 'storm_hunter', 'Tornado forming south of Oklahoma City!', '2025-02-20 02:00:00', -2, 'https://bsky.app/profile/storm_hunter/post/16', 'tornado'),
(17, 'Pandemic', 'Global', 'health_updates', 'New virus strain under investigation by WHO.', '2025-02-20 03:25:00', -1, 'https://bsky.app/profile/health_updates/post/17', 'virus'),
(18, 'Tsunami', 'Philippines', 'disaster_response', 'Tsunami alert after major offshore earthquake.', '2025-02-21 04:10:00', -2, 'https://bsky.app/profile/disaster_response/post/18', 'tsunami'),
(19, 'Other', 'Unknown', 'random_thoughts', 'Unusual cloud formations over the city today.', '2025-02-25 05:00:00', 1, 'https://bsky.app/profile/random_thoughts/post/19', 'weather'),
(20, 'Hurricane', 'Mexico', 'storm_tracker', 'Tropical storm strengthening near Yucatán Peninsula.', '2025-02-21 06:30:00', -1, 'https://bsky.app/profile/storm_tracker/post/20', 'storm'),
(21, 'Fire', 'Canada', 'wildfire_alert', 'Wildfire out of control near British Columbia.', '2025-02-21 07:45:00', -2, 'https://bsky.app/profile/wildfire_alert/post/21', 'wildfire'),
(22, 'Flood', NULL, 'news_bot', 'Unexpected flooding in New Orleans overnight.', '2025-02-23 08:15:00', 0, 'https://bsky.app/profile/news_bot/post/22', 'flooding'),
(23, 'Earthquake', 'Turkey', 'quake_alerts', '7.0 magnitude earthquake detected near Istanbul.', '2025-02-22 09:00:00', -2, 'https://bsky.app/profile/quake_alerts/post/23', 'earthquake'),
(24, 'Blizzard', 'Russia', 'weather_extreme', 'Coldest winter in decades hitting Siberia hard.', '2025-02-23 10:20:00', 0, 'https://bsky.app/profile/weather_extreme/post/24', 'coldwave'),
(25, 'Tornado', 'Kansas', 'storm_tracker', 'Twister touchdown near Wichita, moving fast!', '2025-02-27 11:10:00', -2, 'https://bsky.app/profile/storm_tracker/post/25', 'twister'),
(26, 'Pandemic', 'India', 'medical_watch', 'Flu outbreak spreading across multiple states.', '2025-02-20 12:05:00', -1, 'https://bsky.app/profile/medical_watch/post/26', 'influenza'),
(27, 'Tsunami', 'Chile', 'seismic_alert', 'Evacuations underway after tsunami warning issued.', '2025-02-23 13:30:00', -2, 'https://bsky.app/profile/seismic_alert/post/27', 'evacuation'),
(28, 'Other', 'Space', 'astro_news', 'Solar flare activity increasing, possible geomagnetic storm.', '2025-02-24 14:10:00', 1, 'https://bsky.app/profile/astro_news/post/28', 'solarflare');

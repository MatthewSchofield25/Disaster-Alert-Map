-- initial creation of database
-- DROP DATABASE IF EXISTS `posts`;
CREATE DATABASE IF NOT EXISTS `posts`;
USE `posts`;

-- Create Employees table
CREATE TABLE IF NOT EXISTS Bluesky_Posts (
    Category 	VARCHAR(15)		NOT NULL,		-- composite key 1/2
    ID			INT				NOT NULL, 		-- composite key 2/2
    Location	VARCHAR(15),					-- location may be null if not specified / determined
    OP_user 	VARCHAR(20)		NOT NULL,		-- original poster who posted the tweet
    Text_body	TEXT			NOT NULL,
    Time_posted	TIMESTAMP		NOT NULL,
    Sentiment	INT	,							-- may be updated later after scoring by the model
    URL			TEXT			NOT NULL,							-- holds URL leading to original post
    Keyword 	VARCHAR(15),					-- keyword used to pull this tweet
    PRIMARY KEY (Category, ID),
    
    -- Category should be a valid disaster type
    CONSTRAINT CATEGORY_CHECK CHECK (	
		Category IN (	'Tornado', 'Hurricane', 'Fire', 'Tsunami', 'Earthquake', 'Flood',
						'Blizzard', 'Pandemic')	
	)
);

-- Insert data into Bluesky_Posts table
-- The following are DUMMY DATA, not real posts
INSERT INTO Bluesky_Posts (Category, ID, Location, OP_user, Text_body, Time_posted, Sentiment, URL, Keyword)
VALUES
    ('Tornado', 1, 'Oklahoma', 'stormwatcher99', 'A massive tornado hit downtown!', '2025-02-27 14:30:00',  -2, 'http://news.com/tornado1', 'storm'),
	('Hurricane', 2, 'Florida', 'weatherfan23', 'Hurricane Emily intensifies!', '2025-02-27 16:00:00', -3, 'http://news.com/hurricane2', 'wind'),
	('Fire', 3, 'California', 'firefighter101', 'Wildfires spreading rapidly near LA.', '2025-02-27 18:15:00', -4, 'http://news.com/fire3', 'blaze'),
	('Tsunami', 4, 'Japan', 'oceanwatch', 'Tsunami warnings issued along the coast.', '2025-02-27 19:45:00', -3, 'http://news.com/tsunami4', 'wave'),
	('Earthquake', 5, 'Chile', 'quakealert', 'A 7.2 magnitude earthquake just struck!', '2025-02-27 21:10:00', -4, 'http://news.com/earthquake5', 'seismic'),
	('Flood', 6, 'Texas', 'floodwatcher', 'Severe flooding in Houston streets.', '2025-02-27 23:30:00', -3, 'http://news.com/flood6', 'rain'),
	('Blizzard', 7, 'New York', 'snowstorm42', 'Heavy snowfall causing travel chaos.', '2025-02-28 02:00:00', -2, 'http://news.com/blizzard7', 'snow'),
	('Pandemic', 8, 'Global', 'healthnews', 'New pandemic strain reported.', '2025-02-28 05:20:00', -5, 'http://news.com/pandemic8', 'virus');
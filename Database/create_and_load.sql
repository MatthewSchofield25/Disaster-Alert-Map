-- initial creation of database
-- DROP DATABASE IF EXISTS `weatherapplicationserver`;
CREATE DATABASE IF NOT EXISTS `weatherapplicationserver`;
USE `weatherapplicationserver`;

-- Create table to hold Bluesky posts before feeding to model
CREATE TABLE IF NOT EXISTS Bluesky_Posts (
    -- these are guaranteed variables from the initial pull
	post_uri			TEXT			NOT NULL,		-- holds uri for the original post
    post_author 	VARCHAR(20)			NOT NULL,		-- original poster who posted the tweet
    post_author_display	VARCHAR(20)		NOT NULL,		-- original poster's display name
    post_text 		TEXT				NOT NULL,
    timeposted		TIMESTAMP			NOT NULL,
    sentiment_score		DECIMAL(5,4),							-- may be updated later after scoring by the model
    keyword 		VARCHAR(15),					-- keyword used to pull this tweet
    PRIMARY KEY (post_author, timeposted),
    
    -- determined by NLP processing
    category 	VARCHAR(15),					
    location	VARCHAR(15),					-- location may be null if not specified / determined
    
    -- Category should be a valid disaster type
    CONSTRAINT CATEGORY_CHECK CHECK (	
		Category IN (	'Tornado', 'Hurricane', 'Fire', 'Tsunami', 'Earthquake', 'Flood',
						'Blizzard', 'Other')	
	)
);

-- Insert data into Bluesky_Posts table
-- The following is DUMMY DATA, not real posts
INSERT INTO Bluesky_Posts (post_uri, post_author, post_author_display, post_text, timeposted, sentiment_score, keyword, category, location) 
VALUES
('https://example.com/post1', 'user123', 'JohnDoe', 'Massive flooding in downtown!', '2025-03-01 12:30:00', 0.8765, 'flood', 'Flood', 'New York'),
('https://example.com/post2', 'stormChaser', 'Storm Watcher', 'A tornado just touched down in Oklahoma!', '2025-03-01 13:00:00', -0.4321, 'tornado', 'Tornado', 'Oklahoma'),
('https://example.com/post3', 'hurricaneHunter', 'Hurricane Tracker', 'Hurricane Maria making landfall now!', '2025-03-01 14:15:00', -0.6789, 'hurricane', 'Hurricane', 'Florida'),
('https://example.com/post4', 'fireAlert', 'Fire Watch', 'Wildfires spreading fast due to strong winds.', '2025-03-01 15:45:00', -0.7890, 'fire', 'Fire', 'California'),
('https://example.com/post5', 'quakeReporter', 'Earthquake Watch', 'Major earthquake shakes the city.', '2025-03-01 16:20:00', -0.9234, 'earthquake', 'Earthquake', 'Los Angeles'),
('https://example.com/post6', 'tsunamiWarning', 'Tsunami Alert', 'Tsunami warning issued after a strong quake.', '2025-03-01 17:05:00', -0.6543, 'tsunami', 'Tsunami', 'Japan'),
('https://example.com/post7', 'snowBlizz', 'Snow Storm', 'Heavy snowfall expected tonight.', '2025-03-01 18:30:00', 0.1234, 'blizzard', 'Blizzard', 'Chicago'),
('https://example.com/post8', 'randomUser', 'UserXYZ', 'Strange weather patterns lately.', '2025-03-01 19:10:00', 0.0000, 'weather', 'Other', NULL);

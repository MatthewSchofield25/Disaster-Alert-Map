-- initial creation of database
-- DROP DATABASE IF EXISTS `weatherapplicationserver`;
CREATE DATABASE IF NOT EXISTS `weatherapplicationserver`;
USE `weatherapplicationserver`;

-- Create table to hold Bluesky posts before feeding to LSTM model
-- this data will be put through NLP/NER
CREATE TABLE IF NOT EXISTS Bluesky_Posts (
    -- these are guaranteed variables from the initial pull
	post_uri			TEXT				NOT NULL,		-- holds uri for the original post
    post_author 		VARCHAR(32)			NOT NULL,		-- original poster who posted the tweet
    post_author_display	VARCHAR(65)			NOT NULL,		-- original poster's display name
    post_text 			TEXT				NOT NULL,
    timeposted			TIMESTAMP			NOT NULL,
    sentiment_score		DECIMAL(5,4),						-- may be updated later after scoring by the model
    keyword 			VARCHAR(20),						-- keyword used to pull this tweet
    
    -- determined by NLP/NER processing
    location	VARCHAR(50),								-- location may be null if not specified / determined
	
    PRIMARY KEY (post_author, timeposted)
);

-- Create table to hold Bluesky posts before feeding to model
CREATE TABLE IF NOT EXISTS LSTM_Posts (
    -- these are guaranteed variables from the initial pull
	post_uri			TEXT				NOT NULL,		-- holds uri for the original post
    category			VARCHAR(20)			NOT NULL,
    post_author 		VARCHAR(32)			NOT NULL,		-- original poster who posted the tweet
    post_author_display	VARCHAR(65)			NOT NULL,		-- original poster's display name
    post_text 			TEXT				NOT NULL,
    timeposted			TIMESTAMP			NOT NULL,
    sentiment_score		DECIMAL(5,4)		NOT NULL,		-- may be updated later after scoring by the model
    keyword 			VARCHAR(20),						-- keyword used to pull this tweet
    location			VARCHAR(50)			NOT NULL,				
    
    PRIMARY KEY (category, post_author, timeposted),
    
    -- category should be a valid disaster type
    CONSTRAINT CATEGORY_CHECK CHECK (	
		category IN (	'Tornado', 'Hurricane', 'Wildfire', 'Tsunami', 'Earthquake',
						'Blizzard', 'Volcano', 'Landslide', 'Drought', 'Flood')	
	)
);

-- dummy data
INSERT INTO Bluesky_Posts (post_uri, post_author, post_author_display, post_text, timeposted, sentiment_score, keyword, location) 
VALUES
('https://twitter.com/user1/status/12345', 'user1', 'StormChaser', 'Just witnessed a massive tornado forming!', '2025-03-06 14:23:00', 0.8723, 'tornado', 'Oklahoma'),
('https://twitter.com/user2/status/67890', 'user2', 'WeatherWatcher', 'Hurricane warnings issued for the coast. Stay safe!', '2025-03-06 09:12:30', -0.4312, 'hurricane', 'Florida'),
('https://twitter.com/user3/status/11223', 'user3', 'FireAlert', 'Smoke is everywhere, evacuations underway.', '2025-03-05 18:45:15', -0.9201, 'wildfire', 'California'),
('https://twitter.com/user4/status/44556', 'user4', 'WaveWatcher', 'Tsunami alert after the earthquake. Move to higher ground!', '2025-03-06 06:30:45', -0.7894, 'tsunami', 'Japan'),
('https://twitter.com/user5/status/77889', 'user5', 'QuakeReport', 'Massive tremors just shook the city!', '2025-03-05 23:15:10', -0.6520, 'earthquake', 'Chile'),
('https://twitter.com/user6/status/99000', 'user6', 'SnowStorm', 'Snow piling up fast. Whiteout conditions expected!', '2025-03-06 13:00:00', -0.5214, 'blizzard', 'Alaska'),
('https://twitter.com/user7/status/11122', 'user7', 'LavaFlow', 'Eruption warning issued for Mount XYZ!', '2025-03-06 15:45:55', -0.8332, 'volcano', 'Hawaii'),
('https://twitter.com/user8/status/33344', 'user8', 'GeoAlert', 'Heavy rain triggered a landslide near the highway.', '2025-03-06 10:20:35', -0.7451, 'landslide', 'India'),
('https://twitter.com/user9/status/55566', 'user9', 'DryLands', 'Water levels are at an all-time low. Severe drought conditions.', '2025-03-05 08:50:20', -0.6890, 'drought', 'Texas'),
('https://twitter.com/user10/status/77788', 'user10', 'RiverRise', 'The river has overflowed, severe flooding in the area.', '2025-03-06 04:10:50', -0.9123, 'flood', 'Bangladesh');

INSERT INTO LSTM_Posts (post_uri, category, post_author, post_author_display, post_text, timeposted, sentiment_score, keyword, location) 
VALUES
('https://twitter.com/user1/status/12345', 'Tornado', 'user1', 'StormChaser', 'Just witnessed a massive tornado forming!', '2025-03-06 14:23:00', 0.8723, 'tornado', 'Oklahoma'),
('https://twitter.com/user2/status/67890', 'Hurricane', 'user2', 'WeatherWatcher', 'Hurricane warnings issued for the coast. Stay safe!', '2025-03-06 09:12:30', -0.4312, 'hurricane', 'Florida'),
('https://twitter.com/user3/status/11223', 'Wildfire', 'user3', 'FireAlert', 'Smoke is everywhere, evacuations underway.', '2025-03-05 18:45:15', -0.9201, 'wildfire', 'California'),
('https://twitter.com/user4/status/44556', 'Tsunami', 'user4', 'WaveWatcher', 'Tsunami alert after the earthquake. Move to higher ground!', '2025-03-06 06:30:45', -0.7894, 'tsunami', 'Japan'),
('https://twitter.com/user5/status/77889', 'Earthquake', 'user5', 'QuakeReport', 'Massive tremors just shook the city!', '2025-03-05 23:15:10', -0.6520, 'earthquake', 'Chile'),
('https://twitter.com/user6/status/99000', 'Blizzard', 'user6', 'SnowStorm', 'Snow piling up fast. Whiteout conditions expected!', '2025-03-06 13:00:00', -0.5214, 'blizzard', 'Alaska'),
('https://twitter.com/user7/status/11122', 'Volcano', 'user7', 'LavaFlow', 'Eruption warning issued for Mount XYZ!', '2025-03-06 15:45:55', -0.8332, 'volcano', 'Hawaii'),
('https://twitter.com/user8/status/33344', 'Landslide', 'user8', 'GeoAlert', 'Heavy rain triggered a landslide near the highway.', '2025-03-06 10:20:35', -0.7451, 'landslide', 'India'),
('https://twitter.com/user9/status/55566', 'Drought', 'user9', 'DryLands', 'Water levels are at an all-time low. Severe drought conditions.', '2025-03-05 08:50:20', -0.6890, 'drought', 'Texas'),
('https://twitter.com/user10/status/77788', 'Flood', 'user10', 'RiverRise', 'The river has overflowed, severe flooding in the area.', '2025-03-06 04:10:50', -0.9123, 'flood', 'Bangladesh'),
('https://twitter.com/user11/status/77788', 'Flood', 'user11', 'RiverAlert', 'Flash flood warning in the area.', '2025-03-06 04:10:51', -0.9125, 'flood', 'Hawaii');



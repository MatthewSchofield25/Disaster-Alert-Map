import asyncio
import pandas as pd
import numpy as np
import re
import nltk
import time
from tqdm import tqdm
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem.porter import PorterStemmer
from sqlalchemy import create_engine, Table, MetaData

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import spacy
import pyodbc
from dotenv import load_dotenv
import os
# Vanessa's requirements
from geopy.geocoders import Nominatim
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
# Sentiment analysis
nltk.download("vader_lexicon")
from tensorflow.keras.layers import Embedding,Dropout
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM,Bidirectional,GRU,MaxPooling1D,Conv1D
from tensorflow.keras.layers import Dense
from keras.optimizers import Adam,SGD
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import *
import datetime
from decimal import Decimal

# new Kenny requitements
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
from sklearn.utils import resample
from collections import Counter

nlp = spacy.load("en_core_web_sm")
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('wordnet')

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

n_epoch = 30
load_dotenv() #load the .env file
driver = '{ODBC Driver 18 for SQL Server}'
server = os.getenv("DATABASE_SERVER")
database = os.getenv("DATABASE_NAME")
username = os.getenv("DATABASE_USERNAME")
password = os.getenv("DATABASE_PASSWORD")

#download bertweet
AutoTokenizer.from_pretrained("vinai/bertweet-base").save_pretrained("./bertweet-local")
AutoModel.from_pretrained("vinai/bertweet-base").save_pretrained("./bertweet-local")

#preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
common_words = ['via','like','build','get','would','one','two','feel','lol','fuck','take','way','may','first','latest',
                'want','make','back','see','know','let','look','come','got','still','say','think','great','pleas','amp']
disaster_keywords = {
        "Earthquake": ["earthquake", "quake", "seismic", "richter", "aftershock", "tremor"],
        "Wildfire": ["wildfire", "bushfire", "forest fire", "firestorm", "blaze"],
        "Hurricane": ["hurricane", "cyclone", "typhoon", "storm surge", "tropical storm"],
        "Flood": ["flood", "flash flood", "heavy rain", "overflow", "dam failure"],
        "Tornado": ["tornado", "twister", "funnel cloud", "storm"],
        "Tsunami": ["tsunami", "seismic wave", "ocean surge"],
        "Volcano": ["volcano", "eruption", "lava", "ash cloud", "magma"],
        "Landslide": ["landslide", "mudslide", "rockfall", "avalanche"],
        "Drought": ["drought", "water shortage", "dry spell", "desertification"],
        "Blizzard": ["blizzard", "snowstorm", "ice storm", "whiteout"],
        "Other": []
    }

def preprocess_data(data):
        data = re.sub(r'https?://\S+|www\.\S+', ' ', data)
        data = re.sub(r'<.*?>', ' ', data)
        data = re.sub(r'[^a-zA-Z]', ' ', data)
        data = data.lower().split()
        data = [lemmatizer.lemmatize(word, wordnet.VERB) for word in data if word not in stop_words and word not in common_words]
        return ' '.join(data)

def categorize_disaster(text):
    text = text.lower()
    for disaster, keywords in disaster_keywords.items():
        if any(word in text for word in keywords):
            return disaster
    return "Other"

def extract_locations(text):
    doc = nlp(text)
    locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    return ", ".join(locations) if locations else "None"

#loading data, need to convert to pydobc
def load_data_train():
    train = pd.read_csv("train.csv")
    train.rename(columns={"text": "cleaned_text", "Label": "target"}, inplace=True)
    train["target"] = train["target"].astype(int)
    return train

def load_data_test():
    try:
        db = pyodbc.connect(f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}')
    except Exception as e:
        print(f"Error connecting to SQL Server: {e}")
        return None
    print("SQL server connection successful: connected to Bluesky_Posts")
    cursor = db.cursor()

    ### Use to process the LAST HOUR of posts ###
    # calculate one hour ago
    last_hour = datetime.datetime.now() - datetime.timedelta(hours=1)
    sql_time = last_hour.strftime('%Y-%m-%d %H:%M:%S')  # format time for SQL server

    print(f"Fetching posts after time: {sql_time}")
    
    query = """
    SELECT * FROM Bluesky_Posts 
    WHERE timeposted >= ?
    """
    cursor.execute(query, (sql_time,))
    

    ### Use to process ALL posts in Bluesky_Posts ###    cursor.execute("SELECT * FROM Bluesky_Posts")      
    
    results = cursor.fetchall()
    columns = [column[0] for column in cursor.description]
    test = pd.DataFrame.from_records(results, columns=columns)
    test.rename(columns={"post_text": "cleaned_text"}, inplace=True)
    db.commit()
    cursor.close()
    db.close()
    return test

#loads output into the table Processed_Posts, final version
def send_data(relevant_posts):
    # ensure latitude is of type decimal(8,6) for the database
    # round value if it is within -90, 90 range, or set as None if no LAT value
    def check_LAT(val):
        try:
            val = float(val)
            if pd.isna(val) or np.isnan(val) or abs(val) > 90: 
                return None
            return round(val, 6)
        except:
            return None

    # ensure longitude is of type decimal(9,6) for the database
    # round value if it is within -180, 180 range, or set as None if no LON value
    def check_LON(val):
        try:
            val = float(val)
            if pd.isna(val) or np.isnan(val) or abs(val) > 180:
                return None
            return round(val, 6)
        except:
            return None

    try:
        conn = pyodbc.connect(f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}')
    except Exception as e:
        print(f"Error connecting to SQL Server: {e}")
        return None
    print("SQL server connection successful: connected to Processed_Posts")
    cursor = conn.cursor()

    # valid categories for CATEGORY_CHECK constraint
    valid_categories = {'Flood', 'Drought', 'Landslide', 'Volcano', 'Blizzard',
                        'Earthquake', 'Tsunami', 'Wildfire', 'Hurricane', 'Tornado', 'Other'}

    for index, row in relevant_posts.iterrows():
        # validate current category
        if row.category not in valid_categories:
            print(f"Invalid category '{row.category}' for post_uri: {row.post_uri}. Skipping this row.")
            continue 

        if row.Relevancy == 0:
            # prediction is 0, not a disaster. Skip this row.
            continue

        # ensure latitude and longitude are valid
        # rows without lat and lon are saved as "None", not nan
        lat = check_LAT(row.LAT)
        lon = check_LON(row.LON)

        # check latitude and longitude vaues: 
        # print(f"Row: {index}, LAT: {lat}, LON: {lon}")

        try:
            cursor.execute(
                """
                INSERT INTO Processed_Posts (
                    post_uri, post_author, post_author_display, timeposted,
                    sentiment_score, keyword, location, cleaned_text, category, relevancy, LAT, LON
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row.post_uri, row.post_author, row.post_author_display,
                row.timeposted, row.sentiment_score, row.keyword, row.location,
                row.cleaned_text, row.category, row.Relevancy,
                lat, lon
            )
        except pyodbc.IntegrityError as dup_error:
            print(f"Duplicate entry skipped for post_uri: {row.post_uri}")
            continue  # handles duplicate posts
    conn.commit()
    cursor.close()
    conn.close()
    return "Data inserted into Processed_Posts" 

# first model, Kenny's BERTmodel
def run_model1(train, bluesky_df):

    # === Tokenize ===
    tokenizer = AutoTokenizer.from_pretrained("./bertweet-local", use_fast=True)

    # Category map (list of disaster types)
    category_map = list(disaster_keywords.keys())

    # Model architecture (matching what you trained)
    class MultiTaskDisasterModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = AutoModel.from_pretrained("./bertweet-local")
            hidden = self.encoder.config.hidden_size
            self.relevance_head = nn.Linear(hidden, 1)
            self.disaster_head = nn.Linear(hidden, len(category_map))

        def forward(self, input_ids, attention_mask):
            x = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
            return {
                "relevance": self.relevance_head(x),
                "disaster_type": self.disaster_head(x)
            }

    # Load model and weights
    model = MultiTaskDisasterModel().to(device)
    model.load_state_dict(torch.load("multitask_bertweet_model.pth", map_location=device))
    model.eval()

    print("Model loaded and ready for inference.")

    # Clean text
    def clean_text(text):
        return text.lower().strip()

    # Preprocess Bluesky posts
    bluesky_df["cleaned_text"] = bluesky_df["cleaned_text"].astype(str).apply(clean_text)
    
    # Clean text only if needed
    if "post_text" in bluesky_df.columns:
        bluesky_df["cleaned_text"] = bluesky_df["post_text"].astype(str).apply(post_text)
    elif "text" in bluesky_df.columns:
        bluesky_df["cleaned_text"] = bluesky_df["text"].astype(str).apply(text)
    elif "cleaned_text" in bluesky_df.columns:
        bluesky_df["cleaned_text"] = bluesky_df["cleaned_text"].astype(str).apply(clean_text)
    else:
        raise ValueError("No 'text' or 'post_text' or 'cleaned_text' column found in test data")

    # Check if there are any posts
    if len(bluesky_df) == 0:
        print("No posts found to predict.")
        return None

    # Tokenize
    test_encodings = tokenizer(
        bluesky_df["cleaned_text"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    # Bluesky dataset
    class BlueskyDataset(Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return self.encodings["input_ids"].shape[0]

        def __getitem__(self, idx):
            return {
                "input_ids": self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx]
            }

    test_loader = DataLoader(BlueskyDataset(test_encodings), batch_size=64)

    # Inference
    all_relevance = []
    all_disaster = []

    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Predicting Bluesky"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)

            all_relevance.extend((out["relevance"].squeeze() > 0.5).int().cpu().tolist())
            all_disaster.extend(torch.argmax(out["disaster_type"], dim=1).cpu().tolist())

    # Map predictions
    bluesky_df["Relevancy"] = all_relevance
    bluesky_df["category"] = [category_map[i] for i in all_disaster]
#    bluesky_df["location"] = bluesky_df.apply(
#        lambda row: extract_locations(row["cleaned_text"]) if row["Relevancy"] == 1 else [], axis=1
#    )

    print("Inference complete. Results saved.")

    relevant_posts = bluesky_df[bluesky_df["Relevancy"] == 1].copy()

    relevant_posts["location"] = relevant_posts["cleaned_text"].apply(extract_locations)

    print("Begin clustering...")

    # Vanessa's clustering
    ### MOVED TO END OF K-MEANS ALGO
    ### K-Means Clustering Function; Clusters similar disaster posts with each other so we can detect ongoing situations. Kinda simple for now, will update later.
    ##### UPDATE: Updated K-means with location and time; location is not filtered for states. I'll keep working on that and see if it can work.

    from sklearn.cluster import KMeans
    from scipy.sparse import hstack

    # TF-IDF Encoding Function
    def encoding(train_data, test_data):
        tfidf = TfidfVectorizer(
            ngram_range=(1, 1), use_idf=True, smooth_idf=True, sublinear_tf=True
        )
        tf_df_train = tfidf.fit_transform(train_data).toarray()
        train_df = pd.DataFrame(tf_df_train, columns=tfidf.get_feature_names_out())
        tf_df_test = tfidf.transform(test_data).toarray()
        test_df = pd.DataFrame(tf_df_test, columns=tfidf.get_feature_names_out())

        return train_df, test_df, tfidf
    
    disaster_posts = relevant_posts[relevant_posts['Relevancy'] == 1].copy()
    disaster_posts['original_index'] = disaster_posts.index

    relevant_posts['location'] = relevant_posts['location'].apply(
        lambda x: ', '.join(x) if isinstance(x, list) else x
    )

    from sklearn.preprocessing import OneHotEncoder

    disaster_posts['location'] = disaster_posts['location'].fillna("unknown").astype(str)
    location_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    location_encoded = location_encoder.fit_transform(disaster_posts[['location']])
    

    # Check if there are any disaster posts
    if not disaster_posts.empty:
        disaster_posts['text'] = disaster_posts['cleaned_text']

        # TF-IDF vectorization
        _, disaster_tfidf_matrix, tfidf = encoding(train["cleaned_text"], disaster_posts['text'])

        combined_features = hstack([
            disaster_tfidf_matrix,
            location_encoded
        ])

        # Cluster if enough data
        if combined_features.shape[0] > 1 and combined_features.shape[1] > 1:
            ### IMPORTANT: Change value to a lower number, preferably 100 or less
            num_clusters = max(1, disaster_posts.shape[0] // 10)

            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            kmeans.fit(combined_features)

            # Save original index BEFORE reset
            disaster_posts['cluster'] = kmeans.labels_
            relevant_posts.loc[disaster_posts['original_index'], 'cluster'] = disaster_posts['cluster'].value_counts

            # Initialize 'cluster'
            relevant_posts['cluster'] = None

            # Write the cluster labels into test at the right rows
            relevant_posts.loc[disaster_posts['original_index'], ['cluster']] = disaster_posts['cluster'].values
    
        else:
            print("Not enough data to perform clustering.")
    else:
        print("No disaster posts found for clustering.")
    
    ### Gets locations of clusters and their coordinates
    from geopy.extra.rate_limiter import RateLimiter

    relevant_posts['cluster'] = pd.to_numeric(relevant_posts['cluster'], errors='coerce')
    
    # Geocoding setup
    geolocator = Nominatim(user_agent="cluster_location_geocoder")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    location_cache = {}

    def get_coordinates(location):
        if location in location_cache:
            return location_cache[location]
        try:
            loc = geocode(location)
            if loc:
                coords = (loc.latitude, loc.longitude)
                location_cache[location] = coords
                return coords
        except:
            pass
        location_cache[location] = (np.nan, np.nan)
        return (np.nan, np.nan)
    
    # Get most common location per cluster
    cluster_location_mode = (
        relevant_posts[relevant_posts['cluster'].notnull()]
        .groupby('cluster')['location']
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "unknown")
        .reset_index()
        .rename(columns={"location": "Most_Common_Location"})
    )

    # Geocode locations
    cluster_location_mode['Coordinates'] = cluster_location_mode['Most_Common_Location'].apply(get_coordinates)

    cluster_location_mode['Coordinates'] = cluster_location_mode['Coordinates'].apply(
        lambda x: x if isinstance(x, (tuple, list)) and len(x) == 2 else (np.nan, np.nan)
    )
    cluster_location_mode['lat'] = cluster_location_mode['Coordinates'].apply(lambda x: x[0])
    cluster_location_mode['lon'] = cluster_location_mode['Coordinates'].apply(lambda x: x[1])

    # Filter through US coordinates
    us_clusters = cluster_location_mode[
        (cluster_location_mode['lat'].between(24, 49)) &
        (cluster_location_mode['lon'].between(-125, -66))
    ]

    lat_map = us_clusters.set_index('cluster')['lat'].to_dict()
    lon_map = us_clusters.set_index('cluster')['lon'].to_dict()

    # Save back to db
    relevant_posts['LAT'] = relevant_posts['cluster'].map(lat_map)
    relevant_posts['LON'] = relevant_posts['cluster'].map(lon_map)
    
    return relevant_posts
    #end run_model1

def main() -> None:
    try:
        train = load_data_train() # train data will always remain the same
        test = load_data_test() 
    except Exception as e:
        print(f"Error fetching posts: {e}")

    # run model1, Kenny's model
    relevant_posts = run_model1(train, test)
    
    # connect to Processed_Posts, then insert posts
    try:
        loadSuccess = send_data(relevant_posts)
        print(loadSuccess)                                  # prints when successfully inserted into Processed_Posts
    except Exception as e:
        print(f"Error inserting into Processed_Posts: {e}")

if __name__ == '__main__':
    main()

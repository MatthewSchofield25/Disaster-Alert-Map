import asyncio
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
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
n_epoch = 30

load_dotenv() #load the .env file
driver = '{ODBC Driver 18 for SQL Server}'
server = os.getenv("DATABASE_SERVER")
database = os.getenv("DATABASE_NAME")
username = os.getenv("DATABASE_USERNAME")
password = os.getenv("DATABASE_PASSWORD")

'''
# use to view environmental variables are fetched properly
print(f"DATABASE_SERVER: {server}")
print(f"DATABASE_NAME: {database}")
print(f"DATABASE_USERNAME: {username}")
print(f"DATABASE_PASSWORD: {password}")
'''

nlp = spacy.load("en_core_web_sm")

nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('wordnet')

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
        "Blizzard": ["blizzard", "snowstorm", "ice storm", "whiteout"]
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

def extract_location(text):
    doc = nlp(text)
    locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    return ", ".join(locations) if locations else "None"

#loading data, need to convert to pydobc
def load_data_train():
    train = pd.read_csv("train.csv")
    train.rename(columns={"Text": "text", "Label": "target"}, inplace=True)
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

    # calculate one hour ago
    last_hour = datetime.datetime.now() - datetime.timedelta(hours=1)
    sql_time = last_hour.strftime('%Y-%m-%d %H:%M:%S')  # format time for SQL server

    print(f"Fetching posts after time: {sql_time}")

    query = """
    SELECT * FROM Bluesky_Posts 
    WHERE timeposted >= ?
    """

    #cursor.execute("SELECT * FROM Bluesky_Posts")      # old query: fetches ALL posts
    cursor.execute(query, (sql_time,))
    
    results = cursor.fetchall()
    columns = [column[0] for column in cursor.description]
    test = pd.DataFrame.from_records(results, columns=columns)
    test.rename(columns={"post_text": "text"}, inplace=True)
    db.commit()
    cursor.close()
    db.close()
    return test

#loads Vanessa's model output into the table LSTM_Posts, 4_2
def send_LSTM_data(relevant_posts):
    try:
        conn = pyodbc.connect(f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}')
    except Exception as e:
        print(f"Error connecting to SQL Server: {e}")
        return None
    print("SQL server connection successful: connected to LSTM_Posts")
    cursor = conn.cursor()
    
    # valid categories for CATEGORY_CHECK constraint
    valid_categories = {'Flood', 'Drought', 'Landslide', 'Volcano', 'Blizzard',
                        'Earthquake', 'Tsunami', 'Wildfire', 'Hurricane', 'Tornado', 'Other'}
    
    for index, row in relevant_posts.iterrows():
        # validate current category
        if row.category not in valid_categories:
            print(f"Invalid category '{row.category}' for post_uri: {row.post_uri}. Skipping this row.")
            continue 
        
        try:
            cursor.execute(
                """
                INSERT INTO LSTM_Posts (
                    post_uri, post_author, post_author_display, post_text, timeposted,
                    sentiment_score, keyword, location, cleaned_text, category,
                    sentiment_label, prediction
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row.post_uri, row.post_author, row.post_author_display, row.text,
                row.timeposted, row.sentiment_score, row.keyword, row.location,
                row.cleaned_text, row.category, row.sentiment_label, row.prediction
            )
        except pyodbc.IntegrityError as dup_error:
            print(f"Duplicate entry skipped for post_uri: {row.post_uri}")
            continue  # handles duplicate posts
    conn.commit()
    cursor.close()
    conn.close()
    return "Data inserted into LSTM_Posts" 

# first model, Kenny's model
def run_model1(train, test):
    train["cleaned_text"] = train["text"].apply(preprocess_data)
    test["cleaned_text"] = test["text"].apply(preprocess_data)

    #sentiment analysis
    sia = SentimentIntensityAnalyzer()
    train["sentiment_score"] = train["text"].apply(lambda x: sia.polarity_scores(x)["compound"])
    test["sentiment_score"] = test["text"].apply(lambda x: sia.polarity_scores(x)["compound"])

    #detecting relevance
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_tfidf = vectorizer.fit_transform(train["cleaned_text"])
    y_tfidf = train["target"]

    X_train_tfidf, X_val_tfidf, y_train_tfidf, y_val_tfidf = train_test_split(X_tfidf, y_tfidf, test_size=0.2, random_state=42)
    relevance_model = LogisticRegression()
    relevance_model.fit(X_train_tfidf, y_train_tfidf)

    #evaluating
    y_val_pred = relevance_model.predict(X_val_tfidf)
    val_accuracy = accuracy_score(y_val_tfidf, y_val_pred)
    val_f1 = f1_score(y_val_tfidf, y_val_pred)

    print(f"Relevance Classifier Accuracy: {val_accuracy:.4f}")
    print(f"Relevance Classifier F1 Score: {val_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val_tfidf, y_val_pred))

    X_test_tfidf = vectorizer.transform(test["cleaned_text"])
    test_probs = relevance_model.predict_proba(X_test_tfidf)[:, 1]
    test["Relevant"] = (test_probs > 0.7).astype(int)

    #filter the relevant post
    relevant_posts = test[test["Relevant"] == 1].copy()

    #disaster categorization
    disaster_categories = {k: i for i, k in enumerate(list(disaster_keywords.keys()) + ["Other"])}
    train["category"] = train["cleaned_text"].apply(categorize_disaster)
    train["category_label"] = train["category"].map(disaster_categories)

    #LSTM prep
    MAX_NB_WORDS = 5000
    MAX_SEQUENCE_LENGTH = 100
    EMBEDDING_DIM = 100

    keras_tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True)
    keras_tokenizer.fit_on_texts(train["cleaned_text"])

    X_train_seq = keras_tokenizer.texts_to_sequences(train["cleaned_text"])
    X_test_seq = keras_tokenizer.texts_to_sequences(relevant_posts["cleaned_text"])

    X_train_seq = pad_sequences(X_train_seq, maxlen=MAX_SEQUENCE_LENGTH)
    X_test_seq = pad_sequences(X_test_seq, maxlen=MAX_SEQUENCE_LENGTH)

    y_train_cat = to_categorical(train["category_label"], num_classes=len(disaster_categories))

    #LSTM model
    model = Sequential([
        Embedding(input_dim=MAX_NB_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
        SpatialDropout1D(0.2),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(len(disaster_categories), activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    X_lstm_train, X_lstm_val, y_lstm_train, y_lstm_val = train_test_split(
        X_train_seq, y_train_cat, test_size=0.2, random_state=42
    )

    model.fit(X_lstm_train, y_lstm_train, epochs=5, batch_size=32, validation_data=(X_lstm_val, y_lstm_val))

    #predicting
    predictions = model.predict(X_test_seq)
    predicted_categories = predictions.argmax(axis=1)
    reverse_category_map = {v: k for k, v in disaster_categories.items()}
    relevant_posts["category"] = [reverse_category_map[i] for i in predicted_categories]
    relevant_posts["location"] = relevant_posts["text"].apply(extract_location)

    #final output
    relevant_posts.to_csv("Bluesky_Disaster_Predictions_With_Relevance.csv", index=False)

    return relevant_posts

# second model, Vanessa's model
def run_model2(relevant_posts):
    # Labels on the data
    #id : A unique identifier for each tweet.
    #keyword : A particular keyword from the tweet.
    #location: The location the tweet was sent from (may be blank).
    #text : The text of the tweet.
    #prediction : This denotes whether a tweet is about a real disaster (1) or not (0).

    common_words = ['via','like','build','get','would','one','two','feel','lol','fuck','take','way','may','first','latest'
                    'want','make','back','see','know','let','look','come','got','still','say','think','great','pleas','amp']

    train = pd.read_csv("train.csv")
    #test = pd.read_csv("test.csv")
        # test previously read a csv, now received as pd dataframe from KenLSTMModel.py

    ps = PorterStemmer()
    lm = WordNetLemmatizer()

    X = train.drop(columns=["target"],axis=1)
    y = train["target"]

    # Load the SpaCy NLP model
    nlp = spacy.load('en_core_web_sm')

    # Initialize the GeoPy geocoder
    geolocator = Nominatim(user_agent='project_app')

    def text_cleaning(data):
        return ' '.join(i for i in data.split() if i not in common_words)


    #fixme, changed preprocess_data to preproc_data: def preprocess_data(data):
    def preproc_data(data):
        '''
        Input: Data to be cleaned.
        Output: Cleaned Data.

        '''
        review =re.sub(r'https?://\S+|www\.\S+|http?://\S+',' ',data) #removal of url
        review =re.sub(r'<.*?>',' ',review) #removal of html tags
        review = re.sub("["
                            u"\U0001F600-\U0001F64F"  # removal of emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            "]+",' ',review)
        review = re.sub('[^a-zA-Z]',' ',review) # filtering out miscellaneous text.
        review = review.lower() # Lowering all the words in text
        review = review.split() # split into a list of words
        review = [lm.lemmatize(words) for words in review if words not in stopwords.words('english')] # Turn words into their stems/roots
        review = [i for i in review if len(i)>2] # Removal of words with length<2
        review = ' '.join(review) # Put back to single string with a space separator
        return review

    def sentiment_ana(data):
        sid_obj = SentimentIntensityAnalyzer()
        sentiment_dict = sid_obj.polarity_scores(data)
        return sentiment_dict['compound']

    def sentiment_ana_label(data):
        sid_obj = SentimentIntensityAnalyzer()
        sentiment_dict = sid_obj.polarity_scores(data)
        compound_score = sentiment_dict['compound']
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'

    # Define a function to extract location names from a text using SpaCy NER
    def extract_locations(text):
        doc = nlp(text)
        return [ent.text for ent in doc.ents if ent.label_ in ['LOC', 'GPE']]

    def top_ngrams(data,n,grams):

        if grams == 1:
            count_vec = CountVectorizer(ngram_range=(1,1)).fit(data)
            bow = count_vec.transform(data)
            add_words = bow.sum(axis=0)
            word_freq = [(word, add_words[0, idx]) for word, idx in count_vec.vocabulary_.items()]
            word_freq = sorted(word_freq, key = lambda x: x[1], reverse=True)
        elif grams == 2:
            count_vec = CountVectorizer(ngram_range=(2,2)).fit(data)
            bow = count_vec.transform(data)
            add_words = bow.sum(axis=0)
            word_freq = [(word,add_words[0,idx]) for word,idx in count_vec.vocabulary_.items()]
            word_freq = sorted(word_freq, key = lambda x: x[1], reverse=True)
        elif grams == 3:
            count_vec = CountVectorizer(ngram_range=(3,3)).fit(data)
            bow = count_vec.transform(data)
            add_words = bow.sum(axis=0)
            word_freq = [(word,add_words[0,idx]) for word,idx in count_vec.vocabulary_.items()]
            word_freq = sorted(word_freq, key = lambda x: x[1], reverse=True)

        return word_freq[:n]

    # Liz: renamed to lowercase variables, e.g.) Cleaned_text to cleaned_text
    # variable names should match the database columns
    train["cleaned_text"] = train["text"].apply(preproc_data)
    relevant_posts["cleaned_text"] = relevant_posts["text"].apply(preproc_data)

    train["cleaned_text"] = train["cleaned_text"].apply(text_cleaning)
    relevant_posts["cleaned_text"] = relevant_posts["cleaned_text"].apply(text_cleaning)

    train["sentiment_score"] = train["text"].apply(sentiment_ana)
    relevant_posts["sentiment_score"] = relevant_posts["text"].apply(sentiment_ana)

    train["sentiment_label"] = train["text"].apply(sentiment_ana_label)
    relevant_posts["sentiment_label"] = relevant_posts["text"].apply(sentiment_ana_label)

    train["location"] = train["text"].apply(extract_locations)

    common_words_uni = top_ngrams(train["cleaned_text"],20,1)
    common_words_bi = top_ngrams(train["cleaned_text"],20,2)
    common_words_tri = top_ngrams(train["cleaned_text"],20,3)

    print(common_words_uni)
    print(common_words_bi)
    print(common_words_tri)

    ### CAN DETECT DISASTER TYPE ### Not a predictive model ###

    # Array of disaster types (change later)
    #fixme: renamed to disaster_kwords      disaster_keywords = {
    disaster_kwords = {
        'earthquake': ['earthquake', '#earthquake'],
        'flood': ['flood', '#flood'],
        'fire': ['fire', '#fire'],
        'storm': ['storm', '#storm'],
        'hurricane': ['hurricane', '#hurricane'],
        'tornado': ['tornado', '#tornado'],
        'tsunami': ['tsunami', '#tsunami'],
        'wildfire': ['wildfire', '#wildfire'],
        'drought': ['drought', '#drought'],
        'avalanche': ['avalanche', '#avalanche'],
    }

    def get_disaster_type(text):
        """ Return the type of disaster based on the tweet's content """
        for disaster, keywords in disaster_kwords.items():
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    return disaster
        return 'other'  # Default if no match

    # Add a column for disaster type
    train['disaster_type'] = train['text'].apply(get_disaster_type)

    # Convert the disaster_type column to numeric labels for multi-class classification
    disaster_types = train['disaster_type'].unique()
    disaster_type_dict = {disaster: idx for idx, disaster in enumerate(disaster_types)}
    train['disaster_type_label'] = train['disaster_type'].map(disaster_type_dict)

    ### TF_IDF AND LSTM ###
    ## ACCURACY IS INCONSISTENT ##

    def encoding(train_data,test_data):
        tfidf = TfidfVectorizer(
            ngram_range=(1, 1), use_idf=True, smooth_idf=True, sublinear_tf=True
        )
        tf_df_train = tfidf.fit_transform(train_data).toarray()
        train_df = pd.DataFrame(tf_df_train,columns=tfidf.get_feature_names_out())
        tf_df_test = tfidf.transform(test_data).toarray()
        test_df = pd.DataFrame(tf_df_test,columns=tfidf.get_feature_names_out())

        return train_df,test_df

    x_final,x_test_final = encoding(train["cleaned_text"],relevant_posts["cleaned_text"])
    y_final = np.array(y)

    x_final.shape,y_final.shape,x_test_final.shape

    # Dividing the data into training, validation and testing
    x_train, x_test, y_train, y_test = train_test_split(x_final, y_final, test_size=0.1, random_state=42, stratify = y_final)
    X_train, x_valid, Y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=42, stratify = y_train)
    x_test_final = x_test_final

    embedding_feature_vector = 200 # Since we used glove vector embedding of dim 200.
    model = Sequential()
    model.add(Dense(512, input_dim=X_train.shape[1], activation='relu'))  # Input layer with TF-IDF features
    model.add(Dropout(0.35))  # Dropout layer for regularization
    model.add(Dense(128, activation='relu'))  # Hidden layer
    model.add(Dropout(0.35))  # Dropout layer for regularization
    model.add(Dense(32, activation='relu'))  # Hidden layer
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1,
                            mode='min', restore_best_weights=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5,
                                verbose=1, mode='min')

    history = model.fit(X_train,Y_train,validation_data=(x_valid,y_valid),callbacks=[reduce_lr,early_stop],epochs=n_epoch,batch_size= 64)

    predictions = model.predict(x_valid)

    # Convert probabilities to binary values (0 or 1)
    binary_predictions = (predictions > 0.5).astype(int)

    # Print the first 10 predictions
    print("First 10 predictions on validation set (1 = disaster, 0 = not disaster):")
    print(binary_predictions[:10])

    # Actual prediction probabilities (between 0 and 1)
    print("First 10 raw prediction probabilities:")
    print(predictions[:10])
    accuracy = accuracy_score(y_valid, binary_predictions)

    # Output the overall accuracy
    print(f"Validation accuracy: {accuracy * 100:.2f}%")
    predictions = model.predict(x_test_final)

    # Convert probabilities to binary values (0 or 1)
    binary_predictions = (predictions > 0.5).astype(int)
    relevant_posts['prediction'] = binary_predictions
    
    relevant_posts.to_csv('BlueSkyTestPredictions.csv', index=False)
    
    return relevant_posts    

def main() -> None:
    try:
        train = load_data_train() # train data will always remain the same
        test = load_data_test() 
    except Exception as e:
        print(f"Error fetching posts: {e}")
    
    # run model1, Kenny's model
    relevant_posts = run_model1(train, test)
    
    # model1 outputs and model2 receives:
        # post_uri, post_author, post_author_display, text, timeposted, sentiment_score, keyword, location, cleaned_text, category
    # model2 outputs:
        # post_uri, post_author, post_author_display, post_text, timeposted, sentiment_score, keyword, location, cleaned_text, category, sentiment_label, prediction	
        
    # run model2, Vanessa's model
    relevant_posts = run_model2(relevant_posts)
    
    # connect to LSTM_Posts, then insert posts. 4_2
    try:
        loadSuccess = send_LSTM_data(relevant_posts)
        print(loadSuccess)                                  # for debugging, prints when successfully inserted into LSTM_Posts
    except Exception as e:
        print(f"Error inserting into LSTM_Posts: {e}")

if __name__ == '__main__':
    main()

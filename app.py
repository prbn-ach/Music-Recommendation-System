from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import euclidean_distances
from functools import lru_cache
from typing import List, Dict, Tuple, Optional
import time
from sklearn.preprocessing import RobustScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Error handling
class SongNotFoundError(Exception):
    pass

@lru_cache(maxsize=1)
def load_dataset() -> pd.DataFrame:
    # Load and cache the dataset to prevent multiple reads.
    try:
        #dont forget to unzip the csv file
        return pd.read_csv("formatted.csv")
    except FileNotFoundError:
        logger.error("Dataset file 'formatted.csv' not found")
        raise

# Available features
features = [
    'valence', 'acousticness', 'danceability', 'energy',
    'instrumentalness', 'key', 'liveness', 'mode', 'speechiness','popularity'
]


song_inputs = ['name', 'artists']

def initialize_pipeline() -> Pipeline:
    return Pipeline([
        ('scaler', MinMaxScaler()),
        ('kmeans', KMeans(n_clusters=8, verbose=0))
    ], verbose=False)

@lru_cache(maxsize=1)
def get_trained_pipeline():
    # Train and cache the pipeline to prevent retraining.
    dataset = load_dataset()
    pipeline = initialize_pipeline()
    X = dataset[features]
    pipeline.fit(X)
    return pipeline

def take_input(song_list: List[Dict], dataset: pd.DataFrame) -> np.ndarray:
    song_vectors = []
    
    for song in song_list:
        try:
            song_data = dataset[
                (dataset['name'].str.lower() == song['name'].lower()) & 
                (dataset['artists'].str.lower() == song['artists'].lower())
            ].iloc[0]
            song_vectors.append(song_data[features].values)
        except IndexError:
            logger.warning(f"Song not found: {song['name']} by {song['artists']}")
            continue
            
    if not song_vectors:
        raise SongNotFoundError("None of the provided songs were found in the database")
            
    return np.mean(np.array(song_vectors), axis=0)


def Music_Recommender(song_list: List[Dict], dataset: pd.DataFrame, n_songs: int = 10) -> pd.DataFrame:

    start_time = time.time()
    
    try:
        # Get trained pipeline
        pipeline = get_trained_pipeline()
        
        # Process input
        song_center = take_input(song_list, dataset)
        
        # Scale data
        scaler = pipeline.named_steps['scaler']
        scaled_data = scaler.transform(dataset[features])
        scaled_song_center = scaler.transform(pd.DataFrame([song_center], columns=features))
        
        # Calculate distances and get recommendations
        distances = euclidean_distances(scaled_song_center, scaled_data)[0]
        indices = np.argsort(distances)[:n_songs * 2]  # Get extra songs to account for filtering
        
        # Filter recommendations
        rec_output = dataset.iloc[indices].copy()
        input_songs = {(song['name'].lower(), song['artists'].lower()) for song in song_list}
        
        rec_output = rec_output[~rec_output.apply(
            lambda x: (x['name'].lower(), x['artists'].lower()) in input_songs, axis=1
        )]
        
        # Remove duplicates and limit to n_songs
        rec_output = rec_output.drop_duplicates(subset=['name', 'artists']).head(n_songs)
        
        logger.info(f"Recommendations generated in {time.time() - start_time:.2f} seconds")
        return rec_output[song_inputs]
    
    except Exception as e:
        logger.error(f"Error in recommendation generation: {str(e)}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    """Handle recommendation requests."""
    try:
        song_name = request.form['song_name'].strip()
        artist_name = request.form.get('artist_name', '').strip()
        
        if not song_name:
            return render_template('index.html', error="Please enter a song name")
        
        song_list = [{'name': song_name, 'artists': artist_name or 'Unknown Artist'}]
        dataset = load_dataset()
        
        recommendations = Music_Recommender(song_list, dataset)
        return render_template('recommendations.html', 
                             recommendations=recommendations.to_dict(orient='records'))
    
    except SongNotFoundError:
        return render_template('index.html', 
                             error="Song not found in our database. Please try another song.")
    except Exception as e:
        logger.error(f"Error in recommend route: {str(e)}")
        return render_template('index.html', 
                             error="An error occurred while processing your request")

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    # Handle autocomplete requests.
    try:
        query = request.args.get('term', '').strip().lower()
        if not query:
            return jsonify([])
            
        dataset = load_dataset()
        filtered_songs = dataset[
            dataset['name'].str.lower().str.contains(query, na=False) |
            dataset['artists'].str.lower().str.contains(query, na=False)
        ]
        
        suggestions = (filtered_songs[['name', 'artists']]
                      .drop_duplicates()
                      .head(50)
                      .to_dict(orient='records'))
        
        return jsonify(suggestions)
        
    except Exception as e:
        logger.error(f"Error in autocomplete: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize the pipeline and dataset on startup
    try:
        load_dataset()
        get_trained_pipeline()
        app.run(debug=True)
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")

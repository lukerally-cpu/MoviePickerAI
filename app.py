
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import difflib

app = Flask(__name__)
CORS(app)

# Load ONLY the pre-calculated matrix
# Render will find this in your main folder
item_similarity_df = pd.read_pickle("movie_similarity.pkl")
valid_titles = item_similarity_df.index.tolist()

def find_closest_title(target_title):
    matches = difflib.get_close_matches(target_title, valid_titles, n=1, cutoff=0.6)
    return matches[0] if matches else None

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_input_ratings = data.get('ratings', [])
    recommendation_series = pd.Series(dtype='float64')
    watched_movies = []

    for movie_title, rating in user_input_ratings:
        actual_title = find_closest_title(movie_title)
        if actual_title:
            watched_movies.append(actual_title)
            weight = rating - 2.5
            similar_scores = item_similarity_df[actual_title] * weight
            recommendation_series = recommendation_series.add(similar_scores, fill_value=0)

    if recommendation_series.empty:
        return jsonify({"recommendations": []})

    recommendation_series = recommendation_series.drop(labels=watched_movies, errors='ignore')
    top_recs = recommendation_series.sort_values(ascending=False).head(10).index.tolist()
    return jsonify({"recommendations": top_recs})

if __name__ == '__main__':
    app.run(debug=True)

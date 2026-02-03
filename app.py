import os
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)  # This allows your Vercel frontend to talk to this API

# Load your AI model and data
# Make sure these files are in your main backend folder!
movies = pickle.load(open('movie_list.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

@app.route('/')
def home():
    return "Movie Picker AI Backend is Running!"

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_ratings = data.get('ratings', []) # List of [title, score]

    if not user_ratings:
        return jsonify({"recommendations": []})

    # Simple logic: Find recommendations based on the highest-rated movie
    # Sort by score descending and pick the top one
    user_ratings.sort(key=lambda x: x[1], reverse=True)
    base_movie = user_ratings[0][0]

    try:
        movie_index = movies[movies['title'] == base_movie].index[0]
        distances = similarity[movie_index]
        movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:11]

        recommendations = []
        for i in movie_list:
            recommendations.append(movies.iloc[i[0]].title)

        return jsonify({"recommendations": recommendations})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Render provides a PORT environment variable. 
    # If it doesn't exist (like on your local PC), it defaults to 5000.
    port = int(os.environ.get("PORT", 5000))
    
    # host='0.0.0.0' is REQUIRED for Render to route traffic to your app
    app.run(host='0.0.0.0', port=port)
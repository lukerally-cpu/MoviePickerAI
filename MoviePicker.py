import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

# Setup Paths to use the datasets in pandas
folder_path = r'ml-32m\ml-32m'
movies_path = os.path.join(folder_path, 'movies.csv')
ratings_path = os.path.join(folder_path, 'ratings.csv')

# 2. Load the datasets with a LIMIT
# We are loading the first 1,000,000 rows. 
# This is plenty for a prototype and won't crash your RAM.
print("Loading data...")
movies = pd.read_csv(movies_path)
ratings = pd.read_csv(ratings_path, nrows=1000000) 

print("Files loaded successfully!")

# 3. Merge and Filter
df = pd.merge(ratings, movies, on='movieId')

# Filter: Only movies with at least 100 ratings (higher threshold for large data)
movie_counts = df.groupby('title')['rating'].count()
popular_movies = movie_counts[movie_counts > 100].index
df = df[df['title'].isin(popular_movies)]

# 4. Create the Pivot Table
print("Building movie matrix (this may take a moment)...")
movie_matrix = df.pivot_table(index='title', columns='userId', values='rating').fillna(0)

# 5. Calculate Similarity
print("Calculating similarity scores...")
item_similarity = cosine_similarity(movie_matrix)
item_similarity_df = pd.DataFrame(item_similarity, index=movie_matrix.index, columns=movie_matrix.index)

# 6. Recommendation Function
def get_multi_recommendations(user_ratings):
    """
    user_ratings: a list of tuples like [("Toy Story (1995)", 5), ("Heat (1995)", 1)]
    """
    recommendation_series = pd.Series(dtype='float64')
    
    for movie_title, rating in user_ratings:
        if movie_title in item_similarity_df.columns:
            # Calculate similarity and weight it by the rating (centered at 2.5)
            weight = rating - 2.5
            similar_movies = item_similarity_df[movie_title] * weight
            
            # Add to our running total of scores
            recommendation_series = recommendation_series.add(similar_movies, fill_value=0)
        else:
            print(f"Skipping '{movie_title}': Not in the top filtered movies.")

    # Remove the movies the user has already watched/rated
    watched_movies = [title for title, rating in user_ratings]
    recommendation_series = recommendation_series.drop(labels=watched_movies, errors='ignore')
    
    return recommendation_series.sort_values(ascending=False)

# 7. Test
my_profile = [
    ('Toy Story (1995)', 5),
    ('Aladdin (1992)', 5),
    ('Heat (1995)', 1)
]

print("\n--- Personalized Recommendations for your Profile ---")
results = get_multi_recommendations(my_profile)
print(results.head(10))


# Search Function 
def search_movies(keyword):
    # This looks for the keyword in the 'title' column (case-insensitive)
    matches = movies[movies['title'].str.contains(keyword, case=False, na=False)]
    return matches[['title']].head(10)


item_similarity_df.to_pickle("movie_similarity.pkl")
print("Similarity matrix saved as movie_similarity.pkl")


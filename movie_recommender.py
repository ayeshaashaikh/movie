import ast
import pickle

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

ps = PorterStemmer()

# Load data
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Merge datasets
movies = movies.merge(credits, on='title')

# Select features
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'runtime', 'popularity', 'vote_average']]

# Drop rows with missing overview
movies.dropna(subset=['overview'], inplace=True)

# --- Feature Processing ---

def extract_names(text):
    try:
        return [i['name'] for i in ast.literal_eval(text)]
    except:
        return []

def get_director(text):
    try:
        for i in ast.literal_eval(text):
            if i['job'] == 'Director':
                return [i['name']]
    except:
        return []
    return []

movies['genres'] = movies['genres'].apply(extract_names)
movies['keywords'] = movies['keywords'].apply(extract_names)
movies['cast'] = movies['cast'].apply(lambda x: extract_names(x)[:3])  # Top 3 actors
movies['crew'] = movies['crew'].apply(get_director)

# Convert overview to list of words
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Numeric columns to tags
movies['runtime'] = movies['runtime'].apply(lambda x: [str(int(x)) + "min"] if not pd.isna(x) else [])
movies['popularity'] = movies['popularity'].apply(lambda x: [str(int(x)) + "_pop"] if not pd.isna(x) else [])
movies['vote_average'] = movies['vote_average'].apply(lambda x: [str(x) + "_rated"] if not pd.isna(x) else [])

# Combine everything into 'tags'
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + \
                 movies['cast'] + movies['crew'] + movies['runtime'] + \
                 movies['popularity'] + movies['vote_average']

new_df = movies[['movie_id', 'title', 'tags']]

# Convert list to string
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

# Text Preprocessing
def preprocess(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    return " ".join([ps.stem(word) for word in text if word.isalnum()])

new_df['tags'] = new_df['tags'].apply(preprocess)

# Vectorization - Count
cv = CountVectorizer(max_features=7000, stop_words='english')
count_vectors = cv.fit_transform(new_df['tags']).toarray()

# Vectorization - TF-IDF
tfidf = TfidfVectorizer(max_features=7000, stop_words='english')
tfidf_vectors = tfidf.fit_transform(new_df['tags']).toarray()

# Cosine Similarity
similarity_count = cosine_similarity(count_vectors)
similarity_tfidf = cosine_similarity(tfidf_vectors)

# Save models
pickle.dump(new_df, open('movies.pkl', 'wb'))
pickle.dump(similarity_count, open('similarity_count.pkl', 'wb'))
pickle.dump(similarity_tfidf, open('similarity_tfidf.pkl', 'wb'))

print("✅ Models saved with enhanced features.")

# --- Visualization Section ---

# Genre frequency
all_genres = sum(movies['genres'], [])
genre_counts = pd.Series(all_genres).value_counts().head(10)
plt.figure(figsize=(8,5))
sns.barplot(x=genre_counts.values, y=genre_counts.index, palette='viridis')
plt.title("Top 10 Genres")
plt.xlabel("Number of Movies")
plt.savefig("top_genres.png")
plt.close()

# Top Actors
all_actors = sum(movies['cast'], [])
actor_counts = pd.Series(all_actors).value_counts().head(10)
plt.figure(figsize=(8,5))
sns.barplot(x=actor_counts.values, y=actor_counts.index, palette='magma')
plt.title("Top 10 Actors (appearing)")
plt.xlabel("Appearances")
plt.savefig("top_actors.png")
plt.close()

# Heatmap of similarity (just 20x20 for visualization)
plt.figure(figsize=(10,8))
sns.heatmap(similarity_count[:20,:20], cmap='coolwarm')
plt.title("Movie Similarity (CountVectorizer - Sample)")
plt.savefig("similarity_heatmap.png")
plt.close()
